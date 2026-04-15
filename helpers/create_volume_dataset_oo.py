import os
import re
import zarr
import math
import argparse
import numpy as np
from glob import glob
from datasets import Dataset, Value, Features, Sequence

# ---------------- CONFIGURATION ------------------------------------------------------------
NUM_PROC = 64                          # Reduced from 128 to 64 to save memory while holding 3D volumes
WRITER_BATCH_SIZE = 10                 # number of rows per write op for the .map() cache file writer
LOWER_PERCENTILE = 0.1                 # Lower percentile for intensity normalization
UPPER_PERCENTILE = 99.9                # Upper percentile for intensity normalization
# -------------------------------------------------------------------------------------------

# Global cache dictionary specifically for the isolated worker processes
WORKER_CACHE = {}

def get_recon_sort_key(recon_name):
    match = re.search(r'recon-(\d+)', recon_name)
    return int(match.group(1)) if match else float('inf')

def get_em_subfolder_sort_key(folder_name):
    if folder_name.endswith('-uint8'): return 0
    elif folder_name.endswith('-uint8_1'): return 1
    elif folder_name.endswith('-uint16'): return 2
    elif folder_name.endswith('-int16'): return 3
    return 4

def normalize_to_uint8(vol_3d):
    """Safely casts heterogenous EM arrays to standard 8-bit grayscale."""
    if vol_3d.dtype == np.uint8:
        if vol_3d.min() == vol_3d.max():
            return None  # Volume is completely uniform, return None to drop it
        return vol_3d

    vol_float = vol_3d.astype(np.float32)
    
    p_low, p_high = np.percentile(vol_float, (LOWER_PERCENTILE, UPPER_PERCENTILE))

    if p_high - p_low == 0:
        # Fallback to absolute min/max if 1st and 99th percentiles are identical
        p_low, p_high = vol_float.min(), vol_float.max()
        if p_high - p_low == 0:
            return None  # Volume is completely uniform, return None to drop it

    normalized = np.clip((vol_float - p_low) / (p_high - p_low), 0, 1)
    return (normalized * 255.0).astype(np.uint8)

def process_volume_batch(batch):
    """Worker function executed independently by each CPU core, processing batches of 3D subvolumes."""
    out = {
        "volume": [],      # Storing as bytes to efficiently support variable sized ND arrays in HF Datasets
        "shape": [],       # Required to reconstruct the bytes into a numpy array
        "volume_name": [],
        "z_start": [],
        "y_start": [],
        "x_start": [],
        "part_id": []
    }
    
    for i in range(len(batch['zarr_path'])):
        zarr_path = batch['zarr_path'][i]
        s0_path = batch['s0_path'][i]
        z_start = batch['z_start'][i]
        z_end = batch['z_end'][i]
        y_start = batch['y_start'][i]
        y_end = batch['y_end'][i]
        x_start = batch['x_start'][i]
        x_end = batch['x_end'][i]
        volume_name = batch['volume_name'][i]
        part_id = batch['part_id'][i]

        if WORKER_CACHE.get("current_zarr") != zarr_path:
            WORKER_CACHE.clear()  
            WORKER_CACHE["root"] = zarr.open(zarr_path, mode='r')
            WORKER_CACHE["current_zarr"] = zarr_path
            
        zarr_root = WORKER_CACHE["root"]
        current_array = zarr_root[s0_path]
        
        # Read only the requested 3D bounding box directly from Zarr
        vol_3d = current_array[z_start:z_end, y_start:y_end, x_start:x_end]
        vol_3d = np.array(vol_3d)

        # Find bounding box of non-zero regions along 3 dimensions
        z_any = np.any(vol_3d != 0, axis=(1, 2))
        if not np.any(z_any):
            continue  # Skip completely blank subvolumes
            
        y_any = np.any(vol_3d != 0, axis=(0, 2))
        x_any = np.any(vol_3d != 0, axis=(0, 1))
        
        zmin, zmax = np.where(z_any)[0][[0, -1]]
        ymin, ymax = np.where(y_any)[0][[0, -1]]
        xmin, xmax = np.where(x_any)[0][[0, -1]]
        
        # Crop down to the minimal bounding box around non-zero pixels
        vol_3d_cropped = vol_3d[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        
        vol_3d_uint8 = normalize_to_uint8(vol_3d_cropped)

        if vol_3d_uint8 is None:
            continue  # Skip uniform/featureless subvolumes entirely

        # Store as raw bytes and keep shape to reconstruct
        # To reconstruct in your dataloader: np.frombuffer(row["volume"], dtype=np.uint8).reshape(row["shape"])
        out["volume"].append(vol_3d_uint8.tobytes())
        out["shape"].append(list(vol_3d_uint8.shape))
        
        out["volume_name"].append(volume_name)
        out["z_start"].append(z_start)
        out["y_start"].append(y_start)
        out["x_start"].append(x_start)
        out["part_id"].append(part_id)

    return out

def build_task_list(root_dir, crop_size=512, stride=512):
    """Scans the dataset metadata instantly and builds a master list of all 3D chunks."""
    zarr_paths = sorted(glob(os.path.join(root_dir, '*/*.zarr')))
    all_tasks = []

    print(f"Scanning Zarr volumes and building the task queue (crop_size={crop_size}, stride={stride})...")
    for zarr_path in zarr_paths:
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')

        try:
            zarr_root = zarr.open(zarr_path, mode='r')
        except Exception:
            continue

        recon_keys = [k for k in zarr_root.keys() if k.startswith('recon-')]
        if not recon_keys: continue
        
        earliest_recon = sorted(recon_keys, key=get_recon_sort_key)[0]
        em_path = f"{earliest_recon}/em"
        if em_path not in zarr_root: continue

        em_subfolders = list(zarr_root[em_path].keys())
        if not em_subfolders: continue

        best_em = sorted(em_subfolders, key=get_em_subfolder_sort_key)[0]
        s0_path = f"{em_path}/{best_em}/s0"
        if s0_path not in zarr_root: continue

        shape = zarr_root[s0_path].shape
        
        # Subdivide volume into grids of 3D crops
        for z in range(0, shape[0], stride):
            for y in range(0, shape[1], stride):
                for x in range(0, shape[2], stride):
                    z_end = min(z + crop_size, shape[0])
                    y_end = min(y + crop_size, shape[1])
                    x_end = min(x + crop_size, shape[2])
                    
                    if z_end <= z or y_end <= y or x_end <= x:
                        continue
                        
                    all_tasks.append({
                        "zarr_path": zarr_path,
                        "s0_path": s0_path,
                        "volume_name": dataset_name,
                        "z_start": z,
                        "z_end": z_end,
                        "y_start": y,
                        "y_end": y_end,
                        "x_start": x,
                        "x_end": x_end,
                        "part_id": f"{z}_{y}_{x}"
                    })
                
    return all_tasks

def main():
    parser = argparse.ArgumentParser(description="Process zarr dataset in 3D chunks.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data", help="Root directory for zarr volumes")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo_3d", help="Local path to save dataset parts")

    # Chunking and sampling arguments
    parser.add_argument("--total_parts", type=int, default=100, help="Total number of chunks to divide the dataset into")
    parser.add_argument("--part_index", type=int, default=0, help="The 0-indexed part to process (e.g., 0 to 99)")
    parser.add_argument("--crop_size", type=int, default=512, help="The edge dimension size of the 3D crops")
    parser.add_argument("--stride", type=int, default=512, help="Stride size when iterating over dimensions")
    
    args = parser.parse_args()
    print(f"Args: {args}")

    tasks = build_task_list(args.root_directory, crop_size=args.crop_size, stride=args.stride)
    total_tasks = len(tasks)
    
    if total_tasks == 0:
        print("No tasks generated. Check your directory or stride settings.")
        return

    chunk_size = math.ceil(total_tasks / args.total_parts)
    start_idx = args.part_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_tasks)
    
    chunked_tasks = tasks[start_idx:end_idx]
    print(f"Total parts across all volumes: {total_tasks}")
    print(f"Processing Array Job Part {args.part_index + 1} of {args.total_parts}")
    print(f"Extracting sub-tasks {start_idx} to {end_idx - 1} ({len(chunked_tasks)} tasks)")

    if len(chunked_tasks) == 0:
        print(f"No tasks assigned to Part {args.part_index}. Exiting.")
        return

    dataset = Dataset.from_list(chunked_tasks)

    # Store the 3D volume as binary representation to allow for variable cropped shapes and 
    # avoid standard Hugging Face sequences performance drop on huge nested lists.
    final_features = Features({
        "volume": Value("binary"),
        "shape": Sequence(Value("int32")),
        "volume_name": Value("string"),
        "z_start": Value("int32"),
        "y_start": Value("int32"),
        "x_start": Value("int32"),
        "part_id": Value("string")
    })

    print(f"Igniting parallel extraction with {NUM_PROC} workers...")
    dataset = dataset.map(
        process_volume_batch,
        batched=True,
        batch_size=10,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names, 
        features=final_features,
        desc=f"Processing Part {args.part_index}",
        writer_batch_size=WRITER_BATCH_SIZE
    )
    dataset = dataset.shuffle(seed=42)
    
    part_dir = os.path.join(args.local_save_dir, f"part_{args.part_index}")

    print(f"Saving part {args.part_index} locally to {part_dir}...")
    dataset.save_to_disk(part_dir, max_shard_size="2GB")
    print(f"Part {args.part_index} completed and saved to disk!")

if __name__ == "__main__":
    main()