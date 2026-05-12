import os
import re
import gc
import zarr
import math
import argparse
import multiprocess
import numpy as np
from glob import glob
from datasets import Dataset, Value, Features, Sequence

# ---------------- CONFIGURATION ------------------------------------------------------------
NUM_PROC = 16                          # Number of parallel processes
MAP_BATCH_SIZE = 1                     # Batch_size for map processing
WRITER_BATCH_SIZE = 16                  # Number of rows per write op for the .map() cache file writer
LOWER_PERCENTILE = 1.0                 # Lower percentile for intensity normalization
UPPER_PERCENTILE = 99.0                # Upper percentile for intensity normalization
VOLUMES_TO_BE_INVERTED = [
    "jrc_ccl81-covid-1", "jrc_fly-acc-calyx-1", "jrc_fly-fsb-1", "jrc_hela-4", "jrc_hela-22",
    "jrc_hela-h89-1", "jrc_hela-h89-2", "jrc_hela-nz-1", "jrc_hela-nz-2", "jrc_mus-cerebellum-4",
    "jrc_mus-cerebellum-5", "jrc_mus-cortex-3", "jrc_mus-dorsal-striatum-2", "jrc_mus-dorsal-striatum",
    "jrc_mus-granule-neurons-1", "jrc_mus-granule-neurons-2", "jrc_mus-granule-neurons-3",
    "jrc_mus-hippocampus-2", "jrc_mus-hippocampus-3", "jrc_mus-nacc-2", "jrc_mus-nacc-3",
    "jrc_mus-nacc-4", "jrc_mus-pancreas-3"
]
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

def normalize_to_uint8(vol_3d, volume_name):
    """Safely casts heterogenous EM arrays to standard 8-bit grayscale with a black background."""
    # Drop completely uniform volumes early
    if vol_3d.min() == vol_3d.max():
        return None
        
    # Convert to float for safe math
    vol_float = vol_3d.astype(np.float32)
    
    # Identify background padding values dynamically
    bg_black = vol_float.min()
    bg_white = vol_float.max()
    
    # Create a 1D array of only the valid tissue pixels
    valid_mask = (vol_float != bg_black) & (vol_float != bg_white) & (vol_float != 0.0)
    valid_pixels = vol_float[valid_mask]
    
    # Fallback if the volume is entirely padding
    if valid_pixels.size == 0:
        return None
    
    # Calculate percentiles ONLY on the valid tissue pixels
    p_low, p_high = np.percentile(valid_pixels, (LOWER_PERCENTILE, UPPER_PERCENTILE))

    # Fallback if percentiles are identical
    if p_high - p_low == 0:
        p_low, p_high = valid_pixels.min(), valid_pixels.max()
        if p_high - p_low == 0:
            return None
            
    # Normalize the ENTIRE volume (including padding) to 0.0 - 1.0 range
    normalized = np.clip((vol_float - p_low) / (p_high - p_low), 0, 1)
    
    # Invert brightness 
    if volume_name in VOLUMES_TO_BE_INVERTED:
        normalized = 1.0 - normalized
    
    # Scale to 255 and cast back to uint8
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

        if WORKER_CACHE.get("current_zarr") != zarr_path:
            WORKER_CACHE.clear()  
            WORKER_CACHE["root"] = zarr.open(zarr_path, mode='r')
            WORKER_CACHE["current_zarr"] = zarr_path
            
        zarr_root = WORKER_CACHE["root"]
        current_array = zarr_root[s0_path]
        
        # Read only the requested 3D bounding box directly from Zarr
        vol_3d = current_array[z_start:z_end, y_start:y_end, x_start:x_end]
        vol_3d = np.array(vol_3d)

        # --- PRE-CROP TO MINIMAL BOUNDING BOX ---
        # Find the background values for this specific raw volume
        bg_min = vol_3d.min()
        bg_max = vol_3d.max()
        
        # Mask out exact min, max, and Zarr zero-padding to find the bounding box
        is_foreground = (vol_3d != bg_min) & (vol_3d != bg_max) & (vol_3d != 0)

        z_any = np.any(is_foreground, axis=(1, 2))
        if not np.any(z_any):
            continue  # Skip completely blank subvolumes
            
        y_any = np.any(is_foreground, axis=(0, 2))
        x_any = np.any(is_foreground, axis=(0, 1))
        
        zmin, zmax = np.where(z_any)[0][[0, -1]]
        ymin, ymax = np.where(y_any)[0][[0, -1]]
        xmin, xmax = np.where(x_any)[0][[0, -1]]
        
        # Crop down to the minimal bounding box around non-zero pixels
        vol_3d = vol_3d[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        
        # --- NORMALIZE ---
        vol_3d_uint8 = normalize_to_uint8(vol_3d, volume_name)

        if vol_3d_uint8 is None:
            continue  # Skip uniform/featureless subvolumes entirely

        # --- POST CROP TO MINIMAL BOUNDING BOX---
        is_foreground = (vol_3d_uint8 != 0) & (vol_3d_uint8 != 255)
        z_any = np.any(is_foreground, axis=(1, 2))
        if not np.any(z_any):
            continue  # Skip completely blank or uniformly padded subvolumes
            
        y_any = np.any(is_foreground, axis=(0, 2))
        x_any = np.any(is_foreground, axis=(0, 1))
        
        zmin, zmax = np.where(z_any)[0][[0, -1]]
        ymin, ymax = np.where(y_any)[0][[0, -1]]
        xmin, xmax = np.where(x_any)[0][[0, -1]]
        
        vol_3d_uint8 = vol_3d_uint8[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
        
        if vol_3d_uint8.shape[0] < 64 or vol_3d_uint8.shape[1] < 64 or vol_3d_uint8.shape[2] < 64:
            continue  # Skip small subvolumes

        # Store as raw bytes and keep shape to reconstruct
        # To reconstruct in your dataloader: np.frombuffer(row["volume"], dtype=np.uint8).reshape(row["shape"])
        out["volume"].append(vol_3d_uint8.tobytes())
        out["shape"].append(list(vol_3d_uint8.shape))
        
        out["volume_name"].append(volume_name)
        out["z_start"].append(z_start)
        out["y_start"].append(y_start)
        out["x_start"].append(x_start)

        # Force garbage collection
        gc.collect()
        
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
                    })
                
    return all_tasks

def main():
    parser = argparse.ArgumentParser(description="Process zarr dataset in 3D chunks.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data", help="Root directory for zarr volumes")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo_3d", help="Local path to save dataset parts")

    # Chunking and sampling arguments
    parser.add_argument("--total_parts", type=int, default=1000, help="Total number of chunks to divide the dataset into")
    parser.add_argument("--part_index", type=int, default=0, help="The 0-indexed part to process (e.g., 0 to 99)")
    parser.add_argument("--crop_size", type=int, default=512, help="The edge dimension size of the 3D crops")
    parser.add_argument("--stride", type=int, default=256, help="Stride size when iterating over dimensions")
    
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
    })

    print(f"Igniting parallel extraction with {NUM_PROC} workers...")
    dataset = dataset.map(
        process_volume_batch,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names, 
        features=final_features,
        desc=f"Processing Part {args.part_index}",
        writer_batch_size=WRITER_BATCH_SIZE
    )
    
    part_dir = os.path.join(args.local_save_dir, f"part_{args.part_index}")

    print(f"Saving part {args.part_index} locally to {part_dir}...")
    dataset.save_to_disk(part_dir, max_shard_size="2GB")
    print(f"Part {args.part_index} completed and saved to disk!")

    print("Cleaning up HF dataset cache files to prevent disk overflow...")
    cleaned_count = dataset.cleanup_cache_files()
    print(f"Cleaned up {cleaned_count} cache files.")
    
    print("Terminating any lingering multiprocess workers...")
    for p in multiprocess.active_children():
        p.terminate()
        p.join()

if __name__ == "__main__":
    main()