import os
import re
import zarr
import math
import argparse
import numpy as np
from glob import glob
from PIL import Image
from datasets import Dataset, Image as HFImage, Value, Features

# ---------------- CONFIGURATION ------------------------------------------------------------
NUM_PROC = 128                         # adjust based on available cpu cores
WRITER_BATCH_SIZE = 10                 # number of rows per write op for the .map() cache file writer
MAX_DIM_SIZE = 2048                    # Maximum size of any dimension before chunking
LOWER_PERCENTILE = 0.1                 # Lower percentile for intensity normalization (e.g., 1.0 or 0.1)
UPPER_PERCENTILE = 99.9                # Upper percentile for intensity normalization (e.g., 99.0 or 99.9)
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

def normalize_to_uint8(slice_2d):
    """Safely casts heterogenous EM arrays to standard 8-bit grayscale with a black background."""
    
    # 1. Drop completely uniform slices early
    if slice_2d.min() == slice_2d.max():
        return None
        
    # 2. Bypass for uint8
    if slice_2d.dtype == np.uint8:
        return slice_2d

    # 3. Convert to float for safe math
    slice_float = slice_2d.astype(np.float32)
    
    # 5. Calculate percentiles for contrast stretching
    p_low, p_high = np.percentile(slice_float, (LOWER_PERCENTILE, UPPER_PERCENTILE))

    # 6. Fallback if percentiles are identical
    if p_high - p_low == 0:
        p_low, p_high = slice_float.min(), slice_float.max()
        if p_high - p_low == 0:
            return None

    # 7. Normalize to 0.0 - 1.0 range
    normalized = np.clip((slice_float - p_low) / (p_high - p_low), 0, 1)
    
    # 8. INVERT: Map the bright transmissive background (1.0) to black (0.0) 
    # TODO: not sure about when exactly this is needed
    if slice_2d.dtype == np.int16:
        normalized = 1.0 - normalized
    
    # 9. Scale to 255 and cast back to uint8
    return (normalized * 255.0).astype(np.uint8)

def process_slice_batch(batch):
    """Worker function executed independently by each CPU core, processing batches of slices."""
    out = {
        "image": [],
        "volume_name": [],
        "axis": [],
        "slice": [],
        "part_id": []
    }
    
    for i in range(len(batch['zarr_path'])):
        zarr_path = batch['zarr_path'][i]
        s0_path = batch['s0_path'][i]
        axis = batch['axis'][i]
        slice_idx = batch['slice'][i]
        r_start = batch['r_start'][i]
        r_end = batch['r_end'][i]
        c_start = batch['c_start'][i]
        c_end = batch['c_end'][i]
        volume_name = batch['volume_name'][i]
        part_id = batch['part_id'][i]

        if WORKER_CACHE.get("current_zarr") != zarr_path:
            WORKER_CACHE.clear()  
            WORKER_CACHE["root"] = zarr.open(zarr_path, mode='r')
            WORKER_CACHE["current_zarr"] = zarr_path
            
        zarr_root = WORKER_CACHE["root"]
        current_array = zarr_root[s0_path]
        
        # Read only the requested bounding box directly from Zarr
        if axis == 'z': 
            slice_2d = current_array[slice_idx, r_start:r_end, c_start:c_end]
        elif axis == 'y': 
            slice_2d = current_array[r_start:r_end, slice_idx, c_start:c_end]
        elif axis == 'x': 
            slice_2d = current_array[r_start:r_end, c_start:c_end, slice_idx]
        else:
            raise ValueError(f"Unknown axis: {axis}")

        slice_2d = np.array(slice_2d)

        # --- CROPPING LOGIC ---        
        # Foreground is anything that isn't black AND isn't the white padding
        bg_white = slice_2d.max()
        is_foreground = (slice_2d != 0) & (slice_2d != bg_white)

        # Find bounding box of non-background regions
        rows = np.any(is_foreground, axis=1)
        cols = np.any(is_foreground, axis=0)
        
        if not np.any(rows):
            continue  # Skip completely blank or uniformly padded slices
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop down to bounding box
        slice_2d = slice_2d[rmin:rmax+1, cmin:cmax+1]
        # ------------------------------------

        slice_2d_uint8 = normalize_to_uint8(slice_2d)

        if slice_2d_uint8 is None:
            continue  # Skip uniform/featureless slices entirely

        # --- CROPPING LOGIC ---        
        # Foreground is anything that isn't black AND isn't the white padding
        is_foreground = (slice_2d_uint8 != 0) & (slice_2d_uint8 != 255)

        # Find bounding box of non-background regions
        rows = np.any(is_foreground, axis=1)
        cols = np.any(is_foreground, axis=0)
        
        if not np.any(rows):
            continue  # Skip completely blank or uniformly padded slices
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop down to bounding box
        slice_2d_uint8 = slice_2d_uint8[rmin:rmax+1, cmin:cmax+1]
        # ------------------------------------

        out["image"].append(Image.fromarray(slice_2d_uint8))
        out["volume_name"].append(volume_name)
        out["axis"].append(axis)
        out["slice"].append(slice_idx)
        out["part_id"].append(part_id)

    return out

def build_task_list(root_dir, stride=1):
    """Scans the dataset metadata instantly and builds a master list of all slices and chunks."""
    zarr_paths = sorted(glob(os.path.join(root_dir, '*/*.zarr')))
    all_tasks = []
    axis_names = {0: 'z', 1: 'y', 2: 'x'}

    print(f"Scanning Zarr volumes and building the task queue (stride={stride})...")
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

        volume_identifier = f"{dataset_name}/{earliest_recon}/{best_em}"
        shape = zarr_root[s0_path].shape
        
        for axis in [0, 1, 2]:
            # Determine the 2D dimensions based on the slicing axis
            if axis == 0:   # Z-axis
                h, w = shape[1], shape[2]
            elif axis == 1: # Y-axis
                h, w = shape[0], shape[2]
            elif axis == 2: # X-axis
                h, w = shape[0], shape[1]

            # Calculate chunks required to stay under the MAX_DIM_SIZE
            h_chunks = max(1, math.ceil(h / MAX_DIM_SIZE))
            w_chunks = max(1, math.ceil(w / MAX_DIM_SIZE))
            
            # Calculate the step sizes for this specific slice to split equally
            h_step = math.ceil(h / h_chunks)
            w_step = math.ceil(w / w_chunks)

            # Apply the stride when iterating over the slices
            for slice_idx in range(0, shape[axis], stride):
                # Create a task for each grid section of the slice
                for i in range(h_chunks):
                    for j in range(w_chunks):
                        r_start = i * h_step
                        r_end = min((i + 1) * h_step, h)
                        c_start = j * w_step
                        c_end = min((j + 1) * w_step, w)

                        all_tasks.append({
                            "zarr_path": zarr_path,
                            "s0_path": s0_path,
                            "volume_name": dataset_name,
                            "axis": axis_names[axis],
                            "slice": slice_idx,
                            "part_id": f"{i}_{j}",  # unique grid ID
                            "r_start": r_start,
                            "r_end": r_end,
                            "c_start": c_start,
                            "c_end": c_end
                        })
                
    return all_tasks

def main():
    parser = argparse.ArgumentParser(description="Process zarr dataset in chunks.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data", help="Root directory for zarr volumes")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo_filtered_more", help="Local path to save dataset parts")

    # Chunking and sampling arguments
    parser.add_argument("--total_parts", type=int, default=100, help="Total number of chunks to divide the dataset into")
    parser.add_argument("--part_index", type=int, default=0, help="The 0-indexed part to process (e.g., 0 to 999)")
    parser.add_argument("--slice_stride", type=int, default=15, help="Take every K-th slice along each axis (1 means all slices)")
    
    args = parser.parse_args()
    print(f"Args: {args}")

    tasks = build_task_list(args.root_directory, stride=args.slice_stride)
    total_tasks = len(tasks)
    
    if total_tasks == 0:
        print("No tasks generated. Check your directory or stride settings.")
        return

    # Chunking
    chunk_size = math.ceil(total_tasks / args.total_parts)
    start_idx = args.part_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_tasks)
    
    chunked_tasks = tasks[start_idx:end_idx]
    print(f"Total parts across all volumes (with stride {args.slice_stride}): {total_tasks}")
    print(f"Processing Array Job Part {args.part_index + 1} of {args.total_parts}")
    print(f"Extracting sub-tasks {start_idx} to {end_idx - 1} ({len(chunked_tasks)} tasks)")

    if len(chunked_tasks) == 0:
        print(f"No tasks assigned to Part {args.part_index}. Exiting.")
        return

    # Pass only the chunked list
    dataset = Dataset.from_list(chunked_tasks)

    # Updated schema with the new part_id column
    final_features = Features({
        "image": HFImage(),
        "volume_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32"),
        "part_id": Value("string")
    })

    print(f"Igniting parallel extraction with {NUM_PROC} workers...")
    dataset = dataset.map(
        process_slice_batch,
        batched=True,
        batch_size=10,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names, 
        features=final_features,
        desc=f"Processing Part {args.part_index}",
        writer_batch_size=WRITER_BATCH_SIZE
    )
    dataset = dataset.shuffle(seed=42)
    
    # Define the local path for this specific part
    part_dir = os.path.join(args.local_save_dir, f"part_{args.part_index}")

    print(f"Saving part {args.part_index} locally to {part_dir}...")
    dataset.save_to_disk(part_dir, max_shard_size="1GB")
    print(f"Part {args.part_index} completed and saved to disk!")

if __name__ == "__main__":
    main()