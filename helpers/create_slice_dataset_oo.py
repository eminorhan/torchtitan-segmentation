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
NUM_PROC = 128                          # adjust based on available cpu cores
WORKER_CACHE_SIZE = 1024 * 1024 * 1024  # keep a small cache per worker to prevent I/O thrashing
WRITER_BATCH_SIZE = 100                 # number of rows per write op for the .map() cache file writer
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
    """Safely casts heterogenous EM arrays to standard 8-bit grayscale."""
    if slice_2d.dtype == np.uint8:
        return slice_2d

    slice_float = slice_2d.astype(np.float32)
    p_low, p_high = np.percentile(slice_float, (1, 99))

    if p_high - p_low == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    normalized = np.clip((slice_float - p_low) / (p_high - p_low), 0, 1)
    return (normalized * 255.0).astype(np.uint8)

def process_slice_worker(example):
    """Worker function executed independently by each CPU core."""
    zarr_path = example['zarr_path']
    s0_path = example['s0_path']
    axis = example['axis']
    slice_idx = example['slice']

    if WORKER_CACHE.get("current_zarr") != zarr_path:
        WORKER_CACHE.clear()  
        store = zarr.DirectoryStore(zarr_path)
        cache = zarr.LRUStoreCache(store, max_size=WORKER_CACHE_SIZE) 
        WORKER_CACHE["root"] = zarr.open(store=cache, mode='r')
        WORKER_CACHE["current_zarr"] = zarr_path
        
    zarr_root = WORKER_CACHE["root"]
    current_array = zarr_root[s0_path]
    
    if axis == 'z': 
        slice_2d = current_array[slice_idx, :, :]
    elif axis == 'y': 
        slice_2d = current_array[:, slice_idx, :]
    elif axis == 'x': 
        slice_2d = current_array[:, :, slice_idx]
    else:
        raise ValueError(f"Unknown axis: {axis}")

    slice_2d = np.array(slice_2d)
    slice_2d_uint8 = normalize_to_uint8(slice_2d)

    return {"image": Image.fromarray(slice_2d_uint8)}

def build_task_list(root_dir):
    """Scans the dataset metadata instantly and builds a master list of all slices."""
    zarr_paths = sorted(glob(os.path.join(root_dir, '*/*.zarr')))
    all_tasks = []
    axis_names = {0: 'z', 1: 'y', 2: 'x'}

    print("Scanning Zarr volumes and building the task queue...")
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
        s0_path = f"{em_path}/{best_em}/s1"
        if s0_path not in zarr_root: continue

        volume_identifier = f"{dataset_name}/{earliest_recon}/{best_em}"
        shape = zarr_root[s0_path].shape
        
        for axis in [0, 1, 2]:
            for slice_idx in range(shape[axis]):
                all_tasks.append({
                    "zarr_path": zarr_path,
                    "s0_path": s0_path,
                    "crop_name": volume_identifier,
                    "axis": axis_names[axis],
                    "slice": slice_idx
                })
                
    return all_tasks

def main():
    parser = argparse.ArgumentParser(description="Process zarr dataset in chunks.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data", help="Root directory for zarr volumes")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo", help="Local path to save dataset parts")

    # Chunking arguments
    parser.add_argument("--total_parts", type=int, default=100, help="Total number of chunks to divide the dataset into")
    parser.add_argument("--part_index", type=int, default=0, help="The 0-indexed part to process (e.g., 0 to 999)")
    
    args = parser.parse_args()
    print(f"Args: {args}")

    tasks = build_task_list(args.root_directory)
    total_tasks = len(tasks)
    
    # Chunking
    chunk_size = math.ceil(total_tasks / args.total_parts)
    start_idx = args.part_index * chunk_size
    end_idx = min(start_idx + chunk_size, total_tasks)
    
    chunked_tasks = tasks[start_idx:end_idx]
    print(f"Total slices across all volumes: {total_tasks}")
    print(f"Processing Part {args.part_index + 1} of {args.total_parts}")
    print(f"Extracting slices {start_idx} to {end_idx - 1} ({len(chunked_tasks)} tasks)")

    # Pass only the chunked list
    dataset = Dataset.from_list(chunked_tasks)

    # Define the final schema we want (this enforces the column order and dtypes)
    final_features = Features({
        "image": HFImage(),
        "crop_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32")
    })

    print(f"Igniting parallel extraction with {NUM_PROC} workers...")
    dataset = dataset.map(
        process_slice_worker,
        num_proc=NUM_PROC,
        remove_columns=["zarr_path", "s0_path"], 
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