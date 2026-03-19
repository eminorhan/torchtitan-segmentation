import os
import re
import zarr
import numpy as np
import io
import gc
from glob import glob
from PIL import Image
from datasets import Dataset, Image as HFImage, disable_progress_bar

# Mute the Hugging Face progress bar to save the log file
disable_progress_bar()

# ---------------- CONFIGURATION ------------------------------------------------------------
# Leaving a few cores free for the main process, OS, and I/O operations
NUM_PROC = 140                          
# 512MB is plenty since the OS page cache will help, totaling ~70GB across 140 workers
WORKER_CACHE_SIZE = 512 * 1024 * 1024  
# Increased massively. Main process will now write 1000 rows at a time, clearing the queue
WRITER_BATCH_SIZE = 1000
# -------------------------------------------------------------------------------------------

WORKER_CACHE = {}
WORKER_LOCAL_COUNTER = 0

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
    if slice_2d.dtype == np.uint8:
        return slice_2d

    slice_float = slice_2d.astype(np.float32)
    p_low, p_high = np.percentile(slice_float, (1, 99))

    if p_high - p_low == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    normalized = np.clip((slice_float - p_low) / (p_high - p_low), 0, 1)
    return (normalized * 255.0).astype(np.uint8)

def process_slice_worker(example):
    global WORKER_LOCAL_COUNTER

    zarr_path = example['zarr_path']
    s0_path = example['s0_path']
    axis = example['axis']
    slice_idx = example['slice']

    WORKER_LOCAL_COUNTER += 1
    if WORKER_LOCAL_COUNTER % 1000 == 0:
        pid = os.getpid()
        print(f"[Worker {pid}] Processed {WORKER_LOCAL_COUNTER} slices...")

    # FIX 1: Explicitly destroy old caches and force Garbage Collection
    if WORKER_CACHE.get("current_zarr") != zarr_path:
        if "root" in WORKER_CACHE:
            del WORKER_CACHE["root"]
        WORKER_CACHE.clear()
        gc.collect() # Force RAM cleanup immediately before allocating the next cache
        
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

    # FIX 2: Serialize to PNG bytes inside the worker. 
    # This bypasses `dill` trying to pickle massive PIL objects.
    img = Image.fromarray(slice_2d_uint8)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    # Return raw bytes in the format HF datasets expects
    return {"image": {"bytes": buffer.getvalue(), "path": None}}

def build_task_list(root_dir):
    zarr_paths = glob(os.path.join(root_dir, '*/*.zarr'))
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
        s0_path = f"{em_path}/{best_em}/s0"
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
    root_directory = "/lustre/blizzard/stf218/scratch/emin/seg3d/data"
    repo_id = "eminorhan/openorganelle-2d"

    print(f"Initializing setup. Utilizing {NUM_PROC} CPU cores for processing.")

    tasks = build_task_list(root_directory)
    print(f"Total slices to generate: {len(tasks)}")
    
    if len(tasks) == 0:
        print("No valid data found to process!")
        return

    dataset = Dataset.from_list(tasks)

    print(f"Igniting parallel extraction with {NUM_PROC} workers...")
    
    # FIX 3: WRITER_BATCH_SIZE increased via global config
    dataset = dataset.map(
        process_slice_worker,
        num_proc=NUM_PROC,
        remove_columns=["zarr_path", "s0_path"], 
        desc="Slicing, normalizing, and compressing",
        writer_batch_size=WRITER_BATCH_SIZE 
    )

    dataset = dataset.cast_column("image", HFImage())
    
    dataset = dataset.shuffle(seed=42)
    print("Dataset shuffled!")
    
    dataset.push_to_hub(repo_id, max_shard_size="1GB")
    print("Upload complete!")

if __name__ == "__main__":
    main()