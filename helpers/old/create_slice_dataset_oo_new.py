import os
import re
import zarr
# import numcodecs
import numpy as np
from glob import glob
from PIL import Image
from multiprocessing import Pool
from datasets import Dataset, Features, Image as HFImage, Value, disable_progress_bar

# ---------------- CONFIGURATION ----------------
disable_progress_bar()
# numcodecs.blosc.use_threads = False 

NUM_PROC = 128
WORKER_CACHE_SIZE = 1024 * 1024 * 1024

# A simple dict. In multiprocessing, every worker gets its own isolated copy of this!
WORKER_CACHE = {}
# -----------------------------------------------

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

    # Convert to float32 for precision math
    slice_float = slice_2d.astype(np.float32)

    # Use percentiles to ignore dead/hot pixel outliers common in EM data
    p_low, p_high = np.percentile(slice_float, (1, 99))

    # Handle edge case where the slice is completely flat (e.g., pure background)
    if p_high - p_low == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    # Normalize to 0.0 - 1.0, clip outliers, and scale to 255
    normalized = np.clip((slice_float - p_low) / (p_high - p_low), 0, 1)

    return (normalized * 255.0).astype(np.uint8)

def robust_worker(task):
    """The Producer: Processes a single slice. Built to survive Lustre timeouts."""
    zarr_path, s0_path, crop_name, axis, slice_idx = task
    
    try:
        # Initialize or swap worker-specific cache
        if WORKER_CACHE.get("current_zarr") != zarr_path:
            WORKER_CACHE.clear()
            store = zarr.DirectoryStore(zarr_path)
            cache = zarr.LRUStoreCache(store, max_size=WORKER_CACHE_SIZE)
            WORKER_CACHE["root"] = zarr.open(store=cache, mode='r')
            WORKER_CACHE["current_zarr"] = zarr_path
            
        current_array = WORKER_CACHE["root"][s0_path]
        
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

        # Return the processed payload
        return {
            "image": Image.fromarray(slice_2d_uint8),
            "crop_name": crop_name,
            "axis": axis,
            "slice": slice_idx
        }
        
    except Exception as e:
        # FAULT TOLERANCE: If Lustre drops the connection, print the error but DO NOT crash
        print(f"[ERROR] Failed on {crop_name} | Axis {axis} | Slice {slice_idx}. Reason: {e}")
        return None 

def build_task_list(root_dir):
    zarr_paths = glob(os.path.join(root_dir, '*/*.zarr'))
    all_tasks = []
    axis_names = {0: 'z', 1: 'y', 2: 'x'}

    print("Scanning Zarr volumes and building the task queue...")
    for zarr_path in zarr_paths:
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')
        try: zarr_root = zarr.open(zarr_path, mode='r')
        except Exception: continue

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
                all_tasks.append((zarr_path, s0_path, volume_identifier, axis_names[axis], slice_idx))
                
    return all_tasks

def unordered_generator(tasks):
    """The Consumer: Yields finished slices to Hugging Face the millisecond they are ready."""
    completed = 0
    
    # Fire up the native Python multiprocessing pool
    with Pool(processes=NUM_PROC) as pool:
        # imap_unordered yields results immediately as workers finish them!
        for result in pool.imap_unordered(robust_worker, tasks):
            if result is not None:
                completed += 1
                # Print a clean heartbeat to the log file every 1,000 successes
                if completed % 1000 == 0:
                    print(f"--> Successfully processed {completed} / {len(tasks)} slices...")
                
                yield result

def main():
    root_directory = "/lustre/blizzard/stf218/scratch/emin/seg3d/data"
    repo_id = "eminorhan/openorganelle-2d"

    tasks = build_task_list(root_directory)
    print(f"\nTotal slices to generate: {len(tasks)}")
    
    features = Features({
        "image": HFImage(),
        "crop_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32")
    })

    print(f"Igniting native multiprocessing with {NUM_PROC} workers...")
    
    # Hugging Face safely writes the stream to disk on a single thread
    dataset = Dataset.from_generator(
        unordered_generator,
        gen_kwargs={"tasks": tasks},
        features=features,
        writer_batch_size=10
    )

    print("Shuffling final dataset...")
    dataset = dataset.shuffle(seed=42)
    
    print(f"Pushing to Hub: {repo_id} ...")
    dataset.push_to_hub(repo_id, max_shard_size="1GB")
    print("Upload complete!")

if __name__ == "__main__":
    main()