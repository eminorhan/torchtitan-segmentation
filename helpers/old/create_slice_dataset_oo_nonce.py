import os
import re
import zarr
import numpy as np
import io
import gc
from glob import glob
from PIL import Image
from datasets import Dataset, Image as HFImage, Features, Value
import numcodecs

# ---------------- CONFIGURATION ------------------------------------------------------------
NUM_PROC = 128                          
WORKER_CACHE_SIZE = 1024 * 1024 * 1024  
# -------------------------------------------------------------------------------------------

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

# NEW: Add process index arguments
def slice_generator(zarr_paths, _process_index=0, _num_processes=1):
    axis_names = {0: 'z', 1: 'y', 2: 'x'}
    
    # 1. The worker calculates which chunk of the list it is responsible for
    chunk_size = max(1, len(zarr_paths) // _num_processes)
    start_idx = _process_index * chunk_size
    # Ensure the last worker grabs any remaining files
    end_idx = start_idx + chunk_size if _process_index < _num_processes - 1 else len(zarr_paths)
    
    # 2. Slice the master list so this worker only processes its assigned files
    my_paths = zarr_paths[start_idx:end_idx]
    
    # 3. Iterate over the assigned chunk (now guaranteed to be strings)
    for zarr_path in my_paths:
        try:
            store = zarr.DirectoryStore(zarr_path)
            cache = zarr.LRUStoreCache(store, max_size=WORKER_CACHE_SIZE)
            zarr_root = zarr.open(store=cache, mode='r')
        except Exception as e:
            print(f"[Worker {_process_index}] FAILED to open {zarr_path}: {e}") 
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

        # RE-ADDED: Define the dataset name and crop name
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')
        volume_identifier = f"{dataset_name}/{earliest_recon}/{best_em}"

        current_array = zarr_root[s0_path]
        shape = current_array.shape
        
        for axis in [0, 1, 2]:
            for slice_idx in range(shape[axis]):
                if axis == 0: 
                    slice_2d = current_array[slice_idx, :, :]
                elif axis == 1: 
                    slice_2d = current_array[:, slice_idx, :]
                else: 
                    slice_2d = current_array[:, :, slice_idx]

                slice_2d = np.array(slice_2d)
                slice_2d_uint8 = normalize_to_uint8(slice_2d)

                img = Image.fromarray(slice_2d_uint8)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")

                # RE-ADDED: Yield the full dictionary with your features!
                yield {
                    "image": {"bytes": buffer.getvalue(), "path": None},
                    "crop_name": volume_identifier,
                    "axis": axis_names[axis],
                    "slice": slice_idx
                }

        del zarr_root, cache, store
        gc.collect()

def main():
    root_directory = "/lustre/blizzard/stf218/scratch/emin/seg3d/data"
    repo_id = "eminorhan/openorganelle-2d"

    # 1. This is a flat list of strings
    zarr_paths = glob(os.path.join(root_directory, '*/*.zarr'))
    
    if not zarr_paths:
        print("No valid data found to process!")
        return

    features = Features({
        "image": HFImage(),
        "crop_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32") 
    })

    print(f"Igniting generator extraction with {min(NUM_PROC, len(zarr_paths))} workers...")
    
    # 2. Pass the flat list. Hugging Face duplicates this flat list to every worker,
    # and the logic inside the generator slices it safely.
    dataset = Dataset.from_generator(
        slice_generator, 
        gen_kwargs={"zarr_paths": zarr_paths}, 
        num_proc=min(NUM_PROC, len(zarr_paths)),
        features=features
    )

    dataset = dataset.shuffle(seed=42)
    
    print("Dataset processed and shuffled! Pushing to hub...")
    dataset.push_to_hub(repo_id, max_shard_size="1GB")
    print("Upload complete!")

if __name__ == "__main__":
    main()