import os
import re
import math
import argparse
import zarr
import numpy as np
from glob import glob
from PIL import Image

from datasets import Dataset, Features, Image as HFImage, Value

# ---------------- CONFIG ----------------
NUM_PROC = 128   # tune this (not too high or you’ll thrash I/O)
NUM_CHUNKS = 10
# ----------------------------------------

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

# ---------------- VOLUME DISCOVERY ----------------

def discover_volumes(root_dir):
    zarr_paths = sorted(glob(os.path.join(root_dir, '*/*.zarr')))
    valid = []

    for zarr_path in zarr_paths:
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')

        try:
            zarr_root = zarr.open(zarr_path, mode='r')
        except Exception:
            continue

        recon_keys = [k for k in zarr_root.keys() if k.startswith('recon-')]
        if not recon_keys:
            continue

        earliest_recon = sorted(recon_keys, key=get_recon_sort_key)[0]
        em_path = f"{earliest_recon}/em"
        if em_path not in zarr_root:
            continue

        em_subfolders = list(zarr_root[em_path].keys())
        if not em_subfolders:
            continue

        best_em = sorted(em_subfolders, key=get_em_subfolder_sort_key)[0]
        s0_path = f"{em_path}/{best_em}/s0"
        if s0_path not in zarr_root:
            continue

        volume_identifier = f"{dataset_name}/{earliest_recon}/{best_em}"

        valid.append({
            "zarr_path": zarr_path,
            "s0_path": s0_path,
            "volume_id": volume_identifier
        })

    return valid

# ---------------- GENERATOR ----------------

def slice_generator(volumes):
    axis_names = {0: 'z', 1: 'y', 2: 'x'}

    for vol in volumes:
        zarr_root = zarr.open(vol["zarr_path"], mode='r')
        arr = zarr_root[vol["s0_path"]]
        shape = arr.shape

        for axis in [0, 1, 2]:
            for i in range(shape[axis]):
                if axis == 0:
                    slice_2d = arr[i, :, :]
                elif axis == 1:
                    slice_2d = arr[:, i, :]
                else:
                    slice_2d = arr[:, :, i]

                slice_2d = np.array(slice_2d)
                slice_2d = normalize_to_uint8(slice_2d)

                yield {
                    "image": Image.fromarray(slice_2d),
                    "crop_name": vol["volume_id"],
                    "axis": axis_names[axis],
                    "slice": i
                }

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    args = parser.parse_args()

    volumes = discover_volumes(args.root_dir)
    volumes = sorted(volumes, key=lambda x: x["zarr_path"])

    # split volumes into chunks
    chunk_size = math.ceil(len(volumes) / NUM_CHUNKS)
    start = args.chunk_idx * chunk_size
    end = min(start + chunk_size, len(volumes))

    chunk_volumes = volumes[start:end]

    print(f"Processing chunk {args.chunk_idx}")
    print(f"Volumes in chunk: {len(chunk_volumes)}")

    features = Features({
        "image": HFImage(),
        "crop_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32")
    })

    ds = Dataset.from_generator(
        slice_generator,
        gen_kwargs={"volumes": chunk_volumes},
        features=features,
        num_proc=NUM_PROC
    )

    out_path = os.path.join(args.out_dir, f"chunk_{args.chunk_idx}")
    ds.save_to_disk(out_path)

    print(f"Saved chunk to {out_path}")


if __name__ == "__main__":
    main()