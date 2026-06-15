import os
import zarr
import numpy as np
import scipy.ndimage
import argparse
from glob import glob
from PIL import Image
from datasets import Dataset, Features, Image as HFImage, Value

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0

def normalize_to_uint8(slice_2d):
    """Safely casts heterogenous EM arrays to standard 8-bit grayscale with a black background."""
    
    # Drop completely uniform slices early
    if slice_2d.min() == slice_2d.max():
        return None
        
    # Convert to float for safe math
    slice_float = slice_2d.astype(np.float32)
    
    # Identify background padding values dynamically
    bg_black = slice_float.min()
    bg_white = slice_float.max()
    
    # Create a 1D array of only the valid tissue pixels
    valid_mask = (slice_float != bg_black) & (slice_float != bg_white) & (slice_float != 0.0)
    valid_pixels = slice_float[valid_mask]
    
    # Fallback if the slice is entirely padding
    if valid_pixels.size == 0:
        return None
    
    # Calculate percentiles ONLY on the valid tissue pixels
    p_low, p_high = np.percentile(valid_pixels, (LOWER_PERCENTILE, UPPER_PERCENTILE))

    # Fallback if percentiles are identical
    if p_high - p_low == 0:
        p_low, p_high = valid_pixels.min(), valid_pixels.max()
        if p_high - p_low == 0:
            return None

    # Normalize the ENTIRE slice (including padding) to 0.0 - 1.0 range
    normalized = np.clip((slice_float - p_low) / (p_high - p_low), 0, 1)
    
    # Scale to 255 and cast back to uint8
    return (normalized * 255.0).astype(np.uint8)

def parse_ome_ngff_metadata(attrs, scale_level_name):
    """Extracts scale and translation metadata from OME-NGFF zarr attributes."""
    try:
        multiscales = attrs.get('multiscales', [{}])[0]
        datasets = multiscales.get('datasets', [])
        scale_metadata = next((d for d in datasets if d['path'] == scale_level_name), None)
        
        if scale_metadata:
            transformations = scale_metadata.get('coordinateTransformations', [])
            scale_transform = next((t for t in transformations if t['type'] == 'scale'), None)
            translation_transform = next((t for t in transformations if t['type'] == 'translation'), None)
            
            scale = scale_transform['scale'] if scale_transform else [1.0, 1.0, 1.0]
            translation = translation_transform['translation'] if translation_transform else [0.0, 0.0, 0.0]
            return scale, translation
    except (KeyError, IndexError, StopIteration):
        pass
    return [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

def find_best_raw_scale(target_label_scale, raw_attrs, default_scale='s0'):
    """Finds the optimal raw resolution scale matching the label scale."""
    try:
        multiscales = raw_attrs['multiscales'][0]
        datasets = multiscales['datasets']
    except (KeyError, IndexError):
        return default_scale, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

    available_scales = []
    for d in datasets:
        try:
            scale = next(t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')
            translation = next(t['translation'] for t in d['coordinateTransformations'] if t['type'] == 'translation')
            available_scales.append({'path': d['path'], 'scale': scale, 'translation': translation})
        except (KeyError, StopIteration):
            continue
    
    if not available_scales:
        return default_scale, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

    candidates = [s for s in available_scales if all(rs <= ls for rs, ls in zip(s['scale'], target_label_scale))]

    if candidates:
        best = min(candidates, key=lambda s: sum(ls - rs for ls, rs in zip(target_label_scale, s['scale'])))
        return best['path'], best['scale'], best['translation']
    else:
        highest_res = min(available_scales, key=lambda s: sum(s['scale']))
        return highest_res['path'], highest_res['scale'], highest_res['translation']

def generate_2d_slices(root_dir):
    """Generator yielding unresized 2D slices across X, Y, Z axes for all crops."""
    zarr_paths = glob(os.path.join(root_dir, '*/*.zarr'))
    axis_names = {0: 'z', 1: 'y', 2: 'x'}

    for zarr_path in zarr_paths:

        # Extract e.g. 'jrc_mus-kidney' from '/.../jrc_mus-kidney.zarr'
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')

        try:
            zarr_root = zarr.open(zarr_path, mode='r')
        except Exception as e:
            print(f"Skipping {zarr_path}: {e}")
            continue

        for recon_name in zarr_root.keys():
            if not recon_name.startswith('recon-'): continue

            raw_group_path_str = os.path.join(recon_name, 'em', 'fibsem-uint8')
            labels_base_path_str = os.path.join(recon_name, 'labels', 'groundtruth')

            if raw_group_path_str not in zarr_root or labels_base_path_str not in zarr_root:
                continue
            
            groundtruth_group = zarr_root[labels_base_path_str]
            for crop_name in groundtruth_group.keys():
                if not crop_name.startswith('crop'): continue
                
                label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', 's0')
                if label_path_str not in zarr_root:
                    continue

                # 1. Fetch Metadata
                label_array = zarr_root[label_path_str]
                label_attrs = zarr_root[os.path.dirname(label_path_str)].attrs.asdict()
                label_scale, label_translation = parse_ome_ngff_metadata(label_attrs, 's0')
                
                raw_attrs = zarr_root[raw_group_path_str].attrs.asdict()
                best_raw_path, raw_scale, raw_translation = find_best_raw_scale(label_scale, raw_attrs)
                raw_array_full = zarr_root[os.path.join(raw_group_path_str, best_raw_path)]

                # 2. Calculate Bounding Box matching the crop at native raw resolution
                label_phys_size = [s * sc for s, sc in zip(label_array.shape, label_scale)]
                rel_start = [ls - rs for ls, rs in zip(label_translation, raw_translation)]
                rel_end = [s + sz for s, sz in zip(rel_start, label_phys_size)]
                
                start_raw = [int(round(p / s)) for p, s in zip(rel_start, raw_scale)]
                end_raw = [int(round(p / s)) for p, s in zip(rel_end, raw_scale)]
                
                slices = [slice(max(0, s), min(d, e)) for s, e, d in zip(start_raw, end_raw, raw_array_full.shape)]
                raw_crop_3d = np.array(raw_array_full[tuple(slices)]) # Load into memory at native size
                
                if raw_crop_3d.size == 0:
                    continue
                
                label_vol = np.array(label_array)
                
                if raw_crop_3d.shape != label_vol.shape:
                    print(f"Reshaping raw volume for {crop_name} from {raw_crop_3d.shape} to {label_vol.shape} ...")
                    zoom_factor = [t / s for t, s in zip(label_vol.shape, raw_crop_3d.shape)]  # Target / source
                    raw_crop_3d = scipy.ndimage.zoom(raw_crop_3d, zoom_factor, order=1, prefilter=False)
                
                full_crop_name = f"{dataset_name}/{recon_name}/{crop_name}"
                # print(f"full crop name: {full_crop_name}")
                
                # 3. Yield 2D slices along Z, Y, and X
                for axis in [0, 1, 2]:
                    num_slices = raw_crop_3d.shape[axis]
                    
                    for slice_idx in range(num_slices):
                        if axis == 0:
                            slice_2d = raw_crop_3d[slice_idx, :, :]
                            label_2d = label_vol[slice_idx, :, :]
                        elif axis == 1:
                            slice_2d = raw_crop_3d[:, slice_idx, :]
                            label_2d = label_vol[:, slice_idx, :]
                        else:
                            slice_2d = raw_crop_3d[:, :, slice_idx]
                            label_2d = label_vol[:, :, slice_idx]

                        slice_2d_uint8 = normalize_to_uint8(slice_2d)
                        if slice_2d_uint8 is None:
                            continue

                        # print(f"full crop name: {full_crop_name}, axis: {axis_names[axis]}, slice: {slice_idx}")

                        yield {
                            "image": Image.fromarray(slice_2d_uint8),
                            "label": Image.fromarray(label_2d.astype(np.uint8)),
                            "crop_name": full_crop_name,
                            "axis": axis_names[axis],
                            "slice": slice_idx
                        }

def main():
    parser = argparse.ArgumentParser(description="Create 2D slice dataset for CellMap.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/cellmap-segmentation-challenge/data", help="Root directory for zarr volumes")
    parser.add_argument("--repo_id", type=str, default="eminorhan/cellmap-2d", help="Local path to save dataset")

    args = parser.parse_args()

    # Define the schema explicitly for Hugging Face
    features = Features({
        "image": HFImage(),
        "label": HFImage(),
        "crop_name": Value("string"),
        "axis": Value("string"),
        "slice": Value("int32")
    })

    print("Initializing dataset generation. This may take a while depending on data size...")
    
    # Create the dataset using the generator
    dataset = Dataset.from_generator(
        generate_2d_slices, 
        gen_kwargs={"root_dir": args.root_directory}, 
        features=features
    )
    print(f"Dataset generated with {len(dataset)} slices.")
    
    # Shuffle the dataset to evenly distribute large/small images across shards
    dataset = dataset.shuffle(seed=42)
    print("Dataset shuffled!")

    # Push the dataset directly to Hugging Face with shard size limit
    dataset.push_to_hub(args.repo_id, max_shard_size="2GB")
    print("Dataset uploaded to HF Hub!")

if __name__ == "__main__":
    main()