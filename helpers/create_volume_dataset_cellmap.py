import os
import zarr
import numpy as np
import scipy.ndimage
import argparse
from glob import glob
from datasets import Dataset, Features, Value, Sequence

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
MAX_VOXELS = 2_000_000_000

def normalize_to_uint8(vol_3d):
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

def generate_3d_volumes(root_dir):
    """Generator yielding unresized 3D volumes and their corresponding labels for all crops."""
    zarr_paths = glob(os.path.join(root_dir, '*/*.zarr'))

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
                
                full_crop_name = f"{dataset_name}/{recon_name}/{crop_name}"

                label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', 's0')
                if label_path_str not in zarr_root:
                    print(f"Skipping crop {full_crop_name} because label path not found.")
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
                    print(f"Skipping crop {full_crop_name} because loaded raw volume is empty.")
                    continue
                
                label_vol = np.array(label_array)
                target_shape = label_vol.shape

                # Resize Raw
                if raw_crop_3d.shape != target_shape:
                    print(f"Reshaping raw volume for {crop_name} from {raw_crop_3d.shape} to {label_vol.shape} ...")
                    raw_zoom = [t / s for t, s in zip(target_shape, raw_crop_3d.shape)]
                    raw_crop_3d = scipy.ndimage.zoom(raw_crop_3d, raw_zoom, order=3, prefilter=False)
                
                # 3. Normalize 3D volume
                vol_3d_uint8 = normalize_to_uint8(raw_crop_3d)
                if vol_3d_uint8 is None:
                    print(f"Skipping crop {full_crop_name} because it is entirely uniform or just padding.")
                    continue

                print(f"Label volume shape: {vol_3d_uint8.shape}, Raw volume shape: {raw_crop_3d.shape}")

                label_vol_uint8 = label_vol.astype(np.uint8)

                total_voxels = vol_3d_uint8.size

                if total_voxels > MAX_VOXELS:

                    num_splits = int(np.ceil(total_voxels / MAX_VOXELS))
                    sub_vols = np.array_split(vol_3d_uint8, num_splits, axis=0)
                    sub_labels = np.array_split(label_vol_uint8, num_splits, axis=0)
                    
                    print(f"Dividing {full_crop_name} into {num_splits} parts.")

                    for i, (sub_vol, sub_label) in enumerate(zip(sub_vols, sub_labels)):
                        if sub_vol.size == 0:
                            continue
                            
                        yield {
                            "volume": sub_vol.tobytes(),
                            "label": sub_label.tobytes(),
                            "crop_name": f"{full_crop_name}_part{i}",
                            "shape": list(sub_vol.shape)
                        }
                else:
                    yield {
                        "volume": vol_3d_uint8.tobytes(),
                        "label": label_vol_uint8.tobytes(),
                        "crop_name": full_crop_name,
                        "shape": list(vol_3d_uint8.shape)
                    }

def main():
    parser = argparse.ArgumentParser(description="Create 3D volume dataset for CellMap.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/cellmap-segmentation-challenge/data", help="Root directory for zarr volumes")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_cellmap_3d", help="Local path to save dataset")

    args = parser.parse_args()

    # Define the schema explicitly for Hugging Face
    features = Features({
        "volume": Value("large_binary"),
        "label": Value("large_binary"),
        "crop_name": Value("string"),
        "shape": Sequence(Value("int32"))
    })

    print("Initializing 3D dataset generation. This may take a while depending on data size...")
    
    # Create the dataset using the generator
    dataset = Dataset.from_generator(
        generate_3d_volumes, 
        gen_kwargs={"root_dir": args.root_directory}, 
        features=features
    )
    print(f"Dataset generated with {len(dataset)} 3D crops.")
    
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    print("Dataset shuffled!")

    # Save the dataset directly to local disk with shard size limit
    print(f"Saving dataset locally to {args.local_save_dir}...")
    dataset.save_to_disk(args.local_save_dir, max_shard_size="2GB")
    print("Dataset saved!")

if __name__ == "__main__":
    main()