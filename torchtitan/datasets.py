import os
import zarr
import scipy.ndimage
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from glob import glob

import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2


def make_transform_2d():
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return v2.Compose([to_float, normalize])

def make_transform_3d():
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(mean=(0.449,), std=(0.226,))
    return v2.Compose([to_float, normalize])

transform_2d = make_transform_2d()
transform_3d = make_transform_3d()

def find_and_split_samples(root_dir, labels_scale, val_split=0.11, seed=1):
    """
    Scans the directory once and returns deterministic train/val splits of sample metadata.
    """
    samples = []
    zarr_paths = glob(os.path.join(root_dir, '*/*.zarr'))

    for zarr_path in zarr_paths:
        try:
            zarr_root = zarr.open(zarr_path, mode='r')
        except Exception as e:
            print(f"Could not open {zarr_path}, skipping. Error: {e}")
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
                
                label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', labels_scale)
                
                if label_path_str in zarr_root:
                    samples.append({
                        'zarr_path': zarr_path, 
                        'raw_path_group': raw_group_path_str, 
                        'label_path': label_path_str
                    })

    # Sort deterministically
    samples.sort(key=lambda x: x['zarr_path'] + x['label_path'])

    # Shuffle
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    # Split
    split_idx = int(len(samples) * (1 - val_split))
    return samples[:split_idx], samples[split_idx:]


class ZarrBaseDataset(IterableDataset):
    """
    Base class containing helper methods for metadata parsing and finding scales.
    Does not handle iteration logic.
    """
    def __init__(self, raw_scale='s0', labels_scale='s0'):
        super().__init__()
        self.raw_scale = raw_scale
        self.labels_scale = labels_scale
        self.rng = np.random.default_rng()

    def _parse_ome_ngff_metadata(self, attrs, scale_level_name):
        try:
            multiscales = attrs['multiscales'][0]
            datasets = multiscales['datasets']
            scale_metadata = next((d for d in datasets if d['path'] == scale_level_name), None)
            
            if scale_metadata:
                transformations = scale_metadata['coordinateTransformations']
                scale_transform = next((t for t in transformations if t['type'] == 'scale'), None)
                translation_transform = next((t for t in transformations if t['type'] == 'translation'), None)
                
                scale = scale_transform['scale'] if scale_transform else [1.0, 1.0, 1.0]
                translation = translation_transform['translation'] if translation_transform else [0.0, 0.0, 0.0]
                return scale, translation
        except (KeyError, IndexError, StopIteration):
            pass
        return None, None
    
    def _find_best_raw_scale(self, target_label_scale, raw_attrs):
        try:
            multiscales = raw_attrs['multiscales'][0]
            datasets = multiscales['datasets']
        except (KeyError, IndexError):
            return self.raw_scale, None, None

        available_scales = []
        for d in datasets:
            try:
                scale = next(t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')
                translation = next(t['translation'] for t in d['coordinateTransformations'] if t['type'] == 'translation')
                available_scales.append({'path': d['path'], 'scale': scale, 'translation': translation})
            except (KeyError, StopIteration):
                continue
        
        if not available_scales:
            return self.raw_scale, None, None

        candidates = [s for s in available_scales if all(rs <= ls for rs, ls in zip(s['scale'], target_label_scale))]

        if candidates:
            best_candidate = min(candidates, key=lambda s: sum(ls - rs for ls, rs in zip(target_label_scale, s['scale'])))
            return best_candidate['path'], best_candidate['scale'], best_candidate['translation']
        else:
            highest_res_scale = min(available_scales, key=lambda s: sum(s['scale']))
            return highest_res_scale['path'], highest_res_scale['scale'], highest_res_scale['translation']


class ZarrTrainDataset3D(ZarrBaseDataset):
    """
    Infinite iterator for 3D training crops.
    """
    def __init__(self, samples, crop_size, raw_scale='s0', labels_scale='s0', augment=True):
        super().__init__(raw_scale, labels_scale)
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment

    def _augment_data(self, raw: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Random Flips
        for axis in range(3):
            if self.rng.random() < 0.5:
                raw = np.flip(raw, axis=axis)
                label = np.flip(label, axis=axis)
        
        # Random 90-degree Rotations
        k = self.rng.integers(0, 4)
        if k > 0:
            valid_planes = []
            shape = raw.shape
            if shape[0] == shape[1]: valid_planes.append((0, 1))
            if shape[0] == shape[2]: valid_planes.append((0, 2))
            if shape[1] == shape[2]: valid_planes.append((1, 2))
            
            if valid_planes:
                axes = self.rng.choice(valid_planes)
                raw = np.rot90(raw, k=k, axes=axes)
                label = np.rot90(label, k=k, axes=axes)
        
        return raw.copy(), label.copy()

    def _get_sample(self, sample_info):
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        label_array = zarr_root[sample_info['label_path']]

        # Metadata parsing
        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
        
        if label_scale is None: raise ValueError("Metadata parsing failed")
            
        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        best_raw_scale_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)

        if raw_scale is None:
             # Fallback
             raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

        original_shape = label_array.shape
        target_shape = self.crop_size

        # --- Label Extraction ---
        # NOTE: Resizing only (up/down-sampling)
        label_data = label_array[:]
        zoom_factor = [t / s for t, s in zip(target_shape, original_shape)]
        resampled_label_mask = scipy.ndimage.zoom(label_data, zoom_factor, order=0, prefilter=False)
        
        final_label_mask = np.zeros(target_shape, dtype=resampled_label_mask.dtype)
        slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, resampled_label_mask.shape))
        final_label_mask[slicing_for_copy] = resampled_label_mask[slicing_for_copy]

        adjusted_label_translation = label_translation
        adjusted_label_scale = [ (sh * sc) / ts for sh, sc, ts in zip(original_shape, label_scale, target_shape)]

        # --- Raw Extraction ---
        best_raw_array_path = os.path.join(raw_group_path, best_raw_scale_path)
        raw_array = zarr_root[best_raw_array_path]

        scale_ratio = [ls / rs for ls, rs in zip(adjusted_label_scale, raw_scale)]
        relative_start_physical = [lt - rt for lt, rt in zip(adjusted_label_translation, raw_translation)]
        start_voxels_raw = [int(round(p / s)) for p, s in zip(relative_start_physical, raw_scale)]

        is_downsampling_or_equal = all(r >= 0.999 for r in scale_ratio)

        if is_downsampling_or_equal:
            step = [max(1, int(round(r))) for r in scale_ratio]
            end_voxels_raw = [st + (dim * sp) for st, dim, sp in zip(start_voxels_raw, target_shape, step)]
            slicing = tuple(slice(st, en, sp) for st, en, sp in zip(start_voxels_raw, end_voxels_raw, step))
            raw_crop_from_zarr = raw_array[slicing]
        else:
            label_physical_size = [sh * sc for sh, sc in zip(target_shape, adjusted_label_scale)]
            relative_end_physical = [s + size for s, size in zip(relative_start_physical, label_physical_size)]
            end_voxels_raw = [int(round(p / s)) for p, s in zip(relative_end_physical, raw_scale)]
            slicing = tuple(slice(start, end) for start, end in zip(start_voxels_raw, end_voxels_raw))
            raw_crop = raw_array[slicing]

            if any(s == 0 for s in raw_crop.shape):
                raw_crop_from_zarr = np.zeros(target_shape, dtype=raw_array.dtype)
            else:
                zoom_factor = [t / s for t, s in zip(target_shape, raw_crop.shape)]
                raw_crop_from_zarr = scipy.ndimage.zoom(raw_crop, zoom_factor, order=1, prefilter=False)

        final_raw_crop = np.zeros(target_shape, dtype=raw_array.dtype)
        slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, raw_crop_from_zarr.shape))
        final_raw_crop[slicing_for_copy] = raw_crop_from_zarr[slicing_for_copy]

        if self.augment:
            final_raw_crop, final_label_mask = self._augment_data(final_raw_crop, final_label_mask)

        raw_tensor = torch.from_numpy(final_raw_crop[np.newaxis, ...])
        label_tensor = torch.from_numpy(final_label_mask).long()

        return transform_3d(raw_tensor.expand(3, -1, -1, -1)), label_tensor

    def __iter__(self):
        while True:
            sample_info = self.rng.choice(self.samples)
            yield self._get_sample(sample_info)


class ZarrTrainDataset2D(ZarrBaseDataset):
    """
    Infinite iterator for 2D training slices.
    """
    def __init__(self, samples, crop_size, raw_scale='s0', labels_scale='s0', augment=True):
        super().__init__(raw_scale, labels_scale)
        self.samples = samples
        self.crop_size = crop_size # 2D tuple (H, W)
        self.augment = augment

    def _augment_data(self, raw: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            raw = np.flip(raw, axis=0)
            label = np.flip(label, axis=0)
        if self.rng.random() < 0.5:
            raw = np.flip(raw, axis=1)
            label = np.flip(label, axis=1)

        if raw.shape[0] == raw.shape[1]:
            k = self.rng.integers(0, 4)
        else:
            k = self.rng.choice([0, 2])

        if k > 0:
            raw = np.rot90(raw, k=k)
            label = np.rot90(label, k=k)
        return raw.copy(), label.copy()

    def _get_sample(self, sample_info):
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        label_array_3d = zarr_root[sample_info['label_path']]
        shape_3d = label_array_3d.shape
        
        axis = self.rng.integers(0, 3)
        slice_idx = self.rng.integers(0, shape_3d[axis])

        slicing_3d = [slice(None)] * 3
        slicing_3d[axis] = slice_idx
        label_slice_2d = label_array_3d[tuple(slicing_3d)]

        # Metadata parsing (condensed for brevity, logic identical to original)
        label_attrs = zarr_root[os.path.dirname(sample_info['label_path'])].attrs.asdict()
        label_scale_3d, label_translation_3d = self._parse_ome_ngff_metadata(label_attrs, os.path.basename(sample_info['label_path']))
        
        original_shape_2d = label_slice_2d.shape
        target_shape_2d = self.crop_size
        
        # NOTE: Resizing only (up/down-sampling)
        zoom_factor = [t / s for t, s in zip(target_shape_2d, original_shape_2d)]
        final_label_slice = scipy.ndimage.zoom(label_slice_2d, zoom_factor, order=0, prefilter=False)
        
        axes_2d = [i for i in range(3) if i != axis]
        adjusted_label_translation_2d = [label_translation_3d[axes_2d[0]], label_translation_3d[axes_2d[1]]]
        adjusted_label_scale_2d = [(sh * label_scale_3d[d]) / ts for sh, d, ts in zip(original_shape_2d, axes_2d, target_shape_2d)]

        # Fetch Raw 2D
        raw_attrs = zarr_root[sample_info['raw_path_group']].attrs.asdict()
        temp_target_label_scale_3d = [0.0]*3
        axes_2d = [i for i in range(3) if i != axis]
        temp_target_label_scale_3d[axes_2d[0]] = adjusted_label_scale_2d[0]
        temp_target_label_scale_3d[axes_2d[1]] = adjusted_label_scale_2d[1]
        temp_target_label_scale_3d[axis] = label_scale_3d[axis]

        best_raw_scale_path, raw_scale_3d, raw_translation_3d = self._find_best_raw_scale(temp_target_label_scale_3d, raw_attrs)
        raw_array_3d = zarr_root[os.path.join(sample_info['raw_path_group'], best_raw_scale_path)]

        # Phys to Voxels logic
        phys_start_3d = [0.0]*3
        phys_start_3d[axes_2d[0]] = adjusted_label_translation_2d[0]
        phys_start_3d[axes_2d[1]] = adjusted_label_translation_2d[1]
        phys_start_3d[axis] = label_translation_3d[axis] + slice_idx * label_scale_3d[axis]

        rel_start = [ps - rt for ps, rt in zip(phys_start_3d, raw_translation_3d)]
        start_vox = [int(round(p / s)) for p, s in zip(rel_start, raw_scale_3d)]
        
        size_phys_2d = [sh * sc for sh, sc in zip(target_shape_2d, adjusted_label_scale_2d)]
        size_vox_2d = [int(round(p / s)) for p, s in zip(size_phys_2d, [raw_scale_3d[d] for d in axes_2d])]
        
        raw_shape = raw_array_3d.shape
        safe_start = [np.clip(start_vox[i], 0, raw_shape[i] - 1) for i in range(3)]
        
        safe_slicing = [0, 0, 0]
        safe_slicing[axes_2d[0]] = slice(safe_start[axes_2d[0]], min(safe_start[axes_2d[0]] + size_vox_2d[0], raw_shape[axes_2d[0]]))
        safe_slicing[axes_2d[1]] = slice(safe_start[axes_2d[1]], min(safe_start[axes_2d[1]] + size_vox_2d[1], raw_shape[axes_2d[1]]))
        safe_slicing[axis] = safe_start[axis]

        raw_slice_2d = raw_array_3d[tuple(safe_slicing)]
        
        if raw_slice_2d.shape != target_shape_2d:
             if any(s == 0 for s in raw_slice_2d.shape):
                 final_raw_slice = np.zeros(target_shape_2d, dtype=raw_array_3d.dtype)
             else:
                 zoom_factor = [t / s for t, s in zip(target_shape_2d, raw_slice_2d.shape)]
                 final_raw_slice = scipy.ndimage.zoom(raw_slice_2d, zoom_factor, order=1, prefilter=False)
        else:
            final_raw_slice = raw_slice_2d

        if self.augment:
            final_raw_slice, final_label_slice = self._augment_data(final_raw_slice, final_label_slice)

        raw_tensor = torch.from_numpy(final_raw_slice[np.newaxis, ...])
        label_tensor = torch.from_numpy(final_label_slice).long()
        return transform_2d(raw_tensor.expand(3, -1, -1)), label_tensor

    def __iter__(self):
        while True:
            sample_info = self.rng.choice(self.samples)
            yield self._get_sample(sample_info)


class ZarrValidationDataset3D(ZarrBaseDataset):
    """
    3D Validation Dataset. Iterates over 3D volumes (partitioned by rank). Yields full (C, D, H, W) volumes.
    """
    def __init__(self, samples, val_crop_size, rank, world_size, raw_scale='s0', labels_scale='s0'):
        super().__init__(raw_scale, labels_scale)
        self.samples = samples[rank::world_size]
        self.val_crop_size = val_crop_size
        self.rank = rank

    def __iter__(self):
        for sample_info in self.samples:
            # Reusing logic from 2D Val, but yielding the whole block
            # In a real refactor, this loading logic should be a shared method on Base
            zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
            full_label_vol = zarr_root[sample_info['label_path']][:]
            
            target_shape = self.val_crop_size
            if full_label_vol.shape != target_shape:
                label_zoom = [t / s for t, s in zip(target_shape, full_label_vol.shape)]
                resized_label_vol = scipy.ndimage.zoom(full_label_vol, label_zoom, order=0, prefilter=False)
            else:
                resized_label_vol = full_label_vol

            label_attrs = zarr_root[os.path.dirname(sample_info['label_path'])].attrs.asdict()
            label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, os.path.basename(sample_info['label_path']))
            if label_scale is None: label_scale, label_translation = [1.,1.,1.], [0.,0.,0.]

            raw_attrs = zarr_root[sample_info['raw_path_group']].attrs.asdict()
            best_raw_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)
            raw_array_full = zarr_root[os.path.join(sample_info['raw_path_group'], best_raw_path)]

            label_phys_size = [s * sc for s, sc in zip(full_label_vol.shape, label_scale)]
            rel_start = [ls - rs for ls, rs in zip(label_translation, raw_translation)]
            rel_end = [s + sz for s, sz in zip(rel_start, label_phys_size)]
            start_raw = [int(round(p / s)) for p, s in zip(rel_start, raw_scale)]
            end_raw = [int(round(p / s)) for p, s in zip(rel_end, raw_scale)]
            slices = [slice(max(0, s), min(d, e)) for s, e, d in zip(start_raw, end_raw, raw_array_full.shape)]
            
            raw_crop_3d = raw_array_full[tuple(slices)]
            if any(s == 0 for s in raw_crop_3d.shape): raw_crop_3d = np.zeros(target_shape, dtype=raw_array_full.dtype)

            if raw_crop_3d.shape != target_shape:
                 raw_zoom = [t / s for t, s in zip(target_shape, raw_crop_3d.shape)]
                 resized_raw_vol = scipy.ndimage.zoom(raw_crop_3d, raw_zoom, order=1, prefilter=False)
            else:
                 resized_raw_vol = raw_crop_3d

            # Raw: (1, D, H, W) -> expand to (3, D, H, W) if needed, or keep 1
            # Prompt implies 3 channels usually for 2D, but let's stick to 1 channel 3D + transform
            raw_tensor = torch.from_numpy(resized_raw_vol)
            raw_tensor = raw_tensor.unsqueeze(0).expand(3, -1, -1, -1) # (3, D, H, W)
            label_tensor = torch.from_numpy(resized_label_vol).long()

            yield transform_3d(raw_tensor), label_tensor


class ZarrValidationDataset2D(ZarrBaseDataset):
    """
    2D Validation Dataset. Iterates over 3D volumes (partitioned by rank). For each volume, yields sequential (3, H, W) slices along the Z-axis.
    """
    def __init__(self, samples, val_crop_size, rank, world_size, raw_scale='s0', labels_scale='s0'):
        super().__init__(raw_scale, labels_scale)
        # Distributed Partitioning: Slice the list
        self.samples = samples[rank::world_size]
        self.val_crop_size = val_crop_size # (D, H, W)
        self.rank = rank

    def __iter__(self):
        # Iterate over assigned validation volumes
        for sample_idx, sample_info in enumerate(self.samples):
            zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
            
            # 1. Load Full Label Volume
            full_label_vol = zarr_root[sample_info['label_path']][:]
            
            # 2. Resize to Target (D, H, W)
            target_shape = self.val_crop_size
            if full_label_vol.shape != target_shape:
                label_zoom = [t / s for t, s in zip(target_shape, full_label_vol.shape)]
                resized_label_vol = scipy.ndimage.zoom(full_label_vol, label_zoom, order=0, prefilter=False)
            else:
                resized_label_vol = full_label_vol

            # 3. Load Corresponding Raw Volume
            label_attrs = zarr_root[os.path.dirname(sample_info['label_path'])].attrs.asdict()
            label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, os.path.basename(sample_info['label_path']))
            
            if label_scale is None: 
                label_scale, label_translation = [1.,1.,1.], [0.,0.,0.]

            raw_attrs = zarr_root[sample_info['raw_path_group']].attrs.asdict()
            best_raw_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)
            raw_array_full = zarr_root[os.path.join(sample_info['raw_path_group'], best_raw_path)]

            # Map label bounds to raw indices
            label_phys_size = [s * sc for s, sc in zip(full_label_vol.shape, label_scale)]
            rel_start = [ls - rs for ls, rs in zip(label_translation, raw_translation)]
            rel_end = [s + sz for s, sz in zip(rel_start, label_phys_size)]
            
            start_raw = [int(round(p / s)) for p, s in zip(rel_start, raw_scale)]
            end_raw = [int(round(p / s)) for p, s in zip(rel_end, raw_scale)]
            
            slices = [slice(max(0, s), min(d, e)) for s, e, d in zip(start_raw, end_raw, raw_array_full.shape)]
            raw_crop_3d = raw_array_full[tuple(slices)]

            if any(s == 0 for s in raw_crop_3d.shape):
                raw_crop_3d = np.zeros(target_shape, dtype=raw_array_full.dtype)

            # Resize Raw
            if raw_crop_3d.shape != target_shape:
                 raw_zoom = [t / s for t, s in zip(target_shape, raw_crop_3d.shape)]
                 resized_raw_vol = scipy.ndimage.zoom(raw_crop_3d, raw_zoom, order=1, prefilter=False)
            else:
                 resized_raw_vol = raw_crop_3d

            # 4. Prepare Tensors
            # Normalize Raw: (D, H, W) -> Float Tensor
            raw_tensor_vol = torch.from_numpy(resized_raw_vol)
            label_tensor_vol = torch.from_numpy(resized_label_vol).long()

            # --- ORTHOPLANE ITERATION ---
            # We iterate through axes 0 (Z), 1 (Y), and 2 (X)
            # Using a global unique ID for the sample helps reconstruction (e.g., hash of path or simple index)
            sample_id = f"rank{self.rank}_sample{sample_idx}"

            for axis in [0, 1, 2]:
                num_slices = target_shape[axis]  # (D, H, W) 
                
                for slice_idx in range(num_slices):
                    # Slicing dynamically based on axis
                    # If axis=0 (Z): slice is (H, W)
                    # If axis=1 (Y): slice is (D, W)
                    # If axis=2 (X): slice is (D, H)
                    
                    if axis == 0:
                        r_slice = raw_tensor_vol[slice_idx, :, :]
                        l_slice = label_tensor_vol[slice_idx, :, :]
                    elif axis == 1:
                        r_slice = raw_tensor_vol[:, slice_idx, :]
                        l_slice = label_tensor_vol[:, slice_idx, :]
                    else:
                        r_slice = raw_tensor_vol[:, :, slice_idx]
                        l_slice = label_tensor_vol[:, :, slice_idx]

                    # Expand to (3, H, W) for the model 
                    img = r_slice.unsqueeze(0).expand(3, -1, -1)
                    
                    # Apply transform (normalize)
                    img = transform_2d(img)

                    # Metadata for reconstruction
                    meta = {
                        "sample_id": sample_id,
                        "axis": axis,         # 0=XY, 1=XZ, 2=YZ
                        "slice_idx": slice_idx,
                        "vol_shape": torch.tensor(self.val_crop_size) # pass shape to help init buffers
                    }

                    yield img, l_slice, meta

def build_data_loader(
    batch_size: int,
    root_dir: str,
    crop_size: Union[Tuple[int, int], Tuple[int, int, int]], 
    val_crop_size: Tuple[int, int, int], 
    rank: int = 0,
    world_size: int = 1,
    augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds both Train and Validation loaders.
    """
    
    # Find and split samples (once)
    train_samples, val_samples = find_and_split_samples(root_dir, labels_scale='s0')
    print(f"[Rank {rank}]: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    
    # Instantiate datasets
    if len(crop_size) == 3:
        # 3D
        train_dataset = ZarrTrainDataset3D(
            samples=train_samples,
            crop_size=crop_size,
            augment=augment
        )
        val_dataset = ZarrValidationDataset3D(
            samples=val_samples,
            val_crop_size=val_crop_size,
            rank=rank,
            world_size=world_size
        )
    else:
        # 2D
        train_dataset = ZarrTrainDataset2D(
            samples=train_samples,
            crop_size=crop_size,
            augment=augment
        )
        val_dataset = ZarrValidationDataset2D(
            samples=val_samples,
            val_crop_size=val_crop_size,
            rank=rank,
            world_size=world_size
        )

    # Build loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    
    # We could use a different batch size for validation, but keeping it simple and using same batch_size for now
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    return train_loader, val_loader