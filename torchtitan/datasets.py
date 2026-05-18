import os
import re
import torch
import numpy as np
import scipy.ndimage
from typing import Tuple, Union
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2
from datasets import load_from_disk, load_dataset, Dataset

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

def resize_array(arr: np.ndarray, target_shape: Tuple, order: int) -> np.ndarray:
    """Helper to resize 2D or 3D arrays to the target crop size."""
    if arr.shape == target_shape:
        return arr
    zoom_factor = [t / s for t, s in zip(target_shape, arr.shape)]
    return scipy.ndimage.zoom(arr, zoom_factor, order=order, prefilter=False)

def get_base_crop_name(name: str) -> str:
    """Removes the '_partX' suffix from 3D crop names to ensure 2D and 3D keys match perfectly."""
    return re.sub(r'_part\d+$', '', name)

# ---------------------------------------------------------
# Dataset Iterators
# ---------------------------------------------------------

class HFTrainDataset3D(IterableDataset):
    def __init__(self, hf_dataset: Dataset, crop_size: Tuple[int, int, int], augment: bool = True, seed: int = 42):
        super().__init__()
        self.dataset = hf_dataset
        self.crop_size = crop_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def _augment_data(self, raw: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for axis in range(3):
            if self.rng.random() < 0.5:
                raw = np.flip(raw, axis=axis)
                label = np.flip(label, axis=axis)
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

    def __iter__(self):
        while True:
            idx = self.rng.integers(0, len(self.dataset))
            item = self.dataset[idx]

            shape = tuple(item['shape'])
            raw = np.frombuffer(item['volume'], dtype=np.uint8).reshape(shape)
            label = np.frombuffer(item['label'], dtype=np.uint8).reshape(shape) if 'label' in item else np.zeros_like(raw)

            final_raw = resize_array(raw, self.crop_size, order=1)
            final_label = resize_array(label, self.crop_size, order=0)

            if self.augment:
                final_raw, final_label = self._augment_data(final_raw, final_label)

            raw_tensor = torch.from_numpy(final_raw[np.newaxis, ...])
            label_tensor = torch.from_numpy(final_label).long()

            yield transform_3d(raw_tensor.expand(3, -1, -1, -1)), label_tensor


class HFTrainDataset2D(IterableDataset):
    def __init__(self, hf_dataset: Dataset, crop_size: Tuple[int, int], augment: bool = True, seed: int = 42):
        super().__init__()
        self.dataset = hf_dataset
        self.crop_size = crop_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def _augment_data(self, raw: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < 0.5:
            raw = np.flip(raw, axis=0)
            label = np.flip(label, axis=0)
        if self.rng.random() < 0.5:
            raw = np.flip(raw, axis=1)
            label = np.flip(label, axis=1)

        k = self.rng.integers(0, 4) if raw.shape[0] == raw.shape[1] else self.rng.choice([0, 2])
        if k > 0:
            raw = np.rot90(raw, k=k)
            label = np.rot90(label, k=k)
            
        return raw.copy(), label.copy()

    def __iter__(self):
        while True:
            idx = self.rng.integers(0, len(self.dataset))
            item = self.dataset[idx]

            raw = np.array(item['image'])
            label = np.array(item['label']) if 'label' in item else np.zeros_like(raw)

            final_raw = resize_array(raw, self.crop_size, order=1)
            final_label = resize_array(label, self.crop_size, order=0)

            if self.augment:
                final_raw, final_label = self._augment_data(final_raw, final_label)

            raw_tensor = torch.from_numpy(final_raw[np.newaxis, ...])
            label_tensor = torch.from_numpy(final_label).long()

            yield transform_2d(raw_tensor.expand(3, -1, -1)), label_tensor


class HFValidationDataset3D(IterableDataset):
    def __init__(self, hf_dataset: Dataset, val_crop_size: Tuple[int, int, int]):
        super().__init__()
        self.dataset = hf_dataset
        self.val_crop_size = val_crop_size

    def __iter__(self):
        for item in self.dataset:
            shape = tuple(item['shape'])
            raw = np.frombuffer(item['volume'], dtype=np.uint8).reshape(shape)
            label = np.frombuffer(item['label'], dtype=np.uint8).reshape(shape) if 'label' in item else np.zeros_like(raw)

            final_raw = resize_array(raw, self.val_crop_size, order=1)
            final_label = resize_array(label, self.val_crop_size, order=0)

            raw_tensor = torch.from_numpy(final_raw).unsqueeze(0).expand(3, -1, -1, -1)
            label_tensor = torch.from_numpy(final_label).long()

            yield transform_3d(raw_tensor), label_tensor


class HFValidationDataset2D(IterableDataset):
    def __init__(self, hf_dataset: Dataset, val_crop_size: Tuple[int, int]):
        super().__init__()
        self.dataset = hf_dataset
        self.val_crop_size = val_crop_size

    def __iter__(self):
        for i, item in enumerate(self.dataset):
            raw = np.array(item['image'])
            label = np.array(item['label']) if 'label' in item else np.zeros_like(raw)

            final_raw = resize_array(raw, self.val_crop_size, order=1)
            final_label = resize_array(label, self.val_crop_size, order=0)

            raw_tensor = torch.from_numpy(final_raw).unsqueeze(0).expand(3, -1, -1)
            label_tensor = torch.from_numpy(final_label).long()

            meta = {
                "sample_id": item.get('crop_name', f"val_sample_{i}"),
                "axis": item.get('axis', 'unknown'),
                "slice_idx": item.get('slice', 0)
            }

            yield transform_2d(raw_tensor), label_tensor, meta

# ---------------------------------------------------------
# Loader Builder
# ---------------------------------------------------------

def build_data_loader(
    batch_size: int,
    root_dir: str,
    crop_size: Union[Tuple[int, int], Tuple[int, int, int]], 
    val_crop_size: Union[Tuple[int, int], Tuple[int, int, int]], 
    rank: int = 0,
    world_size: int = 1,
    augment: bool = False,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds Train and Validation loaders. Performs deterministic volume-level splitting if labels are present, 
    otherwise uses 100% of the data for training.
    """
    is_3d = (len(crop_size) == 3)

    # 1. Load the dataset
    try:
        if is_3d:
            dataset = load_from_disk(root_dir)
        else:
            try:
                dataset = load_dataset(root_dir, split="train")
            except Exception:
                dataset = load_from_disk(root_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {root_dir}. Error: {e}")

    # 2. Train / Validation Split Logic
    if "label" in dataset.column_names:
        # We have labeled data -> needs a validation split.
        
        if "crop_name" in dataset.column_names:
            # Extract unique base crop names (stripping _partX from 3D to match 2D)
            all_crop_names = dataset['crop_name']
            unique_crops = sorted(list(set(get_base_crop_name(n) for n in all_crop_names)))
            
            # Deterministically select exactly 64 crops for validation
            rng = np.random.default_rng(42) # Hardcoded seed guarantees consistency
            rng.shuffle(unique_crops)
            val_crop_names = set(unique_crops[:64])
            
            if rank == 0:
                print(f"Detected labeled dataset with crop metadata. Holding out 64 specific volume crops for validation.")
                print(f"Total Unique Volumes: {len(unique_crops)} | Train: {len(unique_crops) - 64} | Val: 64")
                print(f"Val crop names: {val_crop_names}")

            # Use HF batched filtering for extreme speed
            train_hf = dataset.filter(
                lambda batch: [get_base_crop_name(n) not in val_crop_names for n in batch["crop_name"]],
                batched=True,
                num_proc=num_workers,
                desc="Isolating Training Data"
            )
            
            val_hf = dataset.filter(
                lambda batch: [get_base_crop_name(n) in val_crop_names for n in batch["crop_name"]],
                batched=True,
                num_proc=num_workers,
                desc="Isolating Validation Data"
            )
        else:
            # Fallback just in case you ever load a generic labeled dataset without crop names
            if rank == 0:
                print(f"Detected labeled dataset without crop metadata. Falling back to 11% random row split.")
            split_dataset = dataset.train_test_split(test_size=0.11, seed=42)
            train_hf = split_dataset['train']
            val_hf = split_dataset['test']
            
    else:
        # Unlabeled data -> No validation split
        if rank == 0:
             print(f"Detected unlabeled dataset (no 'label' column). Using 100% of data for training.")
        train_hf = dataset
        val_hf = None

    # 3. Apply Distributed Sharding to Validation dataset (Train is iterable and random)
    if val_hf is not None and world_size > 1:
        val_hf = val_hf.shard(num_shards=world_size, index=rank)

    # 4. Instantiate iterators
    if is_3d:
        train_dataset = HFTrainDataset3D(train_hf, crop_size=crop_size, augment=augment)
        val_dataset = HFValidationDataset3D(val_hf, val_crop_size=val_crop_size) if val_hf else None
    else:
        train_dataset = HFTrainDataset2D(train_hf, crop_size=crop_size, augment=augment)
        val_dataset = HFValidationDataset2D(val_hf, val_crop_size=val_crop_size) if val_hf else None

    # 5. Build DataLoaders (num_workers=0 because HF Iterable logic handles data flow internally for now)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) if val_dataset else None

    return train_loader, val_loader