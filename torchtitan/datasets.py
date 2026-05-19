import os
import re
import glob
import torch
import numpy as np
import scipy.ndimage
import torch.distributed as dist

from typing import Tuple, Union, List
from torch.utils.data import IterableDataset, DataLoader
from torchvision.transforms import v2
from datasets import load_from_disk, load_dataset

# =========================================================
# 1. TRANSFORMS & AUGMENTATIONS
# =========================================================

def get_base_transforms(is_3d: bool):
    to_float = v2.ToDtype(torch.float32, scale=True)
    if is_3d:
        return v2.Compose([to_float, v2.Normalize(mean=(0.449,), std=(0.226,))])
    return v2.Compose([to_float, v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

def resize_array(arr: np.ndarray, target_shape: Tuple, order: int) -> np.ndarray:
    if arr.shape == target_shape: return arr
    return scipy.ndimage.zoom(arr, [t / s for t, s in zip(target_shape, arr.shape)], order=order, prefilter=False)

def augment_3d(raw: np.ndarray, rng: np.random.Generator, label: np.ndarray = None):
    for axis in range(3):
        if rng.random() < 0.5:
            raw = np.flip(raw, axis=axis)
            if label is not None: label = np.flip(label, axis=axis)
    k = rng.integers(0, 4)
    if k > 0:
        valid_planes = [(0, 1), (0, 2), (1, 2)]
        axes = rng.choice([p for p in valid_planes if raw.shape[p[0]] == raw.shape[p[1]]])
        raw = np.rot90(raw, k=k, axes=axes)
        if label is not None: label = np.rot90(label, k=k, axes=axes)
    return raw.copy(), (label.copy() if label is not None else None)

def augment_2d(raw: np.ndarray, rng: np.random.Generator, label: np.ndarray = None):
    for axis in (0, 1):
        if rng.random() < 0.5:
            raw = np.flip(raw, axis=axis)
            if label is not None: label = np.flip(label, axis=axis)
    k = rng.integers(0, 4) if raw.shape[0] == raw.shape[1] else rng.choice([0, 2])
    if k > 0:
        raw = np.rot90(raw, k=k)
        if label is not None: label = np.rot90(label, k=k)
    return raw.copy(), (label.copy() if label is not None else None)


# =========================================================
# 2. THE 4 EXPLICIT DATASET CLASSES
# =========================================================

class CellMap3DDataset(IterableDataset):
    """Labeled 3D data. Infinite random loop for train, sequential for val."""
    def __init__(self, hf_dataset, crop_size: Tuple, is_val: bool = False, augment: bool = True, seed: int = 42):
        self.dataset = hf_dataset
        self.crop_size = crop_size
        self.is_val = is_val
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.transform = get_base_transforms(is_3d=True)

    def __iter__(self):
        if self.is_val:
            for item in self.dataset:
                shape = tuple(item['shape'])
                raw = np.frombuffer(item['volume'], dtype=np.uint8).reshape(shape)
                label = np.frombuffer(item['label'], dtype=np.uint8).reshape(shape)
                
                raw = resize_array(raw, self.crop_size, 1)
                label = resize_array(label, self.crop_size, 0)
                
                tensor = torch.from_numpy(raw).unsqueeze(0).expand(3, -1, -1, -1)
                yield self.transform(tensor), torch.from_numpy(label).long()
        else:
            while True:
                item = self.dataset[self.rng.integers(0, len(self.dataset))]
                shape = tuple(item['shape'])
                raw = np.frombuffer(item['volume'], dtype=np.uint8).reshape(shape)
                label = np.frombuffer(item['label'], dtype=np.uint8).reshape(shape)
                
                raw = resize_array(raw, self.crop_size, 1)
                label = resize_array(label, self.crop_size, 0)
                
                if self.augment: 
                    raw, label = augment_3d(raw, self.rng, label)
                
                tensor = torch.from_numpy(raw).unsqueeze(0).expand(3, -1, -1, -1)
                yield self.transform(tensor), torch.from_numpy(label).long()


class CellMap2DDataset(IterableDataset):
    """Labeled 2D data. Infinite random loop for train, sequential (with metadata) for val."""
    def __init__(self, hf_dataset, crop_size: Tuple, is_val: bool = False, augment: bool = True, seed: int = 42):
        self.dataset = hf_dataset
        self.crop_size = crop_size
        self.is_val = is_val
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.transform = get_base_transforms(is_3d=False)

    def __iter__(self):
        if self.is_val:
            for i, item in enumerate(self.dataset):
                raw, label = np.array(item['image']), np.array(item['label'])
                raw = resize_array(raw, self.crop_size, 1)
                label = resize_array(label, self.crop_size, 0)
                
                tensor = torch.from_numpy(raw).unsqueeze(0).expand(3, -1, -1)
                meta = {"sample_id": item.get('crop_name', f"val_{i}"), "axis": item.get('axis', 'unk'), "slice": item.get('slice', 0)}
                yield self.transform(tensor), torch.from_numpy(label).long(), meta
        else:
            while True:
                item = self.dataset[self.rng.integers(0, len(self.dataset))]
                raw, label = np.array(item['image']), np.array(item['label'])
                
                raw = resize_array(raw, self.crop_size, 1)
                label = resize_array(label, self.crop_size, 0)
                
                if self.augment: 
                    raw, label = augment_2d(raw, self.rng, label)
                
                tensor = torch.from_numpy(raw).unsqueeze(0).expand(3, -1, -1)
                yield self.transform(tensor), torch.from_numpy(label).long()


class OpenOrganelle2DDataset(IterableDataset):
    """Unlabeled 2D data. Infinite random stream returning images and dummy zero labels."""
    def __init__(self, hf_dataset, crop_size: Tuple, augment: bool = True, seed: int = 42):
        self.dataset = hf_dataset
        self.crop_size = crop_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.transform = get_base_transforms(is_3d=False)

    def __iter__(self):
        while True:
            item = self.dataset[self.rng.integers(0, len(self.dataset))]
            raw = np.array(item['image'])
            raw = resize_array(raw, self.crop_size, 1)
            
            if self.augment: 
                raw, _ = augment_2d(raw, self.rng)
            
            tensor = torch.from_numpy(raw).unsqueeze(0).expand(3, -1, -1)
            dummy_label = torch.zeros(self.crop_size, dtype=torch.long)
            yield self.transform(tensor), dummy_label


class OpenOrganelle3DStreamingDataset(IterableDataset):
    """Unlabeled 400TB 3D data. Lazily streams pre-sharded part_X directories."""
    def __init__(self, part_dirs: List[str], crop_size: Tuple, augment: bool = True, seed: int = 42):
        self.part_dirs = sorted(part_dirs)
        self.crop_size = crop_size
        self.augment = augment
        self.seed = seed
        self.transform = get_base_transforms(is_3d=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        worker_parts = self.part_dirs[worker_id::num_workers]
        if not worker_parts: return

        rng = np.random.default_rng(self.seed + worker_id)

        while True:
            rng.shuffle(worker_parts)
            for part_dir in worker_parts:
                part_ds = load_from_disk(part_dir)
                indices = rng.permutation(len(part_ds))

                for idx in indices:
                    item = part_ds[int(idx)]
                    raw = np.frombuffer(item['volume'], dtype=np.uint8).reshape(tuple(item['shape']))
                    raw = resize_array(raw, self.crop_size, 1)
                    
                    if self.augment: 
                        raw, _ = augment_3d(raw, rng)
                    
                    tensor = torch.from_numpy(raw).unsqueeze(0).expand(3, -1, -1, -1)
                    dummy_label = torch.zeros(self.crop_size, dtype=torch.long)
                    yield self.transform(tensor), dummy_label


# =========================================================
# 3. DETERMINISTIC SPLIT LOGIC (Only used for CellMap)
# =========================================================

def _get_cellmap_splits(dataset, rank, world_size, num_workers):
    """Holds out exactly 64 volumes consistently across ranks."""
    unique_crops = sorted(list(set(re.sub(r'_part\d+$', '', n) for n in dataset['crop_name'])))
    
    rng = np.random.default_rng(42)
    rng.shuffle(unique_crops)
    val_crop_names = set(unique_crops[:64])
    
    if world_size > 1 and rank > 0: dist.barrier()

    train_hf = dataset.filter(
        lambda b: [re.sub(r'_part\d+$', '', n) not in val_crop_names for n in b["crop_name"]],
        batched=True, num_proc=num_workers if rank == 0 else 1, desc="Train Split" if rank == 0 else None
    )
    val_hf = dataset.filter(
        lambda b: [re.sub(r'_part\d+$', '', n) in val_crop_names for n in b["crop_name"]],
        batched=True, num_proc=num_workers if rank == 0 else 1, desc="Val Split" if rank == 0 else None
    )

    if world_size > 1 and rank == 0: dist.barrier()
    
    return train_hf, val_hf


# =========================================================
# 4. EXPLICIT ROUTER
# =========================================================

def build_data_loader(
    dataset_name: str,
    root_dir: str,
    batch_size: int,
    crop_size: Union[Tuple, int], 
    val_crop_size: Union[Tuple, int] = None, 
    rank: int = 0,
    world_size: int = 1,
    augment: bool = False,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    
    if rank == 0:
        print(f"Initializing DataLoader for: {dataset_name.upper()}")

    # ---------------------------------------------------------
    # CASE 1: OpenOrganelle 3D (Unlabeled, Massive 400TB Streamer)
    # ---------------------------------------------------------
    if dataset_name == 'openorganelle-3d':
        part_dirs = sorted(glob.glob(os.path.join(root_dir, "part_*")))
        if not part_dirs: raise FileNotFoundError(f"No part_X folders found in {root_dir}")
        
        rank_assigned_parts = part_dirs[rank::world_size]
        dataset = OpenOrganelle3DStreamingDataset(rank_assigned_parts, crop_size, augment, seed=42+rank)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return train_loader, None

    # ---------------------------------------------------------
    # CASE 2: OpenOrganelle 2D (Unlabeled, Standard Load)
    # ---------------------------------------------------------
    elif dataset_name == 'openorganelle-2d':
        dataset = load_dataset(root_dir, split="train")
        train_dataset = OpenOrganelle2DDataset(dataset, crop_size, augment, seed=42+rank)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        return train_loader, None

    # ---------------------------------------------------------
    # CASE 3: CellMap 3D (Labeled, Needs Validation Split)
    # ---------------------------------------------------------
    elif dataset_name == 'cellmap-3d':
        dataset = load_dataset(root_dir, split="train")
        train_ds, val_ds = _get_cellmap_splits(dataset, rank, world_size, num_workers)
        
        if world_size > 1:
            val_ds = val_ds.shard(num_shards=world_size, index=rank)

        train_dataset = CellMap3DDataset(train_ds, crop_size, is_val=False, augment=augment, seed=42+rank)
        val_dataset = CellMap3DDataset(val_ds, val_crop_size, is_val=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        return train_loader, val_loader

    # ---------------------------------------------------------
    # CASE 4: CellMap 2D (Labeled, Needs Validation Split)
    # ---------------------------------------------------------
    elif dataset_name == 'cellmap-2d':
        dataset = load_dataset(root_dir, split="train")
        train_ds, val_ds = _get_cellmap_splits(dataset, rank, world_size, num_workers)
        
        if world_size > 1:
            val_ds = val_ds.shard(num_shards=world_size, index=rank)

        train_dataset = CellMap2DDataset(train_ds, crop_size, is_val=False, augment=augment, seed=42+rank)
        val_dataset = CellMap2DDataset(val_ds, val_crop_size, is_val=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        return train_loader, val_loader

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Expected openorganelle-2d, openorganelle-3d, cellmap-2d, or cellmap-3d")