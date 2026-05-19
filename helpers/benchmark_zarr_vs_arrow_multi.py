import os
import re
import time
import glob
import zarr
import argparse
import numpy as np
from datasets import load_from_disk
import concurrent.futures

# --- CONFIGURATION ---
ZARR_ROOT_DIR = "/lustre/blizzard/stf218/scratch/emin/seg3d/data"
HF_PARTITIONED_DIR = "/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo_3d"

VOLUMES_TO_BE_INVERTED = [
    "jrc_ccl81-covid-1", "jrc_fly-acc-calyx-1", "jrc_fly-fsb-1", "jrc_hela-4", "jrc_hela-22",
    "jrc_hela-h89-1", "jrc_hela-h89-2", "jrc_hela-nz-1", "jrc_hela-nz-2", "jrc_mus-cerebellum-4",
    "jrc_mus-cerebellum-5", "jrc_mus-cortex-3", "jrc_mus-dorsal-striatum-2", "jrc_mus-dorsal-striatum",
    "jrc_mus-granule-neurons-1", "jrc_mus-granule-neurons-2", "jrc_mus-granule-neurons-3",
    "jrc_mus-hippocampus-2", "jrc_mus-hippocampus-3", "jrc_mus-nacc-2", "jrc_mus-nacc-3",
    "jrc_mus-nacc-4", "jrc_mus-pancreas-3"
]

def get_recon_sort_key(recon_name):
    match = re.search(r'recon-(\d+)', recon_name)
    return int(match.group(1)) if match else float('inf')

def get_em_subfolder_sort_key(folder_name):
    if folder_name.endswith('-uint8'): return 0
    elif folder_name.endswith('-uint8_1'): return 1
    elif folder_name.endswith('-uint16'): return 2
    elif folder_name.endswith('-int16'): return 3
    return 4

def normalize_to_uint8(vol_3d, volume_name):
    if vol_3d.min() == vol_3d.max(): return None
    vol_float = vol_3d.astype(np.float32)
    bg_black, bg_white = vol_float.min(), vol_float.max()
    valid_mask = (vol_float != bg_black) & (vol_float != bg_white) & (vol_float != 0.0)
    valid_pixels = vol_float[valid_mask]
    if valid_pixels.size == 0: return None
    p_low, p_high = np.percentile(valid_pixels, (1.0, 99.0))
    if p_high - p_low == 0:
        p_low, p_high = valid_pixels.min(), valid_pixels.max()
        if p_high - p_low == 0: return None
    normalized = np.clip((vol_float - p_low) / (p_high - p_low), 0, 1)
    if volume_name in VOLUMES_TO_BE_INVERTED:
        normalized = 1.0 - normalized
    return (normalized * 255.0).astype(np.uint8)

# =========================================================================
# WORKER FUNCTIONS
# =========================================================================
def hf_worker(tasks):
    bytes_loaded = 0
    part_tasks = {}
    for task in tasks:
        part_tasks.setdefault(task['part_dir'], []).append(task['idx'])
        
    for p_dir, indices in part_tasks.items():
        ds_part = load_from_disk(p_dir)
        for idx in indices:
            item = ds_part[idx]
            shape = tuple(item['shape'])
            vol_3d = np.frombuffer(item['volume'], dtype=np.uint8).reshape(shape).copy()
            bytes_loaded += vol_3d.nbytes
    return bytes_loaded

def zarr_worker(tasks, mode):
    bytes_loaded = 0
    failures = 0
    
    # 1. Group tasks by volume to avoid redundant I/O
    volume_tasks = {}
    for task in tasks:
        volume_tasks.setdefault(task['volume_name'], []).append(task)
        
    for volume_name, v_tasks in volume_tasks.items():
        try:
            # 2. Do the glob search ONCE per volume
            possible_zarrs = glob.glob(os.path.join(ZARR_ROOT_DIR, f"*/*{volume_name}*.zarr"))
            if not possible_zarrs:
                failures += len(v_tasks)
                continue
                
            # 3. Open the Zarr store and traverse metadata ONCE per volume
            zarr_path = possible_zarrs[0]
            zarr_root = zarr.open(zarr_path, mode='r')
            
            recon_keys = [k for k in zarr_root.keys() if k.startswith('recon-')]
            earliest_recon = sorted(recon_keys, key=get_recon_sort_key)[0]
            em_subfolders = list(zarr_root[f"{earliest_recon}/em"].keys())
            best_em = sorted(em_subfolders, key=get_em_subfolder_sort_key)[0]
            
            # Keep a persistent reference to the array
            s0_array = zarr_root[f"{earliest_recon}/em/{best_em}/s0"]
            
            # 4. NOW loop through the crops and do pure data fetching
            for task in v_tasks:
                z_s, y_s, x_s = task['z_start'], task['y_start'], task['x_start']
                z_e, y_e, x_e = z_s + 512, y_s + 512, x_s + 512
                
                # Fetching the payload
                vol_3d = np.array(s0_array[z_s:z_e, y_s:y_e, x_s:x_e])

                if mode == 'conservative':
                    bytes_loaded += vol_3d.nbytes
                elif mode == 'realistic':
                    bg_min, bg_max = vol_3d.min(), vol_3d.max()
                    is_foreground = (vol_3d != bg_min) & (vol_3d != bg_max) & (vol_3d != 0)
                    if np.any(is_foreground):
                        z_any, y_any, x_any = np.any(is_foreground, axis=(1, 2)), np.any(is_foreground, axis=(0, 2)), np.any(is_foreground, axis=(0, 1))
                        zmin, zmax = np.where(z_any)[0][[0, -1]]
                        ymin, ymax = np.where(y_any)[0][[0, -1]]
                        xmin, xmax = np.where(x_any)[0][[0, -1]]
                        
                        vol_crop = vol_3d[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
                        vol_uint8 = normalize_to_uint8(vol_crop, volume_name)
                        if vol_uint8 is not None:
                            bytes_loaded += vol_uint8.nbytes
                            
        except Exception as e:
            failures += len(v_tasks)
            
    return bytes_loaded, failures

# =========================================================================
# ORCHESTRATOR
# =========================================================================
def run_concurrent_benchmark(mode, num_parts, samples_per_part, num_workers):
    all_part_dirs = sorted(glob.glob(os.path.join(HF_PARTITIONED_DIR, "part_*")))
    if not all_part_dirs:
        print(f"Error: No part folders found at {HF_PARTITIONED_DIR}")
        return

    rng = np.random.default_rng(42)
    sampled_part_dirs = rng.choice(all_part_dirs, size=min(num_parts, len(all_part_dirs)), replace=False)
    
    print(f"\n--- CONCURRENT BENCHMARK INITIALIZED ---")
    print(f"Mode: {mode.upper()} | Workers: {num_workers}")
    print(f"Sampling {samples_per_part} chunks from {len(sampled_part_dirs)} parts.")
    
    tasks = []
    for p_dir in sampled_part_dirs:
        ds_part = load_from_disk(p_dir)
        indices = rng.choice(len(ds_part), size=min(samples_per_part, len(ds_part)), replace=False)
        for idx in indices:
            item = ds_part[int(idx)]
            tasks.append({
                'part_dir': p_dir, 'idx': int(idx), 'volume_name': item['volume_name'],
                'z_start': item['z_start'], 'y_start': item['y_start'], 'x_start': item['x_start']
            })

    total_chunks = len(tasks)
    print(f"Total Bounding Boxes to fetch concurrently: {total_chunks}\n")

    chunk_size = max(1, total_chunks // num_workers)
    task_chunks = [tasks[i:i + chunk_size] for i in range(0, total_chunks, chunk_size)]

    # We use a single ProcessPoolExecutor for the entire benchmark to act like PyTorch persistent workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        print("Warming up Python worker processes (Mimicking PyTorch persistent_workers=True)...")
        # Submit tiny sleep tasks just to force the OS to boot the processes before we start the timer
        list(executor.map(time.sleep, [0.01] * num_workers))

        # ---------------------------------------------------------
        # PIPELINE A: Hugging Face Arrow Stream
        # ---------------------------------------------------------
        print(f"Testing Hugging Face Arrow...")
        start_hf = time.perf_counter()
        hf_bytes_loaded = 0
        
        futures = [executor.submit(hf_worker, chunk) for chunk in task_chunks]
        for future in concurrent.futures.as_completed(futures):
            hf_bytes_loaded += future.result()

        end_hf = time.perf_counter()
        hf_duration = end_hf - start_hf
        hf_throughput = total_chunks / hf_duration
        print(f"-> HF Arrow finished: {hf_duration:.4f} sec ({hf_throughput:.2f} chunks/sec)\n")

        # ---------------------------------------------------------
        # PIPELINE B: Zarr I/O
        # ---------------------------------------------------------
        print(f"Testing Zarr...")
        start_zarr = time.perf_counter()
        zarr_bytes_loaded = 0
        total_failures = 0
        
        futures = [executor.submit(zarr_worker, chunk, mode) for chunk in task_chunks]
        for future in concurrent.futures.as_completed(futures):
            b_loaded, fails = future.result()
            zarr_bytes_loaded += b_loaded
            total_failures += fails

        end_zarr = time.perf_counter()
        zarr_duration = end_zarr - start_zarr
        valid_zarr_chunks = total_chunks - total_failures
        zarr_throughput = valid_zarr_chunks / zarr_duration if valid_zarr_chunks > 0 else 0
        print(f"-> Raw Zarr finished: {zarr_duration:.4f} sec ({zarr_throughput:.2f} chunks/sec)\n")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("="*65)
    print(f"       CONCURRENT METRICS ({num_workers} WORKERS | {mode.upper()})")
    print("="*65)
    print(f"Zarr Throughput:     {zarr_throughput:7.2f} chunks/sec  ({(zarr_bytes_loaded / zarr_duration) / 1024**2:7.2f} MB/s)")
    print(f"HF Arrow Throughput: {hf_throughput:7.2f} chunks/sec  ({(hf_bytes_loaded / hf_duration) / 1024**2:7.2f} MB/s)")
    print("-" * 65)
    print(f"Concurrency Speedup: {hf_throughput / max(1e-5, zarr_throughput):.2f}x FASTER using Arrow pipeline.")
    print("="*65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent Benchmark: HF Arrow vs Zarr.")
    parser.add_argument("--mode", type=str, choices=["realistic", "conservative"], default="realistic")
    parser.add_argument("--num_parts", type=int, default=10, help="Number of HF part directories to sample from.")
    # INCREASED default samples to simulate realistic streaming and amortize the metadata load
    parser.add_argument("--samples_per_part", type=int, default=500, help="Number of 3D chunks to fetch per part.")
    parser.add_argument("--num_workers", type=int, default=128, help="Number of concurrent multiprocessing workers.")
    
    args = parser.parse_args()
    
    run_concurrent_benchmark(mode=args.mode, num_parts=args.num_parts, samples_per_part=args.samples_per_part, num_workers=args.num_workers)