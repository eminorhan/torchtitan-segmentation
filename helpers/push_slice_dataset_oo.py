import os
import argparse
from glob import glob
from datasets import load_from_disk, concatenate_datasets

def main():
    parser = argparse.ArgumentParser(description="Merge and push dataset to Hugging Face Hub.")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo", help="Directory containing part_* folders")
    parser.add_argument("--repo_id", type=str, default="eminorhan/openorganelle-2d", help="HF repo id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")    
    args = parser.parse_args()

    # Find all part directories and sort them numerically
    # (Sorting "part_2" before "part_10" instead of alphabetically)
    part_dirs = glob(os.path.join(args.local_save_dir, "part_*"))
    part_dirs = sorted(part_dirs, key=lambda x: int(x.split('_')[-1]))
    
    if not part_dirs:
        print(f"No parts found in {args.local_save_dir}. Exiting.")
        return

    print(f"Found {len(part_dirs)} dataset parts. Loading...")

    # Load all individual datasets
    loaded_datasets = []
    for part_path in part_dirs:
        print(f"Loading {os.path.basename(part_path)}...")
        ds = load_from_disk(part_path)
        loaded_datasets.append(ds)

    # Concatenate them into one massive dataset
    print("\nConcatenating parts (this is memory-mapped and should be fast)...")
    full_dataset = concatenate_datasets(loaded_datasets)
    print(f"Concatenation complete! Total slices/chunks: {len(full_dataset)}")

    # Shuffle the entire concatenated dataset
    print(f"\nShuffling the complete dataset globally (seed={args.seed})...")
    print("Note: For massive datasets, this generates an index mapping map and might take a minute or two.")
    full_dataset = full_dataset.shuffle(seed=args.seed)
    print("Shuffle complete!")

    # Push to Hugging Face Hub
    print(f"\nPushing to Hub repository: {args.repo_id}...")
    print("The library will automatically shard the dataset and upload it as Parquet files.")
    
    full_dataset.push_to_hub(
        repo_id=args.repo_id,
        max_shard_size="1GB"  # keep individual hub files manageable
    )
    
    print("\nUpload complete! Your dataset is now live on the Hugging Face Hub.")

if __name__ == "__main__":
    main()