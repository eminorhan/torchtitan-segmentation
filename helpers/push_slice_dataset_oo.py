import os
import time
import argparse
from glob import glob
from datasets import load_from_disk, concatenate_datasets
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Merge and push dataset to Hugging Face Hub.")
    parser.add_argument("--local_save_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo_filtered", help="Directory containing part_* folders")
    parser.add_argument("--repo_id", type=str, default="eminorhan/openorganelle-2d", help="HF repo id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")    
    parser.add_argument("--num_shards", type=int, default=500, help="Target ~1GB per shard (e.g., 6000 for a ~6TB dataset)")
    
    # Flag to clear out the old dataset before uploading
    parser.add_argument("--overwrite", action="store_true", help="Delete the existing 'data' directory on the Hub to force a clean update.")
    
    args = parser.parse_args()

    # Initialize Hugging Face API
    api = HfApi()

    # Create the repo if it doesn't exist yet
    print(f"Ensuring repository '{args.repo_id}' exists...")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    # Delete old data if updating the dataset
    if args.overwrite:
        print("\n⚠️ OVERWRITE FLAG DETECTED: Deleting old 'data' directory on the Hub...")
        try:
            # This creates a single commit that drops the entire data/ folder
            api.delete_folder(
                path_in_repo="data", 
                repo_id=args.repo_id, 
                repo_type="dataset",
                commit_message="Clear old dataset shards for update"
            )
            print("✅ Successfully deleted old data. Starting with a clean slate.")
        except Exception as e:
            # If the folder doesn't exist yet, it will throw an error, which is fine to ignore
            print(f"Note: Could not delete 'data' folder (it might not exist yet): {e}")

    # Get existing files on the Hub to enable instant resuming
    print("\nChecking Hub for existing files to resume...")
    try:
        existing_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
        existing_set = set(existing_files)
        print(f"Found {len(existing_set)} files already safely on the Hub.")
    except Exception as e:
        print(f"Warning: Could not list repo files: {e}")
        existing_set = set()

    # Find all part directories and sort them numerically
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

    # Robust upload logic
    num_shards = args.num_shards
    print(f"\nInitiating resumable upload across {num_shards} shards.")

    for i in range(num_shards):
        file_name = f"data/train-{i:05d}-of-{num_shards:05d}.parquet"

        if file_name in existing_set:
            print(f"✅ Shard {i}/{num_shards} already exists. Skipping.")
            continue

        print(f"⏳ Processing Shard {i}/{num_shards}...")

        shard = full_dataset.shard(num_shards=num_shards, index=i, contiguous=True)
        temp_file = f"temp_upload_shard_{i}.parquet"

        shard.to_parquet(temp_file)

        success = False
        for attempt in range(10):
            try:
                print(f"   ⬆️ Uploading {file_name} (Attempt {attempt + 1}/10)...")
                api.upload_file(
                    path_or_fileobj=temp_file,
                    path_in_repo=file_name,
                    repo_id=args.repo_id,
                    repo_type="dataset"
                )
                success = True
                break
            except Exception as e:
                print(f"   ❌ Network error: {e}. Retrying in 15 seconds...")
                time.sleep(15)

        if os.path.exists(temp_file):
            os.remove(temp_file)

        if not success:
            print(f"💥 FATAL: Failed to upload shard {i} after 10 attempts. Exiting.")
            return

    print("\n🎉 Upload complete! Your dataset is now live on the Hugging Face Hub.")

if __name__ == "__main__":
    main()