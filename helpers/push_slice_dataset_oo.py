import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Directly upload Parquet chunks to HF Hub.")
    parser.add_argument("--input_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo", help="Directory containing the .parquet files.")
    parser.add_argument("--repo_id", type=str, default="eminorhan/openorganelle-2d", help="Hugging Face repo ID (e.g., 'eminorhan/openorganelle-2d').")
    args = parser.parse_args()

    api = HfApi()

    # Ensure the repo exists (creates it if it doesn't)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)

    print(f"Uploading all Parquet files from {args.input_dir} to {args.repo_id}...")
    
    # upload_folder automatically handles multi-threading and resumes interrupted uploads!
    api.upload_folder(
        folder_path=args.input_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo="data", # Placing them in 'data/' tells HF this is the main dataset
        allow_patterns="*.parquet", # Only upload the parquet files
        commit_message="Upload processed 2D slices as Parquet shards"
    )

    print("Upload complete! Hugging Face will now automatically compile the Parquet files into a unified dataset viewer.")

if __name__ == "__main__":
    main()