import os
import argparse
from glob import glob
from datasets import load_from_disk, concatenate_datasets

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_root", type=str, required=True, help="Directory containing chunk_* datasets")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID, e.g. username/dataset-name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    chunk_paths = sorted(glob(os.path.join(args.chunk_root, "chunk_*")))
    print(f"Found {len(chunk_paths)} chunks")

    if not chunk_paths:
        raise RuntimeError("No chunks found. Check --chunk_root path.")

    datasets = []
    for p in chunk_paths:
        print(f"Loading {p}")
        datasets.append(load_from_disk(p))

    print("Concatenating...")
    full_ds = concatenate_datasets(datasets)
    print(full_ds)

    print("Shuffling...")
    full_ds = full_ds.shuffle(seed=args.seed)

    print("Pushing to hub...")
    full_ds.push_to_hub(args.repo_id)

    print("Done!")

if __name__ == "__main__":
    main()