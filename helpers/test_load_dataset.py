# from datasets import load_dataset

# # ds = load_dataset("eminorhan/openorganelle-2d", split="train")
# # ds = load_dataset("eminorhan/cellmap-2d", split="train", download_mode="force_redownload")
# # print(ds[0])
# # print(len(ds))

# ds = load_dataset("eminorhan/cellmap-3d", split="train")

# row = ds[0]

# # Replace bytes with a placeholder string showing the byte length
# summary = {
#     k: f"<{type(v).__name__}, len={len(v)}>" if isinstance(v, bytes) else v 
#     for k, v in row.items()
# }

# print(summary)


import numpy as np
from datasets import load_dataset

def main():
    # Load the dataset in streaming mode so it starts processing immediately
    ds = load_dataset("eminorhan/cellmap-3d", split="train")

    print("Scanning crops for unique labels...\n")
    print("-" * 50)
    
    # Iterate through each row in the dataset
    for i, row in enumerate(ds):
        crop_name = row["crop_name"]
        
        # Convert the raw bytes back into an 8-bit unsigned integer array.
        # No need to reshape() since np.unique processes flat arrays anyway.
        label_array = np.frombuffer(row["label"], dtype=np.uint8)
        
        # Find all unique values in this specific crop
        unique_labels = np.unique(label_array)
        
        # Print the result (converting the numpy array to a standard Python list for cleaner printing)
        print(f"[{i:03d}] {crop_name}")
        print(f"      Unique Labels: {unique_labels.tolist()}")
        print("-" * 50)

if __name__ == "__main__":
    main()