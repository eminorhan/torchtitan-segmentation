from datasets import load_dataset

# ds = load_dataset("eminorhan/openorganelle-2d", split="train")
# ds = load_dataset("eminorhan/cellmap-2d", split="train", download_mode="force_redownload")
ds = load_dataset("eminorhan/cellmap-3d", split="train", download_mode="force_redownload")

print(ds[0])
print(len(ds))
