from datasets import load_dataset

# ds = load_dataset("eminorhan/openorganelle-2d", split="train")
ds = load_dataset("eminorhan/cellmap-2d", split="train")

print(ds[0])
print(len(ds))
