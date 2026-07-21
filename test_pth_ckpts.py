import os
import re
import random
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from datasets import load_dataset
from dinov3.eval.segmentation.models import build_segmentation_decoder
from torchtitan.datasets import CellMap2DDataset 

# =========================================================
# 1. CONFIGURATION & ALLOWED CROPS
# =========================================================
DINOV3_REPO_PATH = "/lustre/blizzard/stf218/scratch/emin/dinov3"
CROP_SIZE = (512, 512)
VAL_CROP_SIZE = (512, 512, 512) 
SEED = 15

CKPT_LVD = "2d_l_last_lvd_3em4_32_1_it10k-8aa4cbdd.pth"
CKPT_SCRATCH = "2d_l_last_scratch_3em4_32_1_it10k-8aa4cbdd.pth"

ALLOWED_CROPS = {
    'jrc_cos7-1a/recon-1/crop234', 'jrc_cos7-1a/recon-1/crop237', 
    'jrc_cos7-1a/recon-1/crop239', 'jrc_cos7-1b/recon-1/crop235', 
    'jrc_cos7-1b/recon-1/crop255', 'jrc_ctl-id8-1/recon-1/crop119', 
    'jrc_hela-2/recon-1/crop16', 'jrc_hela-2/recon-1/crop3', 
    'jrc_hela-2/recon-1/crop8', 'jrc_jurkat-1/recon-1/crop66', 
    'jrc_jurkat-1/recon-1/crop93', 'jrc_mus-kidney/recon-1/crop148', 
    'jrc_mus-kidney/recon-1/crop156', 'jrc_mus-kidney/recon-1/crop158', 
    'jrc_mus-liver-zon-1/recon-1/crop267', 'jrc_mus-liver-zon-1/recon-1/crop313', 
    'jrc_mus-liver-zon-1/recon-1/crop319', 'jrc_mus-liver-zon-1/recon-1/crop336', 
    'jrc_mus-liver-zon-1/recon-1/crop337', 'jrc_mus-liver-zon-1/recon-1/crop349', 
    'jrc_mus-liver-zon-2/recon-1/crop357', 'jrc_mus-liver-zon-2/recon-1/crop368', 
    'jrc_mus-liver/recon-1/crop125', 'jrc_mus-liver/recon-1/crop417', 
    'jrc_sum159-4/recon-1/crop202', 'jrc_sum159-4/recon-1/crop211', 
    'jrc_sum159-4/recon-1/crop213', 'jrc_sum159-4/recon-1/crop217', 
    'jrc_ut21-1413-003/recon-1/crop197', 'jrc_ut21-1413-003/recon-1/crop199', 
    'jrc_ut21-1413-003/recon-1/crop228', 'jrc_zf-cardiac-1/recon-1/crop379'
}

# Class-index mapping
CLASS_MAPPING = {
    'ecs': 0, 'pm': 1, 'mito_mem': 2, 'mito_lum': 3, 'mito_ribo': 4, 'golgi_mem': 5, 
    'golgi_lum': 6, 'ves_mem': 7, 'ves_lum': 8, 'endo_mem': 9, 'endo_lum': 10, 
    'lyso_mem': 11, 'lyso_lum': 12, 'ld_mem': 13, 'ld_lum': 14, 'er_mem': 15, 
    'er_lum': 16, 'eres_mem': 17, 'eres_lum': 18, 'ne_mem': 19, 'ne_lum': 20, 
    'np_out': 21, 'np_in': 22, 'hchrom': 23, 'nhchrom': 24, 'echrom': 25, 
    'nechrom': 26, 'nucpl': 27, 'nucleo': 28, 'mt_out': 29, 'cent': 30, 
    'cent_dapp': 31, 'cent_sdapp': 32, 'ribo': 33, 'cyto': 34, 'mt_in': 35, 
    'nuc': 36, 'vim': 37, 'glyco': 38, 'golgi': 39, 'ves': 40, 'endo': 41, 
    'lyso': 42, 'ld': 43, 'rbc': 44, 'eres': 45, 'perox_mem': 46, 'perox_lum': 47, 
    'perox': 48, 'mito': 49, 'er': 50, 'ne': 51, 'np': 52, 'chrom': 53, 
    'mt': 54, 'isg_mem': 55, 'isg_lum': 56, 'isg_ins': 57, 'isg': 58, 'cell': 59, 
    'actin': 60, 'tbar': 61, 'bm': 62, 'er_mem_all': 63, 'ne_mem_all': 64, 
    'cent_all': 65, 'chlor_mem': 66, 'chlor_lum': 67, 'chlor_sg': 68, 
    'chlor': 69, 'vac_mem': 70, 'vac_lum': 71, 'vac': 72, 'pd': 73
}
IDX_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

# =========================================================
# 2. FAST DATASET PREPARATION & SAMPLING
# =========================================================
print("Loading HuggingFace 3D dataset...")
raw_dataset = load_dataset("eminorhan/cellmap-3d", split="train")

print("Filtering crops using indices...")
all_crop_names = raw_dataset['crop_name']

valid_indices = [
    i for i, name in enumerate(all_crop_names) 
    if re.sub(r'_part\d+$', '', name) in ALLOWED_CROPS
]

filtered_dataset = raw_dataset.select(valid_indices)

random.seed(SEED)
np.random.seed(SEED)

TOTAL_SLICES_PER_VOL = sum(VAL_CROP_SIZE) 
# We now sample 8 volumes instead of 6
sampled_vol_indices = random.sample(range(len(filtered_dataset)), 8)
selected_samples = []

print("Processing sampled volumes...")
for idx in sampled_vol_indices:
    single_vol_ds = filtered_dataset.select([idx])
    crop_name = single_vol_ds[0]['crop_name']
    
    val_dataset = CellMap2DDataset(
        single_vol_ds, 
        crop_size=VAL_CROP_SIZE, 
        is_val=True, 
        augment=False, 
        seed=SEED
    )
    
    target_slice_idx = random.randint(0, TOTAL_SLICES_PER_VOL - 1)
    for i, (img_tensor, label_tensor, meta) in enumerate(val_dataset):
        if i == target_slice_idx:
            selected_samples.append((img_tensor, label_tensor, meta))
            print(f"  Got slice from {crop_name} (axis={meta['axis']}, idx={meta['slice_idx']})")
            break 

# =========================================================
# 3. MODEL INITIALIZATION & INFERENCE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segmentation_model(ckpt_path: str, dinov3_path: str):
    print(f"Loading model checkpoint: {ckpt_path}")
    bbone = torch.hub.load(
        dinov3_path, 
        "dinov3_vitl16", 
        source="local", 
        pretrained=False, 
        use_fa4=True
    )
    model = build_segmentation_decoder(
        bbone,
        backbone_out_layers="last",
        decoder_type="linear", 
        num_classes=64
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

model_lvd = load_segmentation_model(CKPT_LVD, DINOV3_REPO_PATH)
model_scratch = load_segmentation_model(CKPT_SCRATCH, DINOV3_REPO_PATH)

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    res = tensor * std + mean
    return res.permute(1, 2, 0).numpy()

inputs_vis = []
labels_vis = []
preds_lvd = []
preds_scratch = []
titles = []

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    for img_tensor, label_tensor, meta in selected_samples:
        x = img_tensor.unsqueeze(0).to(device)
        
        # Forward passes
        out_lvd = model_lvd(x)
        out_scratch = model_scratch(x)
        
        # Upsample predictions to match CROP_SIZE using bilinear interpolation
        out_lvd = F.interpolate(out_lvd, size=CROP_SIZE, mode='bilinear', align_corners=False)
        out_scratch = F.interpolate(out_scratch, size=CROP_SIZE, mode='bilinear', align_corners=False)
                
        pred_lvd = out_lvd.argmax(dim=1).squeeze(0).cpu().numpy()
        pred_scratch = out_scratch.argmax(dim=1).squeeze(0).cpu().numpy()
        
        inputs_vis.append(denormalize(img_tensor))
        labels_vis.append(label_tensor.numpy())
        preds_lvd.append(pred_lvd)
        preds_scratch.append(pred_scratch)
        
        # Keep only the full sample_id for the title
        titles.append(meta['sample_id'])

# =========================================================
# 4. PLOTTING THE 4x8 GRID
# =========================================================
np.random.seed(SEED)
colors = np.random.rand(NUM_CLASSES, 3)
colors[0] = [0, 0, 0]  # Background class (ecs) stays black
seg_cmap = mcolors.ListedColormap(colors)

# Setting gridspec_kw heavily enforces zero spacing at the subplot level
# Increased figsize width to 24 to accommodate 8 columns while maintaining aspect ratio
fig, axes = plt.subplots(4, 8, figsize=(24, 12), gridspec_kw={'wspace': 0, 'hspace': 0})

row_titles = [
    "raw slice",
    "ground truth",
    "lvd (10k)",
    "scratch (10k)"
]

# Changed loop limit to 8
for col in range(8):
    # 1. Raw slice (aspect='auto' removes padding and fills grid exactly)
    axes[0, col].imshow(inputs_vis[col], aspect='auto')
    
    # We use a slightly smaller font size (9 instead of 10) to accommodate the full, longer name
    axes[0, col].set_title(titles[col], fontsize=9, fontweight='bold', pad=8)
    
    # 2. Ground Truth
    gt_label = labels_vis[col]
    axes[1, col].imshow(gt_label, cmap=seg_cmap, vmin=0, vmax=NUM_CLASSES-1, aspect='auto')
    
    # Annotate GT map: find the largest connected component for each class present to place the text label
    unique_classes = np.unique(gt_label)
    for c_idx in unique_classes:
        if c_idx == 0: continue # Skip 'ecs' background to prevent clutter
        
        mask = (gt_label == c_idx)
        labeled_mask, num_features = scipy.ndimage.label(mask)
        if num_features == 0: continue
        
        # Determine the largest chunk of this class so the text doesn't float in empty space
        sizes = scipy.ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
        largest_idx = np.argmax(sizes) + 1
        y, x = scipy.ndimage.center_of_mass(mask, labeled_mask, largest_idx)
        
        class_name = IDX_TO_CLASS.get(c_idx, str(c_idx))
        txt = axes[1, col].text(x, y, class_name, color='white', 
                                fontsize=9, ha='center', va='center', fontweight='bold')
        # Add a black stroke behind the text so it's readable over any color
        txt.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground='black')])

    # 3. LVD Prediction
    axes[2, col].imshow(preds_lvd[col], cmap=seg_cmap, vmin=0, vmax=NUM_CLASSES-1, aspect='auto')
    
    # 4. Scratch Prediction
    axes[3, col].imshow(preds_scratch[col], cmap=seg_cmap, vmin=0, vmax=NUM_CLASSES-1, aspect='auto')

# Format the grid: eliminate white space, enforce thin black borders, add row labels
for row in range(4):
    axes[row, 0].set_ylabel(row_titles[row], fontsize=12, fontweight='bold', labelpad=10)
    # Changed loop limit to 8
    for col in range(8):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        
        # Enforce fine black borders on every subplot
        for spine in axes[row, col].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)  # <-- Reduced border thickness
            spine.set_visible(True)

# Adjust the outer margins of the figure (interior spacing is handled by gridspec)
fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.02)

plt.savefig(f"cellmap_2d_predictions_comparison_{SEED}.pdf", dpi=300, bbox_inches='tight')
print("Saved predictions to cellmap_2d_predictions_comparison")