import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from datasets import load_from_disk

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

def get_cmap_and_norm():
    max_label_id = max(CLASS_MAPPING.values()) + 1
    colors = plt.cm.get_cmap('gist_ncar', max_label_id)
    new_colors = colors(np.linspace(0, 1, max_label_id))
    new_colors[0, :] = np.array([0, 0, 0, 0]) # Transparent background class
    custom_cmap = ListedColormap(new_colors)
    norm = BoundaryNorm(np.arange(-0.5, max_label_id, 1), custom_cmap.N)
    return custom_cmap, norm

def visualize_volume_gif(volume_array, label_array, metadata, output_path, fps=10):
    """Creates a GIF animating through the Z-axis of the 3D volume with segmentation masks overlaid."""
    z_slices = volume_array.shape[0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    custom_cmap, norm = get_cmap_and_norm()

    # Initialize with the first slice
    im_vol = ax.imshow(volume_array[0], cmap='gray', vmin=0, vmax=255)
    im_lbl = ax.imshow(label_array[0], cmap=custom_cmap, norm=norm, alpha=0.1)
    
    # Text box for metadata overlays
    title_text = ax.text(10, 20, "", color='white', fontsize=8, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    def update(z_idx):
        im_vol.set_data(volume_array[z_idx])
        im_lbl.set_data(label_array[z_idx])
        title_text.set_text(
            f"Crop: {metadata['crop_name']}\n"
            f"Shape: {volume_array.shape}\n"
            f"Z-Slice: {z_idx}/{z_slices-1}"
        )
        return [im_vol, im_lbl, title_text]

    ani = FuncAnimation(fig, update, frames=z_slices, interval=1000/fps, blit=False)
    ani.save(output_path, writer='pillow')
    plt.close(fig)

def visualize_orthogonal_views(volume_array, label_array, metadata, output_path):
    """Saves a static high-res image of the 25%, 50%, and 75% Z, Y, and X slices to verify un-dithered quality."""
    z_dim, y_dim, x_dim = volume_array.shape
    z_slices = [z_dim // 4, z_dim // 2, 3 * z_dim // 4]
    y_slices = [y_dim // 4, y_dim // 2, 3 * y_dim // 4]
    x_slices = [x_dim // 4, x_dim // 2, 3 * x_dim // 4]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f"Crop: {metadata['crop_name']} | Shape: {volume_array.shape}", color='black', fontsize=12, fontweight='bold')
    
    custom_cmap, norm = get_cmap_and_norm()

    for i, z in enumerate(z_slices):
        axes[0, i].imshow(volume_array[z, :, :], cmap='gray', vmin=0, vmax=255)
        axes[0, i].imshow(label_array[z, :, :], cmap=custom_cmap, norm=norm, alpha=0.1)
        axes[0, i].text(10, 20, f"Z-Slice (XY Plane) at Z={z}", color='white', fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        axes[0, i].axis('off')
        
    for i, y in enumerate(y_slices):
        axes[1, i].imshow(volume_array[:, y, :], cmap='gray', vmin=0, vmax=255)
        axes[1, i].imshow(label_array[:, y, :], cmap=custom_cmap, norm=norm, alpha=0.1)
        axes[1, i].text(10, 20, f"Y-Slice (XZ Plane) at Y={y}", color='white', fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        axes[1, i].axis('off')
        
    for i, x in enumerate(x_slices):
        axes[2, i].imshow(volume_array[:, :, x], cmap='gray', vmin=0, vmax=255)
        axes[2, i].imshow(label_array[:, :, x], cmap=custom_cmap, norm=norm, alpha=0.1)
        axes[2, i].text(10, 20, f"X-Slice (YZ Plane) at X={x}", color='white', fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        axes[2, i].axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Test and visualize random volumes and masks from saved CellMap datasets.")
    parser.add_argument("--data_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_cellmap_3d", help="Directory of the dataset")
    parser.add_argument("--num_samples", type=int, default=9, help="Number of random samples to visualize")
    parser.add_argument("--output_dir", type=str, default="test_3d_visualizations_cellmap", help="Output directory for GIFs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from {args.data_dir}...")
    ds = load_from_disk(args.data_dir)
    
    if len(ds) == 0:
        print("Dataset is empty.")
        return
        
    print(f"Dataset has {len(ds)} samples. Randomly selecting {args.num_samples} samples...")
    
    num_to_sample = min(args.num_samples, len(ds))
    sampled_indices = random.sample(range(len(ds)), num_to_sample)

    for i, row_idx in enumerate(sampled_indices):
        print(f"\n[{i+1}/{num_to_sample}] Loading random row (index {row_idx})...")
        row = ds[row_idx]
        
        # Reconstruct the 3D numpy arrays from bytes and shape
        shape = tuple(row["shape"])
        vol_3d = np.frombuffer(row["volume"], dtype=np.uint8).reshape(shape)
        label_3d = np.frombuffer(row["label"], dtype=np.uint8).reshape(shape)
        
        metadata = {"crop_name": row["crop_name"]}
        
        # Avoid creating directory structures out of paths
        safe_crop_name = metadata['crop_name'].replace('/', '_')
        
        print(f"  -> Reconstructed Crop: {metadata['crop_name']}")
        print(f"  -> Shape: {shape} | Vol Min: {vol_3d.min()} | Vol Max: {vol_3d.max()}")
        print(f"  -> Label Min: {label_3d.min()} | Label Max: {label_3d.max()}")
        
        gif_path = os.path.join(args.output_dir, f"sample_{i+1}_{safe_crop_name}.gif")
        ortho_path = os.path.join(args.output_dir, f"sample_{i+1}_{safe_crop_name}_ortho.png")
        
        visualize_volume_gif(vol_3d, label_3d, metadata, gif_path)
        visualize_orthogonal_views(vol_3d, label_3d, metadata, ortho_path)
        
        print(f"  -> Saved visualizations to {gif_path} and {ortho_path}")

if __name__ == '__main__':
    main()