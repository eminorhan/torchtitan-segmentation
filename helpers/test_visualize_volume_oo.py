import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datasets import load_from_disk

def visualize_volume_gif(volume_array, metadata, output_path, fps=10):
    """Creates a GIF animating through the Z-axis of the 3D volume."""
    z_slices = volume_array.shape[0]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')

    # Initialize with the first slice
    im = ax.imshow(volume_array[0], cmap='gray', vmin=0, vmax=255)
    
    # Text box for metadata overlays
    title_text = ax.text(10, 20, "", color='white', fontsize=8, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

    def update(z_idx):
        im.set_data(volume_array[z_idx])
        title_text.set_text(
            f"Vol: {metadata['volume_name']}\n"
            f"Shape: {volume_array.shape}\n"
            f"Starts (Z,Y,X): {metadata['z_start']}, {metadata['y_start']}, {metadata['x_start']}\n"
            f"Z-Slice: {z_idx}/{z_slices-1}"
        )
        return [im, title_text]

    ani = FuncAnimation(fig, update, frames=z_slices, interval=1000/fps, blit=False)
    ani.save(output_path, writer='pillow')
    plt.close(fig)

def visualize_orthogonal_views(volume_array, metadata, output_path):
    """Saves a static high-res image of the 25%, 50%, and 75% Z, Y, and X slices to verify un-dithered quality."""
    z_dim, y_dim, x_dim = volume_array.shape
    z_slices = [z_dim // 4, z_dim // 2, 3 * z_dim // 4]
    y_slices = [y_dim // 4, y_dim // 2, 3 * y_dim // 4]
    x_slices = [x_dim // 4, x_dim // 2, 3 * x_dim // 4]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f"Volume: {metadata['volume_name']} | Shape: {volume_array.shape}\n"
                 f"Starts (Z,Y,X): {metadata['z_start']}, {metadata['y_start']}, {metadata['x_start']}", 
                 color='black', fontsize=12, fontweight='bold')
    
    for i, z in enumerate(z_slices):
        axes[0, i].imshow(volume_array[z, :, :], cmap='gray', vmin=0, vmax=255)
        axes[0, i].text(10, 20, f"Z-Slice (XY Plane) at Z={z}", color='white', fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        axes[0, i].axis('off')
        
    for i, y in enumerate(y_slices):
        axes[1, i].imshow(volume_array[:, y, :], cmap='gray', vmin=0, vmax=255)
        axes[1, i].text(10, 20, f"Y-Slice (XZ Plane) at Y={y}", color='white', fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        axes[1, i].axis('off')
        
    for i, x in enumerate(x_slices):
        axes[2, i].imshow(volume_array[:, :, x], cmap='gray', vmin=0, vmax=255)
        axes[2, i].text(10, 20, f"X-Slice (YZ Plane) at X={x}", color='white', fontsize=10, fontweight='bold', va='top', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
        axes[2, i].axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Test and visualize random volumes from saved arrow datasets.")
    parser.add_argument("--data_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data_oo_3d", help="Directory with part_* folders")
    parser.add_argument("--num_samples", type=int, default=9, help="Number of random samples to visualize")
    parser.add_argument("--output_dir", type=str, default="test_visualizations", help="Output directory for GIFs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all part directories
    part_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if d.startswith("part_")]
    if not part_dirs:
        print(f"No part_* directories found in {args.data_dir}.")
        return
    
    print(f"Found {len(part_dirs)} part directories. Randomly selecting {args.num_samples} parts...")
    sampled_parts = random.sample(part_dirs, min(args.num_samples, len(part_dirs)))

    for i, part_dir in enumerate(sampled_parts):
        print(f"\n[{i+1}/{len(sampled_parts)}] Loading random row from {os.path.basename(part_dir)}...")
        ds = load_from_disk(part_dir)
        
        if len(ds) == 0:
            print("  -> Dataset part is empty, skipping.")
            continue
        
        # Pick a random row from this specific chunk
        row_idx = random.randint(0, len(ds) - 1)
        row = ds[row_idx]
        
        # Reconstruct the 3D numpy array from bytes and shape
        shape = tuple(row["shape"])
        vol_3d = np.frombuffer(row["volume"], dtype=np.uint8).reshape(shape)
        
        metadata = {k: row[k] for k in ["volume_name", "z_start", "y_start", "x_start"]}
        
        print(f"  -> Reconstructed Volume: {metadata['volume_name']}")
        print(f"  -> Shape: {shape} | Min: {vol_3d.min()} | Max: {vol_3d.max()}")
        
        gif_path = os.path.join(args.output_dir, f"sample_{i+1}_{metadata['volume_name']}_z{metadata['z_start']}.gif")
        ortho_path = os.path.join(args.output_dir, f"sample_{i+1}_{metadata['volume_name']}_z{metadata['z_start']}_ortho.png")
        
        visualize_volume_gif(vol_3d, metadata, gif_path)
        visualize_orthogonal_views(vol_3d, metadata, ortho_path)
        
        print(f"  -> Saved visualizations to {gif_path} and {ortho_path}")

if __name__ == '__main__':
    main()