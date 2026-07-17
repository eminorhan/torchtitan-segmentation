import os
import argparse
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

def convert_checkpoint(dcp_dir: str, output_path: str):
    print(f"Loading distributed checkpoint from: {dcp_dir}")
    
    # 1. Unshard and consolidate the distributed checkpoint into a temporary file
    temp_path = output_path + ".temp.full.pth"
    dcp_to_torch_save(dcp_dir, temp_path)
    print("Consolidated full checkpoint saved to temporary file. Extracting backbone...")
    
    # 2. Load the consolidated dictionary
    full_ckpt = torch.load(temp_path, map_location="cpu", weights_only=False)
    
    # torchtitan saves the model state under the "model" dictionary key
    model_state = full_ckpt["model"]
        
    backbone_state = {}
    prefix_to_find = "segmentation_model.0.feature_model."
    
    # 3. Filter out the decoder/wrapper and strip prefixes
    for k, v in model_state.items():
        clean_k = k.replace("_checkpoint_wrapped_module.", "")
        
        if prefix_to_find in clean_k:
            new_k = clean_k.split(prefix_to_find)[-1]
            backbone_state[new_k] = v
            
    # 4. Save and clean up
    if len(backbone_state) == 0:
        print("\nWarning: No backbone weights found! Printing first 20 available keys:")
        for k in list(model_state.keys())[:20]:
            print(f"  {k}")
    else:
        print(f"\n--- Extracted Backbone Keys ({len(backbone_state)} total) ---")
        # Sort the keys so they are grouped logically (e.g., all blocks.0 together)
        for k in sorted(backbone_state.keys()):
            # Print the key and the shape of the tensor for extra verification
            print(f"{k}: {list(backbone_state[k].shape)}")

            if k == "rope_embed.depth_scale": print(f"{k}: {backbone_state[k]}")
            
        print("-" * 50)
        torch.save(backbone_state, output_path)
        print(f"\nSaved clean backbone checkpoint to: {output_path}")
        
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a distributed SimMIM checkpoint to a single backbone .pth file")
    parser.add_argument("--dcp_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/torchtitan-mae/outputs/oo_3d_sp_l_lvd_3em4_1_lightweight/checkpoint/step-1500", help="Path to the distributed checkpoint folder")
    parser.add_argument("--output", type=str, default="oo_3d_sp_l_lvd_3em4_1_lightweight-8aa4cbdd.pth", help="Output .pth file path")
    args = parser.parse_args()
    
    convert_checkpoint(args.dcp_dir, args.output)