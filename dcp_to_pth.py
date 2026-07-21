import os
import argparse
import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

def convert_checkpoint(dcp_dir: str, output_path: str):
    print(f"Loading distributed checkpoint from: {dcp_dir}")
    
    # 1. Unshard and consolidate the distributed checkpoint into a temporary file
    temp_path = output_path + ".temp.full.pth"
    dcp_to_torch_save(dcp_dir, temp_path)
    print("Consolidated full checkpoint saved to temporary file. Extracting full model...")
    
    # 2. Load the consolidated dictionary
    full_ckpt = torch.load(temp_path, map_location="cpu", weights_only=False)
    
    # torchtitan saves the model state under the "model" dictionary key
    model_state = full_ckpt["model"]
        
    full_model_state = {}
    
    # Target prefix to ensure we are grabbing the model weights (both backbone and head)
    target_prefix = "segmentation_model."
    
    # 3. Filter out the wrapper and keep the correct keys
    for k, v in model_state.items():
        # Remove the FSDP artifact
        clean_k = k.replace("_checkpoint_wrapped_module.", "")
        
        if target_prefix in clean_k:
            full_model_state[clean_k] = v
            
    # 4. Save and clean up
    if len(full_model_state) == 0:
        print("\nWarning: No model weights found! Printing first 20 available keys:")
        for k in list(model_state.keys())[:20]:
            print(f"  {k}")
    else:
        print(f"\n--- Extracted Model Keys ({len(full_model_state)} total) ---")
        # Sort the keys so they are grouped logically
        for k in sorted(full_model_state.keys()):
            # Print the key and the shape of the tensor for extra verification
            print(f"{k}: {list(full_model_state[k].shape)}")

            # Accommodates the full prefix for the depth_scale debug print
            if "rope_embed.depth_scale" in k:
                print(f"{k}: {full_model_state[k]}")
            
        print("-" * 50)
        torch.save(full_model_state, output_path)
        print(f"\nSaved clean full model checkpoint to: {output_path}")
        
    if os.path.exists(temp_path):
        os.remove(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a distributed checkpoint to a unified .pth file")
    parser.add_argument("--dcp_dir", type=str, default="/lustre/blizzard/stf218/scratch/emin/torchtitan-mae/outputs/3d_sp_l_last_lvd_3em4_32_1/checkpoint/step-4000", help="Path to the distributed checkpoint folder")
    parser.add_argument("--output", type=str, default="3d_sp_l_last_lvd_3em4_32_1_it4k-8aa4cbdd.pth", help="Output .pth file path")
    args = parser.parse_args()
    
    convert_checkpoint(args.dcp_dir, args.output)