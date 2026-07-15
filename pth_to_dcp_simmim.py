import argparse
import torch
import torch.nn as nn
import torch.distributed.checkpoint as DCP
import numpy as np
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from pathlib import Path


class _BackboneAdapter(nn.Module):
    """Adapter to ensure parallelize_dino.py can find the backbone blocks via the expected attribute path."""
    def __init__(self, backbone):
        super().__init__()
        self.feature_model = backbone


class SimMIM(nn.Module):
    """SimMIM wrapper utilizing DINOv3's native token-space masking."""
    def __init__(self, backbone, encoder_stride, in_chans=3, is_3d=True):
        super().__init__()
        # Wrap backbone to satisfy parallelize_dino expectations 
        self.segmentation_model = nn.ModuleList([_BackboneAdapter(backbone)])
        
        self.encoder_stride = tuple(encoder_stride)
        self.is_3d = is_3d
        self.in_chans = in_chans

        # Extract embed dim directly from the model attribute
        embed_dim = backbone.embed_dim

        # Simple Linear Decoder for SimMIM
        self.patch_dim = in_chans * int(np.prod(self.encoder_stride))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.patch_dim)
        )

    @property
    def backbone(self):
        return self.segmentation_model[0].feature_model

    def forward(self, x, mask=None):
        # By passing is_training=True, DINO returns the full dictionary
        # By passing masks=mask, DINO natively replaces masked locations with self.mask_token
        out = self.backbone(x, masks=mask, is_training=True)
        
        # Grab the dense patch tokens -> Shape: (B, L, embed_dim)
        feats = out["x_norm_patchtokens"]
        
        # Decode to pixel space -> Shape: (B, L, patch_dim)
        pred = self.decoder(feats) 

        if mask is not None:
            # Patchify the original image to serve as the ground truth
            target = self.patchify(x) # Shape: (B, L, patch_dim)
            
            # SimMIM loss: L1 loss computed ONLY on masked patches
            loss = F.l1_loss(pred, target, reduction='none').mean(dim=-1)
            loss = (loss * mask.float()).sum() / (mask.sum() + 1e-5)
            return loss, pred
            
        return pred

    def patchify(self, imgs):
        """Flattens raw images/volumes into patches for direct L1 comparison."""
        B, C = imgs.shape[:2]
        if self.is_3d:
            sD, sH, sW = self.encoder_stride
            D_p, H_p, W_p = imgs.shape[2]//sD, imgs.shape[3]//sH, imgs.shape[4]//sW
            x = imgs.view(B, C, D_p, sD, H_p, sH, W_p, sW)
            x = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous()
            x = x.view(B, C, D_p*H_p*W_p, sD*sH*sW).transpose(1, 2).contiguous()
            return x.view(B, D_p*H_p*W_p, -1)
        else:
            sH, sW = self.encoder_stride
            H_p, W_p = imgs.shape[2]//sH, imgs.shape[3]//sW
            x = imgs.view(B, C, H_p, sH, W_p, sW)
            x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
            x = x.view(B, C, H_p*W_p, sH*sW).transpose(1, 2).contiguous()
            return x.view(B, H_p*W_p, -1)


def parse_args():
    parser = argparse.ArgumentParser(description="Load DINOv3 SimMIM models and save them as DCP checkpoints.")
    parser.add_argument("--torch_hub_path", type=Path, default=Path("/lustre/blizzard/stf218/scratch/emin/torch_hub"), help="Root of torch_hub")
    parser.add_argument("--dinov3_repo_path", type=Path, default=Path("/lustre/blizzard/stf218/scratch/emin/dinov3"), help="DINOv3 repo path")
    parser.add_argument("--dcp_root", type=Path, default=Path("outputs"), help="Root path where DCP checkpoints will be saved")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Full path where the pretrained pth checkpoints are stored
    BACKBONE_PTH_ROOT = args.torch_hub_path / "checkpoints"

    # You can add a few more dinov3 checkpoints below
    BACKBONE_CKPT_DICT = {
        # simmim w/ 3d superposition rope
        "oo_3d_sp_l_lvd_3em4_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        "cm_3d_sp_l_lvd_3em4_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
    }

    # Set torch_hub dir
    torch.hub.set_dir(str(args.torch_hub_path))

    for variant_name, config in BACKBONE_CKPT_DICT.items():
        weights_path = BACKBONE_PTH_ROOT / config["ckpt"]
        kwargs = config.get("kwargs", {})
        print(f"[{variant_name}] Config: {config}")
        
        bbone = torch.hub.load(
            str(args.dinov3_repo_path), 
            config["arch"], 
            source="local", 
            weights=str(weights_path), 
            pretrained=config["pretrained"], 
            use_fa4=True,
            **kwargs
        )
                
        model = SimMIM(
            backbone=bbone,
            encoder_stride=[16, 16, 16],
            in_chans=3,
            is_3d=True
        )

        model_state_dict = model.state_dict()
        print(f"Loaded and built model {variant_name}...")
        print(f"Model: {model}")

        # Construct full path where dcp checkpoint will be saved
        dcp_path = args.dcp_root / variant_name / "checkpoint" / "step-0"
        
        # Make directories before writing
        dcp_path.mkdir(parents=True, exist_ok=True)
        
        storage_writer = DCP.filesystem.FileSystemWriter(dcp_path, thread_count=1)
        DCP.save({"model": model_state_dict}, storage_writer=storage_writer)
        print(f"Wrote DCP ckpt to {dcp_path}...")