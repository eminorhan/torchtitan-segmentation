import argparse
import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from dinov3.eval.segmentation.models import build_segmentation_decoder
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Load DINOv3 models and save them as DCP checkpoints.")
    parser.add_argument("--torch_hub_path", type=Path, default=Path("/lustre/blizzard/stf218/scratch/emin/torch_hub"), help="Root of torch_hub")
    parser.add_argument("--dinov3_repo_path", type=Path, default=Path("/lustre/blizzard/stf218/scratch/emin/dinov3"), help="DINOv3 repo path")
    parser.add_argument("--dcp_root", type=Path, default=Path("outputs"), help="Root path where DCP checkpoints will be saved")
    parser.add_argument("--decoder_type", type=str, choices=["linear", "m2f"], default="linear", help="Segmentation head type (choices: linear, m2f)")
    parser.add_argument("--num_classes", type=int, default=64, help="Number of classes in the current semantic segmentation task")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    # Full path where the pretrained pth checkpoints are stored
    BACKBONE_PTH_ROOT = args.torch_hub_path / "checkpoints"

    # You can add a few more dinov3 checkpoints below
    BACKBONE_CKPT_DICT = {
        # 2d
        # "2d_l_last_scratch_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "2d_l_last_lvd_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_sat_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_scratch_3em5_32_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em5_32_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em5_32_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em5_32_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "2d_l_last_lvd_3em5_32_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em5_32_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em5_32_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em5_32_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_sat_3em5_32_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em5_32_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em5_32_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em5_32_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_scratch_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "2d_l_last_lvd_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_sat_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_scratch_3em5_256_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em5_256_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em5_256_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "2d_l_last_scratch_3em5_256_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "2d_l_last_lvd_3em5_256_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em5_256_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em5_256_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_lvd_3em5_256_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "2d_l_last_sat_3em5_256_1": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em5_256_2": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em5_256_3": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "2d_l_last_sat_3em5_256_4": {"arch": "dinov3_vitl16", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # 3d
        # "3d_l_last_scratch_3em4_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em4_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em4_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em4_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "3d_l_last_lvd_3em4_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em4_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em4_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em4_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_sat_3em4_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em4_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em4_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em4_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_scratch_3em5_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em5_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em5_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em5_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "3d_l_last_lvd_3em5_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em5_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em5_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em5_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_sat_3em5_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em5_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em5_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em5_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_scratch_3em4_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em4_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em4_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em4_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "3d_l_last_lvd_3em4_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em4_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em4_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em4_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_sat_3em4_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em4_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em4_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em4_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_scratch_3em5_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em5_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em5_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},
        # "3d_l_last_scratch_3em5_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": False},

        # "3d_l_last_lvd_3em5_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em5_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em5_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_lvd_3em5_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "3d_l_last_sat_3em5_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em5_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em5_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},
        # "3d_l_last_sat_3em5_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", "backbone_out_layers": "last", "pretrained": True},

        # 3d superposition rope
        # "3d_sp_l_last_lvd_3em4_32_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "3d_sp_l_last_lvd_3em4_32_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "3d_sp_l_last_lvd_3em4_32_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "3d_sp_l_last_lvd_3em4_32_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},

        # "3d_sp_l_last_lvd_3em4_256_1": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "3d_sp_l_last_lvd_3em4_256_2": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "3d_sp_l_last_lvd_3em4_256_3": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "3d_sp_l_last_lvd_3em4_256_4": {"arch": "dinov3_vitl16_3D", "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True, "kwargs": {"pos_embed_rope_type": "superposition"}},

        # oo 2d ssl
        # "oo_2d_l_last_lvd_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_lvd8_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd8_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd8_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd8_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_lvd25_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd25_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd25_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd25_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_scratch_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_scratch_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_scratch_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_scratch_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_lvd_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_lvd8_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd8_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd8_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd8_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd8-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_lvd25_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd25_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd25_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_lvd25_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_lvd25-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "oo_2d_l_last_scratch_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_scratch_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_scratch_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "oo_2d_l_last_scratch_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "oo_2d_l_scratch-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # cellmap 2d ssl
        # "cm_2d_l_last_lvd_3em4_32_1": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "cm_2d_l_last_lvd_3em4_32_2": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "cm_2d_l_last_lvd_3em4_32_3": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "cm_2d_l_last_lvd_3em4_32_4": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # "cm_2d_l_last_lvd_3em4_256_1": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "cm_2d_l_last_lvd_3em4_256_2": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "cm_2d_l_last_lvd_3em4_256_3": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},
        # "cm_2d_l_last_lvd_3em4_256_4": {"arch": "dinov3_vitl16", "ckpt": "cm_2d_l_lvd-8aa4cbdd.pth", "backbone_out_layers": "last", "pretrained": True},

        # model size
        "2d_7b_last_lvd_3em4_32_1": {"arch": "dinov3_vit7b16", "ckpt": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_7b_last_lvd_3em4_32_2": {"arch": "dinov3_vit7b16", "ckpt": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_7b_last_lvd_3em4_32_3": {"arch": "dinov3_vit7b16", "ckpt": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_7b_last_lvd_3em4_32_4": {"arch": "dinov3_vit7b16", "ckpt": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth", "backbone_out_layers": "last", "pretrained": True},

        "2d_hplus_last_lvd_3em4_32_1": {"arch": "dinov3_vith16plus", "ckpt": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_hplus_last_lvd_3em4_32_2": {"arch": "dinov3_vith16plus", "ckpt": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_hplus_last_lvd_3em4_32_3": {"arch": "dinov3_vith16plus", "ckpt": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_hplus_last_lvd_3em4_32_4": {"arch": "dinov3_vith16plus", "ckpt": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", "backbone_out_layers": "last", "pretrained": True},

        "2d_b_last_lvd_3em4_32_1": {"arch": "dinov3_vitb16", "ckpt": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_b_last_lvd_3em4_32_2": {"arch": "dinov3_vitb16", "ckpt": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_b_last_lvd_3em4_32_3": {"arch": "dinov3_vitb16", "ckpt": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "backbone_out_layers": "last", "pretrained": True},
        "2d_b_last_lvd_3em4_32_4": {"arch": "dinov3_vitb16", "ckpt": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", "backbone_out_layers": "last", "pretrained": True},

        # "dinov3_vit7b16_3D_linear":         {"arch": "dinov3_vit7b16_3D",    "ckpt": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"},
        # "dinov3_vit7b16_2D_linear":         {"arch": "dinov3_vit7b16",       "ckpt": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"},
        # "dinov3_vith16plus_3D_linear":      {"arch": "dinov3_vith16plus_3D", "ckpt": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"},
        # "dinov3_vith16plus_2D_linear":      {"arch": "dinov3_vith16plus",    "ckpt": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"},
        # "dinov3_vitl16_3D_linear":          {"arch": "dinov3_vitl16_3D",     "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"},
        # "dinov3_vitl16_3D_linear_sp":       {"arch": "dinov3_vitl16_3D",     "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "kwargs": {"pos_embed_rope_type": "superposition"}},
        # "dinov3_vitl16_2D_linear":          {"arch": "dinov3_vitl16",        "ckpt": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"},
        # "dinov3_vitb16_3D_linear":          {"arch": "dinov3_vitb16_3D",     "ckpt": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"},
        # sat-493m
        # "dinov3_vitl16_3D_linear_sat493m":  {"arch": "dinov3_vitl16_3D",     "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"},
        # "dinov3_vitl16_2D_linear_sat493m":  {"arch": "dinov3_vitl16",        "ckpt": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"},
        # "dinov3_vit7b16_3D_linear_sat493m": {"arch": "dinov3_vit7b16_3D",    "ckpt": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"},
        # "dinov3_vit7b16_2D_linear_sat493m": {"arch": "dinov3_vit7b16",       "ckpt": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"},
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
        
        model = build_segmentation_decoder(
            bbone,
            backbone_out_layers=config["backbone_out_layers"],
            decoder_type=args.decoder_type, 
            num_classes=args.num_classes
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