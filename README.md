### Large-scale distributed training of 2D/3D segmentation models on volume EM data
This repository can be used to train large-scale 2D/3D segmentation models on volume EM data. It currently supports masked autoencoder (MAE) type pretraining on unlabeled data, as well as supervised training on labeled data (segmentation masks).

The skeleton of the training code here is based on an earlier version of the [`torchtitan`](https://github.com/pytorch/torchtitan) library, although the model definitions, data loading, and parallelization components are substantially rewritten. The code currently supports pure **DDP** (distributed data parallelism), **FSDP** (fully sharded data parallelism) and **TP** (tensor parallelism). TP is unlikely to be needed unless you're training very large models and/or models with very large context sizes.

This repository is currently being developed and tested on Arch, an HPE Cray EX254n supercomputer hosted at OLCF with 168 NVIDIA GH200 superchips (42 nodes x 4 GH200s; each GH200 has 96GB HBM3 high-bandwidth GPU memory).

### Requirements

* Create a python virtual environment and activate it:
```bash
python -m venv myvenv
source myvenv/bin/activate
``` 

* Clone this repository and `cd` into it:
```bash
git clone https://github.com/eminorhan/torchtitan-segmentation.git
cd torchtitan-segmentation
```

* Install the required dependencies:
```bash
pip install -r requirements.txt
```

* **[FlashAttention-4]** If you're running this repository on Hopper or Blackwell GPUs, we strongly recommend installing [FlashAttention-4](https://github.com/dao-ailab/flash-attention?tab=readme-ov-file#flashattention-4-cutedsl) for a highly efficient, state-of-the-art self-attention implementation (with the `cu13` extra for optimal performance):
```bash
pip install "flash-attn-4[cu13]"
```

* **[aws-ofi-nccl]** (On Arch only) For a more performant interconnect, install the [`aws-ofi-nccl`](https://github.com/aws/aws-ofi-nccl) plugin, which will enable `nccl` to use `libfabric`. I provide an example bash shell script [here](build_aws_ofi_nccl.sh), demonstrating how to install the `aws-ofi-nccl` plugin (note that this is Arch specific; you would need to modify the script depending on your set-up).

### Data
Download the full CellMap challenge data as described [here](https://github.com/janelia-cellmap/cellmap-segmentation-challenge?tab=readme-ov-file#download-the-data), *e.g.* (after installing the `cellmap-segmentation-challenge` repository):
```bash
csc fetch-data --raw-padding 128 --fetch-all-em-resolutions --batch-size 1024 --num-workers 64
```

### Code components
The following is a brief description of the main components of the code base so users can navigate and modify the code base more easily according to their needs:

* [`torchtitan/parallelisms`](torchtitan/parallelisms/): implements the main parallelization techniques (DDP, FSDP, TP), as well as activation checkpointing (AC), JIT compilation (`torch.compile`), and mixed precision training for both [MAE](torchtitan/parallelisms/parallelize_mae.py) and [DINOv3](torchtitan/parallelisms/parallelize_dino.py) models used for segmentation.
* [`torchtitan/checkpoint.py`](torchtitan/checkpoint.py): implements the distributed checkpoint saving and loading logic.
* [`torchtitan/config_manager.py`](torchtitan/config_manager.py): implements all config options and defaults.
* [`torchtitan/datasets.py`](torchtitan/datasets.py): implements the 2D/3D dataset and dataloader classes.
* [`torchtitan/evaluation.py`](torchtitan/evaluation.py): implements the evaluation metrics for the 2D/3D models.
* [`torchtitan/model.py`](torchtitan/model.py): implements the MAE encoder/decoder models. The segmentation models are borrowed from DINOv3 and are implemented in a [separate repository](https://github.com/eminorhan/dinov3).
* [`torchtitan/train_configs`](torchtitan/train_configs): contains config files for various training experiments (these override the defaults in [`torchtitan/config_manager.py`](torchtitan/config_manager.py)).

### Training
We recommend using `torchrun` to launch distributed training jobs.

#### MAE pretraining 
Use the [`train_mae.py`](train_mae.py) script to launch an MAE pretraining job, *e.g.*:
```python
torchrun \
    --nnodes NNODES \
    --nproc_per_node GPUS_PER_NODE \
    --max_restarts 1 \
    --node_rank NODEID \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint "MASTER_ADDR:MASTER_PORT" \
    ./train_mae.py \
    --job.config_file CONFIG_FILE
```
where `CONFIG_FILE` specifies the config file to be used for the training job. A complete example SLURM batch file can be found in [`train_mae.sh`](train_mae.sh). This uses the example config file in [`train_configs/demo_mae.toml`](train_configs/demo_mae.toml), which implements a very generic 16-layer 3D ViT encoder with **~2B** parameters and a generic 4-layer ViT decoder.

#### Segmentation training
Use the [`train_segmentation.py`](train_segmentation.py) script to launch a supervised segmentation training job, *e.g.*:
```python
torchrun \
    --nnodes NNODES \
    --nproc_per_node GPUS_PER_NODE \
    --max_restarts 1 \
    --node_rank NODEID \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint "MASTER_ADDR:MASTER_PORT" \
    ./train_segmentation.py \
    --job.config_file CONFIG_FILE
```
where `CONFIG_FILE` specifies the config file to be used for the training job. Example config files for training 2D and 3D segmentation models can be found in [`train_configs/demo_segmentation_2d.toml`](train_configs/demo_segmentation_2d.toml) and [`train_configs/demo_segmentation_3d.toml`](train_configs/demo_segmentation_3d.toml), respectively. A complete example SLURM batch file is provided in [`train_segmentation.sh`](train_segmentation.sh). 

Currently, only backbones with the DINOv3 encoder architecture are supported in the segmentation models (pretrained or randomly initialized). The default segmentation head uses a linear segmentation head bolted on top of the concatenation of four uniformly spaced feature maps (layers) from the encoder backbone. The provided demo segmentation configs will train 2D or 3D segmentation models from scratch. 

To utilize the pretrained DINOv3 checkpoints, you will need first to convert the `.pth` checkpoints (provided by Meta) to distributed `dcp` checkpoints. You can use the [`pth_to_dcp.py`](pth_to_dcp.py) script to achieve this conversion, *e.g.*:
```python
python -u pth_to_dcp.py \
    --torch_hub_path TORCH_HUB_PATH \
    --dinov3_repo_path DINOV3_REPO_PATH \
    --dcp_root DCP_ROOT
```
where `TORCH_HUB_PATH` is the root `torch_hub` path (where the `pth` checkpoints are saved locally), `DINOV3_REPO_PATH` is the path to the local dinov3 repository, and `DCP_ROOT` is the root `dcp` path where the converted `dcp` checkpoint will be saved (`outputs` by default). Finally, in the training config file, make sure that `config.job.dump_folder` matches `DCP_ROOT / f"{backbone}_{decoder_type}"` (*e.g.* `outptus/dinov3_vit7b16_linear`) to be able start training from the corresponding pretrained backbone.

During training: 

* Distributed (`dcp`) checkpoints will be saved under `config.job.dump_folder/checkpoint`
* Training and evaluation metrics will be saved under `config.job.dump_folder/logs`
* Visualization of predicted vs. ground truth masks for each 3D validation crop will be saved under `config.job.dump_folder/visuals` as `.gif` animations like the following example (this example was taken from a pretrained model with a `dinov3_vitl16` backbone that was then further trained on the CellMap data using the self-supervised DINOv3 algorithm):

![](assets/dinov3_vitl16_val_sample_rank8_sample0.gif)

The frequency with which these artifacts are produced during training can be controlled from the training config files.

### Helpers

The [`helpers`](helpers) directory provides a few useful utilities:
* [`helpers/create_slice_dataset.py`](helpers/create_slice_dataset.py): creates a Hugging Face dataset class of all 2D slices from a `zarr` volume dataset and pushes it to the Hugging Face hub (as in [this example](https://huggingface.co/datasets/eminorhan/cellmap-2d) for the CellMap challenge data).
* [`helpers/plot_volumes_grid.py`](helpers/plot_volumes_grid.py): visualizes a grid of EM volumes with their segmentation masks as a `.gif` animation. 
* [`helpers/plot_volumes_single.py`](helpers/plot_volumes_single.py): visualizes individual EM volumes with their segmentation masks and labels as a `.gif` animation, like the following example:

<div align="center">
  <img src="assets/sample_27_viz.gif" width="480">
</div>