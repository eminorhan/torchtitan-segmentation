#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --job-name=train_segmentation_model_size
#SBATCH --output=train_segmentation_model_size_%A_%a.out
#SBATCH --array=0-11

# activate venv
source /lustre/blizzard/stf218/scratch/emin/blizzardvenv/bin/activate

# set misc env vars
export LD_LIBRARY_PATH=/lustre/blizzard/stf218/scratch/emin/aws-ofi-nccl-1.19.0/lib:$LD_LIBRARY_PATH  # enable aws-ofi-nccl
export NCCL_NET=ofi
export FI_PROVIDER=cxi
export LOGLEVEL=INFO
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export GLOO_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export HF_HOME="/lustre/blizzard/stf218/scratch/emin/huggingface"
export HF_DATASETS_CACHE="/lustre/blizzard/stf218/scratch/emin/huggingface"
export TRITON_CACHE_DIR="/lustre/blizzard/stf218/scratch/emin/triton"
export PYTORCH_KERNEL_CACHE_PATH="/lustre/blizzard/stf218/scratch/emin/pytorch_kernel_cache"
export MPLCONFIGDIR="/lustre/blizzard/stf218/scratch/emin/mplconfigdir"
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=4

# set network
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=3442

# --- CONFIG RESOLUTION ---
# 1. Create a 0-indexed bash array of all toml files in the directory
CONFIG_FILES=(./configs/model_size/*.toml)

# 2. Select the specific file for this array task
CONFIG_FILE=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}

# 3. Print the filename
echo "================================================================="
echo "Starting job array task ID: $SLURM_ARRAY_TASK_ID"
echo "Using config file: $CONFIG_FILE"
echo "================================================================="

# 4. Print the full contents of the config file
echo "Config contents:"
cat "$CONFIG_FILE"
echo "================================================================="

srun torchrun --nnodes $SLURM_NNODES --nproc_per_node $GPUS_PER_NODE --max_restarts 1 --node_rank $SLURM_NODEID --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" ./train_segmentation.py --job.config_file ${CONFIG_FILE}

echo "Done"