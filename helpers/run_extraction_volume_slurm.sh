#!/bin/bash

#SBATCH --account=stf218-arch
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --job-name=extract_volume
#SBATCH --output=extract_volume_%A_%a.out
#SBATCH --array=0-999

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
export HF_HUB_OFFLINE=1
export GPUS_PER_NODE=4

# --- Configuration ---
TOTAL_PARTS=3000
PYTHON_SCRIPT="create_volume_dataset_oo.py"
PART_INDEX=$((SLURM_ARRAY_TASK_ID + 1000))

echo "========================================"
echo "Processing Part $PART_INDEX of $((TOTAL_PARTS-1))..."
echo "========================================"

# Execute the python script with the current index
python $PYTHON_SCRIPT --total_parts $TOTAL_PARTS --part_index $PART_INDEX

# Catch errors
if [ $? -ne 0 ]; then
    echo "Error: Extraction failed at part $PART_INDEX."
    exit 1
fi

echo "Part $PART_INDEX finished successfully."
