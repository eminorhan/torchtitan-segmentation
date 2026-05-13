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
#SBATCH --array=0,13-14,37,39-40,42,46,53-57,999

# 0: 103,106,153,156,170,181,206,382-384,386-387,390,393-395,397-398,400-402,404-406,408-409,412-413,415,419-420,422,429-431,433,437-438,440-441,444,448-449,451,453,455-456,458-460,462-463,469-471,473-474,478,482,484-485,487-489,491-492,495-496,498,500,503,506-507,510-511,513,516-518,520-521,524-525,527-529,531-532,535,538-540,542-543,546-547,550,553-554,556-557,560-561,563-565,568-569,571-572,574-576,582-583,585-587,641
# 1: 277-279,284,288,291,303,343,371,373,380,385,399,410,422,424,442,445,450,453,456,458-459,462,464,467,473,475-476,478,481,489-490,495,498,501,504,507,509-510,512,515,518,520-521,524,527,529,532,537,541,543,546,549,552,554-555,558,560,563,568,572,574-575,580,585,589,594,597,600,602-603,605,719,722,742,751,762,765,768,771,774,975-978,980,985,989,992-994
# 2: 0,13-14,37,39-40,42,46,53-57,999

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
PART_INDEX=$((SLURM_ARRAY_TASK_ID + 2000))

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
