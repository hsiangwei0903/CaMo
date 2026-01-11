#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --qos=normal
#SBATCH --gres=gpu:8             
#SBATCH --cpus-per-task=64    
#SBATCH --mem=1600G             
#SBATCH --time=04:00:00         
#SBATCH --output=log/preprocess_%j.log
#SBATCH --error=log/preprocess_%j.err

echo "=== SLURM job started ==="
echo "Node: $(hostname)"
echo "GPUs allocated: $SLURM_GPUS"

# ---------------------------
# Load CUDA & GCC
# ---------------------------
module purge
module load gcc/11.5.0
module load cuda/12.4
module load conda

# ---------------------------
# Initialize Conda
# ---------------------------
conda activate camo

# ---------------------------
# Cache directories
# ---------------------------
export HF_HOME=/gpfs/projects/embodied3d/hf-cache
export HF_HUB_CACHE=/gpfs/projects/embodied3d/hf-cache
export TRANSFORMERS_CACHE=/gpfs/projects/embodied3d/hf-cache
export HF_DATASETS_CACHE=/gpfs/projects/embodied3d/hf-cache
export XDG_CACHE_HOME=/gpfs/projects/embodied3d/hf-cache

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Using python: $(which python)"
echo "Using pip: $(which pip)"
echo "CUDA: $(which nvcc)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available in Torch: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Launch training with config YAML
llamafactory-cli train examples/train_full/qwen2_5vl_3b_full_sft_camo.yaml

echo "=== Job finished ==="