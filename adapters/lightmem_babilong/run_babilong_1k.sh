#!/bin/bash
#SBATCH --job-name=lightmem_babilong_1k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lightmem_babilong_1k_%j.log
#SBATCH --error=logs/lightmem_babilong_1k_%j.err

# Setup environment
set -e
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source /data/hzy/miniconda3/etc/profile.d/conda.sh
conda activate amadeus1

# Set environment variables
export MODEL_PATH="/data/hzy/models/Qwen2.5-7B-Instruct"
export EMBEDDING_MODEL_PATH="/data/hzy/Amadeus/amadeus/models/all-MiniLM-L6-v2"
export QDRANT_DIR="/data/hzy/Amadeus/amadeus/adapters/lightmem_babilong/qdrant_data"
export LIGHTMEM_PATH="/data/hzy/Amadeus/lightmem/LightMem"
export BABILONG_PATH="/data/hzy/Amadeus/amadeus/experiments/babilong"

# Create logs directory
mkdir -p logs

# Print environment info
echo "=========================================="
echo "Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=========================================="

# Change to adapter directory
cd /data/hzy/Amadeus/amadeus/adapters/lightmem_babilong

# Run experiment on 1k split
echo "Starting LightMemory BABILong experiment on 1k split..."
python run_experiment.py \
    --config config.yaml \
    --output-dir ./results \
    --dataset babilong \
    --splits 1k \
    --tasks qa1,qa2,qa3,qa4,qa5

echo "Experiment completed!"
