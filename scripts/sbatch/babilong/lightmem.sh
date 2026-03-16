#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=lightmem_babilong
#SBATCH -o /data/user/zhu851/tzx/amadeus/slurm_logs/%j_lightmem_babilong.out
#SBATCH -e /data/user/zhu851/tzx/amadeus/slurm_logs/%j_lightmem_babilong.err
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# Setup environment
set -e

# Activate conda environment
source /data/user/zhu851/miniconda3/bin/activate
conda activate amadeus1

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH="/data/user/zhu851/models/Qwen2.5-7B-Instruct"
export EMBEDDING_MODEL_PATH="/data/user/zhu851/models/all-MiniLM-L6-v2"
export QDRANT_DIR="/data/user/zhu851/tzx/amadeus/adapters/lightmem_babilong/qdrant_data"
export HPC_DATASET_PATH="/data/user/zhu851/hf_cache/datasets/RMT-team___babilong"

# Configuration
SPLITS="${SPLITS:-1k,4k,16k}"
TASKS="${TASKS:-qa1,qa2,qa3,qa4,qa5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user/zhu851/tzx/amadeus/results/lightmem_babilong}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p /data/user/zhu851/tzx/amadeus/slurm_logs

# Print environment info
echo "=========================================="
echo "LightMemory BABILong Experiment on HPC"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Splits: $SPLITS"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "Dataset Path: $HPC_DATASET_PATH"
echo "=========================================="

# Change to adapter directory
cd /data/user/zhu851/tzx/amadeus/adapters/lightmem_babilong

# Run experiment
echo "Starting LightMemory BABILong experiment..."
python run_experiment.py \
    --config config.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --dataset babilong \
    --splits ${SPLITS} \
    --tasks ${TASKS}

echo "Experiment completed!"
echo "Results saved to: ${OUTPUT_DIR}"
