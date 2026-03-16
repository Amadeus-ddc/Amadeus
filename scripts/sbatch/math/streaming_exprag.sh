#!/bin/bash
#SBATCH --job-name=math-evo-exprag
#SBATCH --output=/data/hzy/Amadeus/amadeus/experiments/MATH/logs/results_evo/sbatch_exprag_%j.log
#SBATCH --error=/data/hzy/Amadeus/amadeus/experiments/MATH/logs/results_evo/sbatch_exprag_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ===========================================================================
# MATH-500 Evo-Memory: ExpRAG (32B via DashScope API)
# ===========================================================================
set -e

MATH_DIR="/data/hzy/Amadeus/amadeus/experiments/MATH"
ROOT_DIR="/data/hzy/Amadeus/amadeus"

# Offline mode
export HF_HOME="${ROOT_DIR}/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Load API credentials
source "${MATH_DIR}/../.env"
API_BASE="${OPENAI_API_BASE}"
API_KEY="${OPENAI_API_KEY}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-32b-instruct}"
TOP_K="${TOP_K:-4}"

DATA_PATH="${ROOT_DIR}/dataset/MATH/HuggingFaceH4___math-500"
OUTPUT_DIR="${MATH_DIR}/logs/results_evo"
EMBEDDER_PATH="${ROOT_DIR}/models/all-MiniLM-L6-v2"

echo "=== MATH-500 Evo-Memory: ExpRAG ==="
echo "Job ID: ${SLURM_JOB_ID}"
date

cd "${MATH_DIR}"
python -m evo_memory.run_evo_eval \
    --agent exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --embedder_path "${EMBEDDER_PATH}" \
    --top_k "${TOP_K}" \
    --run_tag "exprag_${MODEL_NAME}_topk${TOP_K}" \
    --seed 0 \
    --verbose

echo "Done. Exit code: $?"
date
