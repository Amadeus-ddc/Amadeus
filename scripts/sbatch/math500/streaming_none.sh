#!/bin/bash
#SBATCH --job-name=math500-streaming-none
#SBATCH --output=/data/user/zhu851/tzx/amadeus/scripts/sbatch/math500/streaming_none_%j.log
#SBATCH --error=/data/user/zhu851/tzx/amadeus/scripts/sbatch/math500/streaming_none_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ===========================================================================
# MATH-500 Streaming + No Memory (baseline)
# Single-turn tasks
# Metrics: Accuracy
# ===========================================================================
set -e

# HuggingFace offline mode (HPC compute nodes have no internet)
export HF_HOME="/data/user/zhu851/tzx/amadeus/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="/data/user/zhu851/tzx/amadeus/experiments/MATH"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT=8012
CONDA_ENV="amadeus1"

echo "=== MATH-500 Streaming + No Memory ==="
echo "Job ID: ${SLURM_JOB_ID}"
date

source /data/user/zhu851/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype auto &
VLLM_PID=$!

for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

cd "${SCRIPT_DIR}"
python run_math500.py \
    --method none \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key "token-abc123" \
    --temperature 0.0

EXIT_CODE=$?
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
