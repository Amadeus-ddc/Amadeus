#!/bin/bash
#SBATCH --job-name=babilong-none
#SBATCH --output=/data/user/zhu851/tzx/amadeus/scripts/sbatch/babilong/none_%j.log
#SBATCH --error=/data/user/zhu851/tzx/amadeus/scripts/sbatch/babilong/none_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# ===========================================================================
# BABILong + No Memory (baseline)
# Tasks: qa1-qa5, Lengths: 0k 1k 2k 4k 8k 16k
# Direct model inference via vLLM API
# Metrics: Accuracy per task/length
# ===========================================================================
set -e

# HuggingFace offline mode (HPC compute nodes have no internet)
export HF_HOME="/data/user/zhu851/tzx/amadeus/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# HPC dataset path (for local dataset loading)
export HPC_DATASET_PATH="/data/user/zhu851/hf_cache/datasets/RMT-team___babilong"

BABILONG_DIR="/data/user/zhu851/tzx/amadeus/experiments/babilong"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT=8014
CONDA_ENV="amadeus1"
PYTHON_BIN="/data/user/zhu851/miniconda3/envs/${CONDA_ENV}/bin/python"

echo "=== BABILong + No Memory (baseline) ==="
echo "Job ID: ${SLURM_JOB_ID}"
date

source /data/user/zhu851/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --served-model-name "${MODEL_NAME}" \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
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

cd "${BABILONG_DIR}"
${PYTHON_BIN} scripts/run_model_on_babilong.py \
    --results_folder "${BABILONG_DIR}/logs/babilong_evals" \
    --dataset_name "RMT-team/babilong-1k-samples" \
    --model_name "${MODEL_NAME}" \
    --tokenizer_path "${MODEL}" \
    --tasks qa1 qa2 qa3 qa4 qa5 \
    --lengths 0k 1k 2k 4k 8k 16k \
    --api_url "http://localhost:${PORT}/v1/completions" \
    --use_chat_template \
    --use_instruction \
    --use_examples \
    --use_post_prompt

EXIT_CODE=$?
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
