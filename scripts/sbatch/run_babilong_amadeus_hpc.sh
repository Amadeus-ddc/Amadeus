#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=run_babilong_amadeus
#SBATCH -o /data/user/zhu851/tzx/amadeus/slurm_logs/%j_run_babilong_amadeus.out
#SBATCH -e /data/user/zhu851/tzx/amadeus/slurm_logs/%j_run_babilong_amadeus.err
#SBATCH -n 8
#SBATCH --gres=gpu:2

source /data/user/zhu851/miniconda3/envs/amadeus1/bin/activate

# HuggingFace offline mode (HPC nodes have no internet)
export HF_HOME="/data/user/zhu851/tzx/amadeus/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ============================================================================
# Run Amadeus (full self-play) on BABILong benchmark
#
# Pipeline per chunk (same as LoCoMo adaptive_buffer_fixed_sp):
#   Builder(build KG) -> Optimizer.step(mode=fixed, N questions)
#     -> Questioner(attack) -> Answerer(defend) -> Meta-Critic(judge/patch/gradient)
# After all chunks: Answerer.answer(question) directly
#
# Prerequisites:
#   - vLLM server running (e.g. on port 8004):
#     CUDA_VISIBLE_DEVICES=4 vllm serve /data/user/zhu851/models/Qwen2.5-7B-Instruct \
#       --served-model-name qwen2.5-7b-instruct --port 8004 \
#       --gpu-memory-utilization 0.9 --max_model_len 16384 --trust_remote_code
#   - Conda env: amadeus1
#   - Embedder model: models/all-MiniLM-L6-v2
#
# Tasks: qa1-qa5
# Lengths: 0k 1k 2k 4k 8k 16k
# Each sample: fresh graph + fresh agents (no cross-sample accumulation)
# ============================================================================
set -e

PYTHON="${PYTHON:-/data/user/zhu851/miniconda3/envs/amadeus1/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BABILONG_DIR="/data/user/zhu851/tzx/amadeus/experiments/babilong"
MODEL_PATH="/data/user/zhu851/models/Qwen2.5-7B-Instruct"

export OPENAI_BASE_URL="${API_BASE:-http://localhost:8004/v1}"
export OPENAI_API_KEY="${API_KEY:-EMPTY}"

API_PORT="${API_PORT:-8004}"

# --- 启动 vLLM 服务 ---
echo "Starting vLLM server on port ${API_PORT}..."
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name qwen2.5-7b-instruct \
    --port "${API_PORT}" \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --trust-remote-code &
VLLM_PID=$!

# 等待 vLLM 就绪
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready."
        break
    fi
    sleep 5
done

# 退出时自动关闭 vLLM
trap "echo 'Shutting down vLLM...'; kill ${VLLM_PID} 2>/dev/null; wait ${VLLM_PID} 2>/dev/null" EXIT

RESULTS_FOLDER="${RESULTS_FOLDER:-${BABILONG_DIR}/logs/babilong_evals_amadeus}"
DATASET_NAME="RMT-team/babilong-1k-samples"
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b-instruct}"
API_BASE="${API_BASE:-http://localhost:8004/v1}"
API_KEY="${API_KEY:-EMPTY}"

TASKS=(qa1 qa2 qa3 qa4 qa5)
LENGTHS=(0k 1k 2k 4k 8k 16k)

# Self-play settings
ABLATION_MODE="${ABLATION_MODE:-adaptive_buffer_fixed_sp}"
FIXED_SP_COUNT="${FIXED_SP_COUNT:-3}"        # questions per flush
FIXED_BUFFER_SIZE="${FIXED_BUFFER_SIZE:-3}"  # chunks per flush (fixed_buffer modes)
NO_SELFPLAY="${NO_SELFPLAY:-false}"          # set true for ablation

echo "========================================"
echo "Amadeus x BABILong (Full Self-Play)"
echo "========================================"
echo "Model:         ${MODEL_NAME}"
echo "API:           ${API_BASE}"
echo "Tasks:         ${TASKS[*]}"
echo "Lengths:       ${LENGTHS[*]}"
echo "Self-play:     ${ABLATION_MODE} (${FIXED_SP_COUNT} questions/flush)"
echo "No self-play:  ${NO_SELFPLAY}"
echo "Output:        ${RESULTS_FOLDER}"
echo "========================================"

cd "${BABILONG_DIR}"

NOSELFPLAY_FLAG=""
if [ "${NO_SELFPLAY}" = "true" ]; then
    NOSELFPLAY_FLAG="--no_selfplay"
fi

${PYTHON} scripts/run_amadeus_on_babilong.py \
    --results_folder "${RESULTS_FOLDER}" \
    --dataset_name "${DATASET_NAME}" \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --tasks "${TASKS[@]}" \
    --lengths "${LENGTHS[@]}" \
    --max_chunk_chars 1200 \
    --ablation_mode "${ABLATION_MODE}" \
    --fixed_sp_count "${FIXED_SP_COUNT}" \
    --fixed_buffer_size "${FIXED_BUFFER_SIZE}" \
    ${NOSELFPLAY_FLAG}

echo "Done. Results in ${RESULTS_FOLDER}"
echo "Log file: ${RESULTS_FOLDER}/run.log"
