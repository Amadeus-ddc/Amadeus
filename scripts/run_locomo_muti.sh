#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
# ROOT_DIR is the parent 'amadeus' package directory
ROOT_DIR="$(cd "${BASE_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# Correct paths relative to ROOT_DIR
RUN_SCRIPT="${ROOT_DIR}/experiments/LoCoMo/run_locomo.py"
DATA_FILE="${DATA_FILE:-${ROOT_DIR}/data/locomo10.json}"
EMBED_MODEL="${EMBED_MODEL:-${ROOT_DIR}/models/all-MiniLM-L6-v2}"

# Logs base directory
LOG_BASE_DIR="${ROOT_DIR}/experiments/LoCoMo/logs"
TS="$(date +%Y%m%d_%H%M%S)"

# Main model (7B) served via vLLM; adjust MODEL_NAME to match your served model name.
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b-instruct}"
# Local vLLM model路径（如需本地加载，取消注释下一行） 
#VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/data/hzy/models/Qwen2.5-7B-Instruct}"
# 默认改为云端仓库名称；如需本地路径，可通过环境变量覆盖或上面行注释去掉
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
# Extra args for vLLM if needed, e.g., VLLM_EXTRA_ARGS="--tensor-parallel-size 1"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
# Judge model stays remote
JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5-32b-instruct}"
JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-sk-f418eb8b1eb941a7975419cceb23bc89}"
ABLATION_MODE="${ABLATION_MODE:-adaptive_buffer_fixed_sp}"
FIXED_SP_COUNT="${FIXED_SP_COUNT:-3}"
COMMON_OUT="fixed_connection_error_link${TS}"
# Edit the SAMPLE_IDS/OUT/MAX_WORKERS/API_BASE/PORT per GPU if you want a different split.
declare -a JOBS=(
  "GPU=0 SAMPLE_IDS=conv-26,conv-30,conv-41,conv-42,conv-43 OUT=COT_ANSWERER1_1 MAX_WORKERS=3 PORT=8000 MAIN_API_BASE=http://localhost:8000/v1 MAIN_API_KEY=local"
  "GPU=4 SAMPLE_IDS=conv-44,conv-47,conv-48,conv-49,conv-50 OUT=COT_ANSWERER1_2 MAX_WORKERS=3 PORT=8004 MAIN_API_BASE=http://localhost:8004/v1 MAIN_API_KEY=local"
  #"GPU=1 SAMPLE_IDS=conv-42,conv-43,conv-44 OUT=gpu1-tem0-3 MAX_WORKERS=3 PORT=8001 MAIN_API_BASE=http://localhost:8001/v1 MAIN_API_KEY=local"
  #"GPU=2 SAMPLE_IDS=conv-47,conv-48 OUT=gpu2-tem0-3 MAX_WORKERS=2 PORT=8002 MAIN_API_BASE=http://localhost:8002/v1 MAIN_API_KEY=local"
  #"GPU=3 SAMPLE_IDS=conv-49,conv-50 OUT=gpu3-tem0-3 MAX_WORKERS=2 PORT=8003 MAIN_API_BASE=http://localhost:8003/v1 MAIN_API_KEY=local"
)

  for job in "${JOBS[@]}"; do
    eval "$job"
    
    : "${PORT:?PORT not set in job entry}"
    IFS=',' read -ra SAMPLE_ARR <<< "${SAMPLE_IDS}"
    for SAMPLE in "${SAMPLE_ARR[@]}"; do
      SAMPLE_TRIMMED="$(echo "${SAMPLE}" | xargs)"
      
      # Ensure the target run directory exists so we can place the shell log inside it
      TARGET_RUN_DIR="${LOG_BASE_DIR}/${OUT}"
      mkdir -p "${TARGET_RUN_DIR}"
      
      # Place shell execution log inside the run directory
      LOG_FILE="${TARGET_RUN_DIR}/console_${SAMPLE_TRIMMED}.log"
      
      CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
        --data_file "${DATA_FILE}" \
        --sample_id "${SAMPLE_TRIMMED}" \
        --model_name "${MODEL_NAME}" \
        --api_base "${MAIN_API_BASE}" \
        --api_key "${MAIN_API_KEY}" \
        --judge_model_name "${JUDGE_MODEL}" \
        --judge_api_base "${JUDGE_API_BASE}" \
        --judge_api_key "${JUDGE_API_KEY}" \
        --ablation_mode "${ABLATION_MODE}" \
        --fixed_sp_count "${FIXED_SP_COUNT}" \
        --embedding_model "${EMBED_MODEL}" \
        --run_name "${OUT}" \
        --max_workers "${MAX_WORKERS}" \
        > "${LOG_FILE}" 2>&1 &
      echo "Launched GPU ${GPU} -> sample ${SAMPLE_TRIMMED}, output ${OUT}, log ${LOG_FILE}"
    done
  done

echo "Waiting for all jobs to complete..."
wait

# Calculate Memory Graph Token Stats for each unique output directory
echo "Calculating Memory Graph Token Stats..."
for job in "${JOBS[@]}"; do
    eval "$job"
    "${PYTHON_BIN}" "${ROOT_DIR}/experiments/LoCoMo/analyze_memory_tokens.py" --run_dirs "${LOG_BASE_DIR}/${OUT}"
done

echo "All jobs completed. Use 'tail -f experiments/LoCoMo/logs/*/experiment.log' to monitor."
