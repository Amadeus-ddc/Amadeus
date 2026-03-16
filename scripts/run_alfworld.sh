#!/usr/bin/env bash
# ===========================================================================
# Amadeus × ALFWorld 运行脚本
#
# 用法:
#   1) 先启动 vLLM 服务端（或使用远程 API）
#       bash scripts/start_vllm_qwen.sh
#   2) 运行 ALFWorld 实验
#       bash scripts/run_alfworld.sh
#
# 可通过环境变量覆盖默认设置:
#   GPU=0 ENV_NUM=50 MODEL_NAME=qwen2.5-7b-instruct bash scripts/run_alfworld.sh
# ===========================================================================
set -euo pipefail

# ---- Paths ----
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${BASE_DIR}/.." && pwd)"          # amadeus/
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"      # /data/hzy/Amadeus
PYTHON_BIN="${PYTHON:-python}"

RUN_SCRIPT="${ROOT_DIR}/experiments/ALFWorld/run_alfworld.py"

# ---- GPU ----
GPU="${GPU:-6}"
export CUDA_VISIBLE_DEVICES="${GPU}"

# ---- LLM API ----
# 默认使用本地 vLLM 服务 (端口 8000)，可设为远程 API
API_BASE="${API_BASE:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-local}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-7b-instruct}"

# ---- Embedding Model (可选) ----
EMBED_MODEL="${EMBED_MODEL:-${ROOT_DIR}/models/all-MiniLM-L6-v2}"
# 如果 embedding 模型不存在，则禁用
if [[ ! -d "${EMBED_MODEL}" ]]; then
    echo "[WARN] Embedding model not found at ${EMBED_MODEL}, disabling semantic KG search."
    EMBED_FLAG=""
else
    EMBED_FLAG="--embedding_model ${EMBED_MODEL}"
fi

# ---- ALFWorld 数据 ----
# ALFWORLD_DATA 需要指向包含 json_2.1.1/ 目录的 alfworld 数据路径
if [[ -z "${ALFWORLD_DATA:-}" ]]; then
    # 优先使用 dataset/ALFWorld/
    REPO_DATA="${ROOT_DIR}/dataset/ALFWorld"
    CACHE_DATA="$HOME/.cache/alfworld"
    if [[ -d "${REPO_DATA}/json_2.1.1" ]]; then
        export ALFWORLD_DATA="${REPO_DATA}"
        echo "[INFO] Using ALFWORLD_DATA: ${ALFWORLD_DATA}"
    elif [[ -d "${CACHE_DATA}" ]]; then
        export ALFWORLD_DATA="${CACHE_DATA}"
        echo "[INFO] Fallback to ALFWORLD_DATA: ${ALFWORLD_DATA}"
    else
        echo "[ERROR] ALFWORLD_DATA not set. No data at dataset/ALFWorld/ or ~/.cache/alfworld"
        exit 1
    fi
fi

# ---- Experiment Parameters ----
ENV_NUM="${ENV_NUM:-134}"          # ALFWorld eval_in_distribution 共 134 个任务
MAX_STEPS="${MAX_STEPS:-50}"       # 每个 episode 最多 50 步
TEST_TIMES="${TEST_TIMES:-1}"      # 测试轮数（多轮取平均）
SEED="${SEED:-42}"
HISTORY_LEN="${HISTORY_LEN:-5}"    # Agent 动作历史长度
EVAL_DOMAIN="${EVAL_DOMAIN:-in}"   # "in" = in-distribution, "out" = out-of-distribution

# ---- Optimizer (Self-Play) ----
USE_OPTIMIZER="${USE_OPTIMIZER:-false}"
SP_COUNT="${SP_COUNT:-3}"

# ---- Output ----
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-amadeus_alfworld_${TS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/ALFWorld/logs}"

# ---- Build command ----
CMD="${PYTHON_BIN} ${RUN_SCRIPT}"
CMD+=" --model_name ${MODEL_NAME}"
CMD+=" --api_base ${API_BASE}"
CMD+=" --api_key ${API_KEY}"
CMD+=" --env_num ${ENV_NUM}"
CMD+=" --max_steps ${MAX_STEPS}"
CMD+=" --test_times ${TEST_TIMES}"
CMD+=" --seed ${SEED}"
CMD+=" --history_length ${HISTORY_LEN}"
CMD+=" --output_dir ${OUTPUT_DIR}"
CMD+=" --run_name ${RUN_NAME}"
CMD+=" --sp_count ${SP_COUNT}"

# Eval domain
if [[ "${EVAL_DOMAIN}" == "out" ]]; then
    CMD+=" --eval_out_domain"
fi

# Optimizer
if [[ "${USE_OPTIMIZER}" == "true" ]]; then
    CMD+=" --use_optimizer"
fi

# Embedding model
if [[ -n "${EMBED_FLAG}" ]]; then
    CMD+=" ${EMBED_FLAG}"
fi

echo "================================================================"
echo "  Amadeus × ALFWorld"
echo "================================================================"
echo "  GPU:           ${GPU}"
echo "  Model:         ${MODEL_NAME}"
echo "  API Base:      ${API_BASE}"
echo "  Env Num:       ${ENV_NUM}"
echo "  Max Steps:     ${MAX_STEPS}"
echo "  Test Times:    ${TEST_TIMES}"
echo "  History Len:   ${HISTORY_LEN}"
echo "  Optimizer:     ${USE_OPTIMIZER}"
echo "  Eval Domain:   ${EVAL_DOMAIN}"
echo "  Run Name:      ${RUN_NAME}"
echo "  Output Dir:    ${OUTPUT_DIR}"
echo "================================================================"
echo ""
echo "Running: ${CMD}"
echo ""

# ---- Execute ----
LOG_FILE="${OUTPUT_DIR}/${RUN_NAME}/console.log"
mkdir -p "$(dirname "${LOG_FILE}")"

${CMD} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "================================================================"
echo "  Done! Results saved to: ${OUTPUT_DIR}/${RUN_NAME}"
echo "================================================================"
