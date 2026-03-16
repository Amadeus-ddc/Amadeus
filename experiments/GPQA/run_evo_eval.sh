#!/bin/bash
# ===========================================================================
# Evo-Memory GPQA 流式评测 - Amadeus + Qwen2.5-7B-Instruct
#
# 前置条件:
#   1. vLLM 服务已启动 (例如端口 8003/8004/8005)
#   2. conda 环境: amadeus1
#
# 用法:
#   bash run_evo_eval.sh                         # 默认: diamond, 全量, 自博弈开启
#   bash run_evo_eval.sh diamond 10              # diamond, 前10题 (调试)
#   bash run_evo_eval.sh main                    # main split, 全量
#
# 环境变量:
#   API_PORT=8003         # vLLM 端口 (默认 8003)
#   SELF_PLAY=true        # 自博弈开关 (默认 true)
#   SELF_PLAY_MODE=adaptive_buffer_fixed_sp  # 自博弈模式 (默认 adaptive_buffer_fixed_sp)
#   SELF_PLAY_N=3         # 每轮自博弈问题数 (默认 3)
#   USE_COT=false         # Optimizer CoT 模式 (默认 false)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QPQA_DIR="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

# --- 参数 ---
DATASET="${1:-diamond}"
MAX_EXAMPLES="${2:-}"
API_PORT="${API_PORT:-8003}"

# --- 自博弈配置 ---
SELF_PLAY="${SELF_PLAY:-true}"
SELF_PLAY_MODE="${SELF_PLAY_MODE:-adaptive_buffer_fixed_sp}"
SELF_PLAY_N="${SELF_PLAY_N:-3}"
USE_COT="${USE_COT:-false}"

DATA_FILE="$(cd "${SCRIPT_DIR}/../.." && pwd)/dataset/GPQA/gpqa_${DATASET}.csv"
OUTPUT_DIR="${QPQA_DIR}/logs/results_evo"

echo "============================================"
echo "  Evo-Memory GPQA Streaming Evaluation"
echo "  Agent:      Amadeus (full pipeline)"
echo "  Model:      Qwen2.5-7B-Instruct"
echo "  Dataset:    gpqa_${DATASET}"
echo "  API:        http://localhost:${API_PORT}/v1"
echo "  Self-play:  ${SELF_PLAY} (mode=${SELF_PLAY_MODE}, N=${SELF_PLAY_N}, CoT=${USE_COT})"
echo "============================================"

EXTRA_ARGS=""
if [ -n "${MAX_EXAMPLES}" ]; then
    EXTRA_ARGS="--max_examples ${MAX_EXAMPLES}"
    echo "  Max examples: ${MAX_EXAMPLES} (debug mode)"
fi

# 自博弈参数
if [ "${SELF_PLAY}" == "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --self_play --self_play_mode ${SELF_PLAY_MODE} --self_play_questions ${SELF_PLAY_N}"
else
    EXTRA_ARGS="${EXTRA_ARGS} --no_self_play"
fi

if [ "${USE_COT}" == "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_cot"
fi

echo ""

cd "${QPQA_DIR}"

${PYTHON_BIN} -m evo_memory.run_evo_eval \
    --agent amadeus \
    --model_name qwen2.5-7b-instruct \
    --api_base "http://localhost:${API_PORT}/v1" \
    --api_key EMPTY \
    --data_path "${DATA_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 0 \
    --top_k 4 \
    --verbose \
    ${EXTRA_ARGS}
