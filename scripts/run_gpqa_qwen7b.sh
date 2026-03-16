#!/bin/bash
# ===========================================================================
# GPQA 评测 - Qwen2.5-7B-Instruct
# 环境: amadeus1
# 注意: 此脚本直接加载模型到 GPU（非 API 模式），需要空闲 GPU
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
QPQA_DIR="${ROOT_DIR}/experiments/GPQA"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
REPO_PATH="${QPQA_DIR}/repo"
GPU="${GPU:-6}"

PROMPT_TYPE="${1:-zero_shot}"
DATASET="${2:-main}"
DATA_FILE="${ROOT_DIR}/dataset/GPQA/gpqa_${DATASET}.csv"

echo "============================================"
echo "  GPQA × Qwen2.5-7B-Instruct"
echo "  Prompt: ${PROMPT_TYPE}"
echo "  Dataset: gpqa_${DATASET}"
echo "  GPU: ${GPU}"
echo "============================================"

CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_BIN} "${QPQA_DIR}/run_qwen.py" \
    --model_path "${MODEL_PATH}" \
    --data_filename "${DATA_FILE}" \
    --repo_path "${REPO_PATH}" \
    --prompt_type "${PROMPT_TYPE}" \
    --output_dir "${QPQA_DIR}/logs/results" \
    --seed 0 \
    --temperature 0.0 \
    --max_tokens 1000 \
    --max_model_len 4096 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --verbose
