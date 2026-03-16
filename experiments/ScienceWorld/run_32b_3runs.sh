#!/usr/bin/env bash
# ScienceWorld Streaming — 32B API: none x1 + exprag x3
# 使用 DashScope API (qwen2.5-32b-instruct)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/hzy/miniconda3/envs/amadeus1/bin/python"
MODEL_NAME="qwen2.5-32b-instruct"
API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY="sk-62e9728bb07249ec9b5696da52fdc5d2"

export HF_HOME="/data/hzy/Amadeus/amadeus/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${SCRIPT_DIR}"

echo "=== ScienceWorld 32B: none x1 + exprag x3 ==="
date

# --- none baseline (resume from partial run) ---
echo ""
echo "=========================================="
echo "  [1/4] none baseline (32B) — resume"
echo "=========================================="
date

${PYTHON} run_sciworld_streaming.py \
    --method none \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --max_steps 30 \
    --output_dir "logs/none_32b_run1" \
    --resume

echo "none baseline done."
date

# --- exprag run 1 ---
echo ""
echo "=========================================="
echo "  [2/4] exprag run1 (32B)"
echo "=========================================="
date

${PYTHON} run_sciworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --max_steps 30 \
    --output_dir "logs/exprag_32b_run1"

echo "exprag run1 done."
date

# --- exprag run 2 ---
echo ""
echo "=========================================="
echo "  [3/4] exprag run2 (32B)"
echo "=========================================="
date

${PYTHON} run_sciworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --max_steps 30 \
    --output_dir "logs/exprag_32b_run2"

echo "exprag run2 done."
date

# --- exprag run 3 ---
echo ""
echo "=========================================="
echo "  [4/4] exprag run3 (32B)"
echo "=========================================="
date

${PYTHON} run_sciworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --max_steps 30 \
    --output_dir "logs/exprag_32b_run3"

echo "exprag run3 done."
date

# --- Summary ---
echo ""
echo "=========================================="
echo "  FINAL RESULTS"
echo "=========================================="
echo "--- none baseline (32B) ---"
cat "logs/none_32b_run1/summary.json" 2>/dev/null || echo "(no summary)"
echo ""
echo "--- exprag run1 (32B) ---"
cat "logs/exprag_32b_run1/summary.json" 2>/dev/null || echo "(no summary)"
echo ""
echo "--- exprag run2 (32B) ---"
cat "logs/exprag_32b_run2/summary.json" 2>/dev/null || echo "(no summary)"
echo ""
echo "--- exprag run3 (32B) ---"
cat "logs/exprag_32b_run3/summary.json" 2>/dev/null || echo "(no summary)"
date
