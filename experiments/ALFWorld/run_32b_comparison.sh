#!/usr/bin/env bash
# ALFWorld Streaming — 32B API comparison: none vs exprag
# 使用 DashScope API (qwen2.5-32b-instruct)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(command -v /data/hzy/miniconda3/envs/amadeus1/bin/python 2>/dev/null || command -v python)}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-32b-instruct}"
ALFWORLD_DATA="${HOME}/.cache/alfworld"

export ALFWORLD_DATA="${ALFWORLD_DATA}"
# .env 会被 run_alfworld_streaming.py 自动加载 (DashScope API)

cd "${SCRIPT_DIR}"

echo "=== ALFWorld 32B Comparison: none vs exprag ==="
date

# --- Run 1: none baseline ---
echo ""
echo "=========================================="
echo "  [1/2] none baseline (32B)"
echo "=========================================="
date

${PYTHON} run_alfworld_streaming.py \
    --method none \
    --model_name "${MODEL_NAME}" \
    --max_steps 50 \
    --output_dir "logs/none_32b_run1"

echo "none baseline done."
date

# --- Run 2: exprag ---
echo ""
echo "=========================================="
echo "  [2/2] exprag (32B)"
echo "=========================================="
date

${PYTHON} run_alfworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --max_steps 50 \
    --output_dir "logs/exprag_32b_run1"

echo "exprag done."
date

# --- Summary ---
echo ""
echo "=== Results ==="
echo "--- none baseline (32B) ---"
cat "logs/none_32b_run1/summary.json" 2>/dev/null || echo "(no summary)"
echo ""
echo "--- exprag (32B) ---"
cat "logs/exprag_32b_run1/summary.json" 2>/dev/null || echo "(no summary)"
date
