#!/usr/bin/env bash
# ALFWorld ExpRAG — 32B via DashScope API, 3 runs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(command -v /data/hzy/miniconda3/envs/amadeus1/bin/python 2>/dev/null || command -v python)}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-32b-instruct}"
ALFWORLD_DATA="${HOME}/.cache/alfworld"
export ALFWORLD_DATA

cd "${SCRIPT_DIR}"

echo "=== ALFWorld ExpRAG 32B (3 runs) ==="
date

${PYTHON} run_alfworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --max_steps 50 \
    --output_dir "logs/exprag_32b_thinkprune_run1"

echo ""
echo "=== 32B Result ==="
cat "logs/exprag_32b_thinkprune_run1/summary.json" 2>/dev/null || echo "(no summary)"
date
