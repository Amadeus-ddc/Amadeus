#!/usr/bin/env bash
# ScienceWorld ExpRAG — 32B via DashScope API
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/hzy/miniconda3/envs/amadeus1/bin/python"
MODEL_NAME="qwen2.5-32b-instruct"

cd "${SCRIPT_DIR}"

echo "=== ScienceWorld ExpRAG 32B ==="
date

${PYTHON} run_sciworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --api_key "sk-0e5d4e369c1f4bfc9aecd63d979dc008" \
    --max_steps 30 \
    --output_dir "logs/exprag_32b_thinkprune_run1"

echo ""
echo "=== 32B Result ==="
cat "logs/exprag_32b_thinkprune_run1/summary.json" 2>/dev/null || echo "(no summary)"
date
