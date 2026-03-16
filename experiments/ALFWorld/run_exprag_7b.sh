#!/usr/bin/env bash
# ALFWorld ExpRAG — 7B on GPU1 (vLLM port 8101), 3 runs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/hzy/miniconda3/envs/amadeus1/bin/python"
MODEL_NAME="/data/hzy/models/Qwen2.5-7B-Instruct"
PORT=8101
ALFWORLD_DATA="${HOME}/.cache/alfworld"
export ALFWORLD_DATA

cd "${SCRIPT_DIR}"

echo "=== ALFWorld ExpRAG 7B (3 runs) ==="
date

${PYTHON} run_alfworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key "token-abc123" \
    --max_steps 50 \
    --output_dir "logs/exprag_7b_thinkprune_run1"

echo ""
echo "=== 7B Result ==="
cat "logs/exprag_7b_thinkprune_run1/summary.json" 2>/dev/null || echo "(no summary)"
date
