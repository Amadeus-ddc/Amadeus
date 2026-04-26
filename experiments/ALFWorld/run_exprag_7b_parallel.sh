#!/usr/bin/env bash
# ALFWorld 7B ExpRAG+ThinkPrune — run2 & run3 并行（GPU1:8101 + GPU2:8102）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-$(command -v /data/hzy/miniconda3/envs/amadeus1/bin/python 2>/dev/null || command -v python)}"
MODEL_NAME="${MODEL_NAME:-${ROOT_DIR}/models/Qwen2.5-7B-Instruct}"
ALFWORLD_DATA="${HOME}/.cache/alfworld"
export ALFWORLD_DATA="${ALFWORLD_DATA}"

echo "=== ALFWorld 7B ExpRAG+Prune: run2 & run3 并行 ==="
date

cd "${SCRIPT_DIR}"

# Run2 on GPU1 (port 8101)
${PYTHON} run_alfworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:8101/v1" \
    --api_key "token-abc123" \
    --max_steps 50 \
    --output_dir "logs/exprag_7b_thinkprune_run2" &
PID1=$!

# Run3 on GPU2 (port 8102)
${PYTHON} run_alfworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:8102/v1" \
    --api_key "token-abc123" \
    --max_steps 50 \
    --output_dir "logs/exprag_7b_thinkprune_run3" &
PID2=$!

echo "Run2 PID=$PID1 (port 8101), Run3 PID=$PID2 (port 8102)"
wait $PID1
echo "Run2 done."
wait $PID2
echo "Run3 done."

echo ""
echo "=== ALFWorld 结果 ==="
cat "logs/exprag_7b_thinkprune_run2/summary.json" 2>/dev/null
echo ""
cat "logs/exprag_7b_thinkprune_run3/summary.json" 2>/dev/null
date

# 接着跑 ScienceWorld
echo ""
echo "=== 开始 ScienceWorld 7B ExpRAG+Prune: run2 & run3 并行 ==="
date

cd "${ROOT_DIR}/experiments/ScienceWorld"

${PYTHON} run_sciworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:8101/v1" \
    --api_key "token-abc123" \
    --max_steps 30 \
    --output_dir "logs/exprag_7b_thinkprune_run2" &
PID3=$!

${PYTHON} run_sciworld_streaming.py \
    --method exprag \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:8102/v1" \
    --api_key "token-abc123" \
    --max_steps 30 \
    --output_dir "logs/exprag_7b_thinkprune_run3" &
PID4=$!

echo "SciWorld Run2 PID=$PID3 (port 8101), Run3 PID=$PID4 (port 8102)"
wait $PID3
echo "SciWorld Run2 done."
wait $PID4
echo "SciWorld Run3 done."

echo ""
echo "=== ScienceWorld 结果 ==="
cat "logs/exprag_7b_thinkprune_run2/summary.json" 2>/dev/null
echo ""
cat "logs/exprag_7b_thinkprune_run3/summary.json" 2>/dev/null
date
echo "=== 全部完成 ==="
