#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"

source "${SCRIPT_DIR}/../.env"
API_BASE="${OPENAI_API_BASE}"
API_KEY="${OPENAI_API_KEY}"
MODEL_NAME="qwen2.5-32b-instruct"

DATA_FILE="$(cd "${SCRIPT_DIR}/../.." && pwd)/dataset/GPQA/gpqa_diamond.csv"
OUTPUT_DIR="${SCRIPT_DIR}/logs/amadeus_32b_3rounds"

echo "============================================"
echo "  GPQA Amadeus 32B - 3 Rounds"
echo "  Model:   ${MODEL_NAME}"
echo "  API:     ${API_BASE}"
echo "  Dataset: gpqa_diamond (198 questions)"
echo "============================================"

for SEED in 0 1 2; do
    echo ""
    echo ">>> Round $((SEED+1))/3 (seed=${SEED}) starting at $(date)"
    cd "${SCRIPT_DIR}"
    ${PYTHON_BIN} -m evo_memory.run_evo_eval \
        --agent amadeus \
        --model_name "${MODEL_NAME}" \
        --api_base "${API_BASE}" \
        --api_key "${API_KEY}" \
        --data_path "${DATA_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_tag "amadeus_32b_seed${SEED}" \
        --seed ${SEED} \
        --top_k 4 \
        --buffer_size 3 \
        --self_play \
        --self_play_mode adaptive_buffer_fixed_sp \
        --self_play_questions 3 \
        --verbose 2>&1 | tee "${OUTPUT_DIR}/round_seed${SEED}.log"
    echo ">>> Round $((SEED+1))/3 done at $(date)"
done

echo ""
echo "============================================"
echo "  All 3 rounds completed!"
echo "  Results in: ${OUTPUT_DIR}"
echo "============================================"

# Summary
echo ""
echo "=== Summary ==="
for SEED in 0 1 2; do
    echo "--- Seed ${SEED} ---"
    grep "Accuracy" "${OUTPUT_DIR}/round_seed${SEED}.log" | tail -1
done
