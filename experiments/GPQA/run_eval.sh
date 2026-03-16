#!/bin/bash
# ===========================================================================
# GPQA Evaluation — method-agnostic
#
# Supports:
#   streaming  — Evo-Memory streaming evaluation (memory evolves)
#   baseline   — Non-streaming baseline (direct inference)
#
# Usage:
#   bash run_eval.sh streaming [--agent amadeus] [extra args]
#   bash run_eval.sh streaming --agent none        # Streaming baseline
#   bash run_eval.sh serve                         # Start vLLM server
#   bash run_eval.sh baseline                      # Non-streaming baseline
#   bash run_eval.sh all --agent amadeus           # Server + streaming eval
#
# Environment variables:
#   GPU_ID=0                  # GPU for vLLM server (default: 0)
#   API_PORT=8003             # vLLM port (default: 8003)
#   DATASET=diamond           # GPQA split (default: diamond)
#   SELF_PLAY=true            # Self-play toggle (default: true)
#   AGENT=amadeus             # Agent/method (default: amadeus)
# ===========================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="/data/hzy/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT="${API_PORT:-8003}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-/data/hzy/miniconda3/envs/amadeus1/bin/python}"
DATASET="${DATASET:-diamond}"
SELF_PLAY="${SELF_PLAY:-true}"
SELF_PLAY_MODE="${SELF_PLAY_MODE:-adaptive_buffer_fixed_sp}"
SELF_PLAY_N="${SELF_PLAY_N:-3}"
USE_COT="${USE_COT:-false}"
AGENT="${AGENT:-amadeus}"

serve() {
    echo "Starting vLLM server with model: ${MODEL} on GPU: ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --port ${PORT} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 8192 \
        --trust-remote-code \
        --dtype auto
}

wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        sleep 2
    done
    echo "ERROR: Server did not start within 4 minutes"
    return 1
}

streaming() {
    DATA_FILE="${SCRIPT_DIR}/../../dataset/GPQA/gpqa_${DATASET}.csv"
    OUTPUT_DIR="${SCRIPT_DIR}/logs/results_evo"

    echo "============================================"
    echo "  GPQA Evo-Memory STREAMING Evaluation"
    echo "  Agent:     ${AGENT}"
    echo "  Model:     ${MODEL_NAME}"
    echo "  Dataset:   gpqa_${DATASET}"
    echo "  API:       http://localhost:${PORT}/v1"
    echo "  Self-play: ${SELF_PLAY} (mode=${SELF_PLAY_MODE}, N=${SELF_PLAY_N})"
    echo "============================================"

    EXTRA_ARGS=""
    if [ "${SELF_PLAY}" == "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --self_play --self_play_mode ${SELF_PLAY_MODE} --self_play_questions ${SELF_PLAY_N}"
    else
        EXTRA_ARGS="${EXTRA_ARGS} --no_self_play"
    fi
    if [ "${USE_COT}" == "true" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --use_cot"
    fi

    cd "${SCRIPT_DIR}"
    ${PYTHON_BIN} -m evo_memory.run_evo_eval \
        --agent "${AGENT}" \
        --model_name "${MODEL_NAME}" \
        --api_base "http://localhost:${PORT}/v1" \
        --api_key EMPTY \
        --data_path "${DATA_FILE}" \
        --output_dir "${OUTPUT_DIR}" \
        --seed 0 \
        --top_k 4 \
        --verbose \
        ${EXTRA_ARGS} \
        "$@"
}

baseline() {
    echo "============================================"
    echo "  GPQA Non-streaming Baseline"
    echo "  Model: ${MODEL_NAME}"
    echo "  Dataset: gpqa_${DATASET}"
    echo "============================================"
    python "${SCRIPT_DIR}/run_qwen.py" \
        --model_path "${MODEL}" \
        --data_filename "${SCRIPT_DIR}/../../dataset/GPQA/gpqa_${DATASET}.csv" \
        --repo_path "${SCRIPT_DIR}/repo" \
        --prompt_type zero_shot \
        --output_dir "${SCRIPT_DIR}/logs/results" \
        --seed 0 \
        --temperature 0.0 \
        --verbose \
        "$@"
}

case "${1}" in
    serve)
        serve
        ;;
    streaming)
        shift
        streaming "$@"
        ;;
    baseline)
        shift
        baseline "$@"
        ;;
    all)
        shift
        serve &
        SERVER_PID=$!
        echo "vLLM server PID: ${SERVER_PID}"
        wait_for_server || { kill ${SERVER_PID} 2>/dev/null; exit 1; }
        streaming "$@"
        EVAL_EXIT=$?
        echo "Stopping vLLM server..."
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
        exit ${EVAL_EXIT}
        ;;
    *)
        echo "Usage: $0 {serve|streaming|baseline|all} [extra args]"
        echo ""
        echo "Examples:"
        echo "  $0 serve                                   # Start vLLM server"
        echo "  $0 streaming --agent amadeus               # Streaming with Amadeus"
        echo "  $0 streaming --agent none                  # Streaming baseline (no memory)"
        echo "  $0 streaming --agent amadeus --max_examples 10  # Quick test"
        echo "  $0 baseline                                # Non-streaming baseline"
        echo "  AGENT=amadeus $0 all                       # Server + streaming eval"
        echo ""
        echo "Agents:   none, amadeus"
        echo "Env vars: GPU_ID, API_PORT, DATASET, SELF_PLAY, AGENT"
        exit 1
        ;;
esac
