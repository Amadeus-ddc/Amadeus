#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=run_eval
#SBATCH -o /data/user/zhu851/tzx/amadeus/slurm_logs/%j_run_eval.out
#SBATCH -e /data/user/zhu851/tzx/amadeus/slurm_logs/%j_run_eval.err
#SBATCH -n 8
#SBATCH --gres=gpu:1

source /data/user/zhu851/miniconda3/envs/amadeus1/bin/activate

set -e

# HuggingFace offline mode (HPC nodes have no internet)
export HF_HOME="/data/user/zhu851/tzx/amadeus/data/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR="/data/user/zhu851/tzx/amadeus/experiments/MATH"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct/"
MODEL_NAME="qwen2.5-7b-instruct"
PORT=8000
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EMBEDDING_MODEL="/data/user/zhu851/tzx/amadeus/models/all-MiniLM-L6-v2"

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

eval_run() {
    echo "Running MATH-500 evaluation..."
    python "${SCRIPT_DIR}/run_math500.py" \
        --model_name "${MODEL_NAME}" \
        --api_base "http://localhost:${PORT}/v1" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --temperature 0.0 \
        "$@"
}

case "${1}" in
    serve)
        serve
        ;;
    eval)
        shift
        eval_run "$@"
        ;;
    all)
        shift
        serve &
        SERVER_PID=$!
        echo "vLLM server PID: ${SERVER_PID}"
        wait_for_server || { kill ${SERVER_PID} 2>/dev/null; exit 1; }
        eval_run "$@"
        EVAL_EXIT=$?
        echo "Stopping vLLM server..."
        kill ${SERVER_PID} 2>/dev/null
        wait ${SERVER_PID} 2>/dev/null
        exit ${EVAL_EXIT}
        ;;
    *)
        echo "Usage: $0 {serve|eval|all} [extra args]"
        echo ""
        echo "Examples:"
        echo "  $0 serve                      # Start vLLM server"
        echo "  $0 eval                       # Run with Amadeus memory (default)"
        echo "  $0 eval --no_memory           # Run baseline without memory"
        echo "  $0 eval --max_problems 10     # Quick test with 10 problems"
        echo "  $0 eval --resume              # Resume interrupted run"
        echo "  $0 all                        # Start server + run eval"
        exit 1
        ;;
esac
