#!/bin/bash
#SBATCH -p acd_u
#SBATCH --job-name=run_sciworld_streaming
#SBATCH -o /data/user/zhu851/tzx/amadeus/slurm_logs/%j_run_sciworld_streaming.out
#SBATCH -e /data/user/zhu851/tzx/amadeus/slurm_logs/%j_run_sciworld_streaming.err
#SBATCH -n 8
#SBATCH --gres=gpu:1

source /data/user/zhu851/miniconda3/envs/amadeus1/bin/activate

SCRIPT_DIR="/data/user/zhu851/tzx/amadeus/experiments/ScienceWorld"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct/"
PORT=8000

# Start vLLM server in background
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Run ScienceWorld streaming evaluation
python "${SCRIPT_DIR}/run_sciworld_streaming.py" \
    --method amadeus \
    --model_name "${MODEL}" \
    --api_base "http://localhost:${PORT}/v1" \
    --temperature 0.0 \
    --max_steps 30 \
    "$@"

EVAL_EXIT=$?

# Clean up vLLM server
echo "Stopping vLLM server..."
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

exit ${EVAL_EXIT}
