
#!/bin/bash
#SBATCH --job-name=alfworld-streaming-amadeus
#SBATCH --output=/data/user/zhu851/tzx/amadeus/scripts/sbatch/alfworld/streaming_amadeus_%j.log
#SBATCH --error=/data/user/zhu851/tzx/amadeus/scripts/sbatch/alfworld/streaming_amadeus_%j.err
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# ===========================================================================
# ALFWorld Streaming + Amadeus Memory
# max_steps=50, top_k=4
# Metrics: SR, PR
# ===========================================================================
set -e

export ALFWORLD_DATA="/data/user/zhu851/tzx/amadeus/dataset/ALFWorld"

SCRIPT_DIR="/data/user/zhu851/tzx/amadeus/experiments/ALFWorld"
MODEL="/data/user/zhu851/models/Qwen2.5-7B-Instruct"
MODEL_NAME="qwen2.5-7b-instruct"
PORT=8010
EMBEDDING_MODEL="/data/user/zhu851/tzx/amadeus/models/all-MiniLM-L6-v2"
CONDA_ENV="amadeus1"

echo "=== ALFWorld Streaming + Amadeus ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
date

# Activate conda
source /data/user/zhu851/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype auto &
VLLM_PID=$!

# Wait for server
echo "Waiting for vLLM server (PID: ${VLLM_PID})..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

# Run streaming evaluation
cd "${SCRIPT_DIR}"
python run_alfworld_streaming.py \
    --method amadeus \
    --model_name "${MODEL_NAME}" \
    --api_base "http://localhost:${PORT}/v1" \
    --api_key "token-abc123" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --max_steps 50 \
    --top_k 4 \
    --no_selfplay

EXIT_CODE=$?

# Cleanup
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

echo "Done. Exit code: ${EXIT_CODE}"
date
exit ${EXIT_CODE}
