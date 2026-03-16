#!/bin/bash
# 启动 vLLM 服务器
# 模型: Qwen2.5-7B-Instruct (本地)
# Usage: bash start_vllm_qwen.sh [max_model_len] [port]

MAX_MODEL_LEN=${1:-32768}
PORT=${2:-8006}

export CUDA_VISIBLE_DEVICES=3,6

echo "Starting vLLM server: Qwen2.5-7B-Instruct (max_model_len=${MAX_MODEL_LEN}, port=${PORT}, GPU 3,6)"
python -m vllm.entrypoints.openai.api_server \
    --model /data/hzy/models/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b-instruct \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --port ${PORT} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization 0.90
