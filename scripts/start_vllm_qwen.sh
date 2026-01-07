#!/bin/bash
# 启动 vLLM 服务器
# 模型: Qwen/Qwen2.5-7B-Instruct
# 端口: 8000

# 卡号
export CUDA_VISIBLE_DEVICES=3

echo "Starting vLLM server with Qwen/Qwen2.5-7B-Instruct..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen2.5-7b-instruct \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --port 8003 \
    --gpu-memory-utilization 0.8 
echo "vLLM server started on port 8000."