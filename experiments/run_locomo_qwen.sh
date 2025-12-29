#!/bin/bash
# 使用本地 vLLM 运行实验

# 确保 vLLM 服务已经启动 (运行 ./start_vllm_qwen.sh)

python amadeus/experiments/run_locomo.py \
    --model_name qwen2.5-14b-instruct \
    --api_base http://localhost:8000/v1 \
    --api_key EMPTY \
    --sample_id conv-26-qwen
