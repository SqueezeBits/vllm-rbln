#!/bin/bash
# This script shows how to run the throughput benchmark with vLLM.
MODEL_ID="meta-llama/Llama-3.2-1B"

# optimum-rbln path
VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 USE_VLLM_MODEL=0 VLLM_USE_V1=0 OPTIMUM_RBLN_ATTN_IMPL="flash_attn" \
python3 benchmark_throughput.py \
    --model $MODEL_ID \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 1024 \
    --num-prompts 100 \
    --max-model-len 8192 \
    --block-size 4096 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --warmup-requests 1

# torch.compile path
VLLM_LOGGING_LEVEL=warning RBLN_PROFILER=0 RBLN_KERNEL_MODE=triton USE_VLLM_MODEL=1 VLLM_DISABLE_COMPILE_CACHE=1 VLLM_USE_V1=1 \
python3 benchmark_throughput.py \
    --model $MODEL_ID \
    --backend vllm \
    --dataset-name random --input-len 1024 --output-len 1024 \
    --num-prompts 100 \
    --max-model-len 4096 \
    --block-size 1024 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --warmup-requests 1