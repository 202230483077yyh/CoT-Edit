#!/bin/bash
# =============================================================================
# Local Model Deployment Script (vLLM)
# =============================================================================
# This script starts a local model server using vLLM.
# After the server is running, use run_eval.sh to evaluate the model.
#
# Usage:
#   1. Configure the variables below
#   2. Run: bash test_local_model.sh
#   3. Wait for server to start (look for "Uvicorn running on http://...")
#   4. In another terminal, run: bash run_eval.sh
#
# Requirements:
#   - vLLM installed: pip install vllm
#   - CUDA-compatible GPU
# =============================================================================

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# Model name - this will be used in the API requests
MODEL_NAME="zed"

# Path to model weights (local path or HuggingFace model ID)
# Examples:
#   - Local: "/path/to/your/model"
#   - HuggingFace: "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_PATH="/storage01/chenwuya/code_generation/projects/zeta/models/zeta/"

# -----------------------------------------------------------------------------
# Server Configuration
# -----------------------------------------------------------------------------
# Port for the API server
PORT=8006

# Host address (0.0.0.0 allows external access)
HOST="0.0.0.0"

# -----------------------------------------------------------------------------
# GPU Configuration
# -----------------------------------------------------------------------------
# GPU device IDs to use (comma-separated for multiple GPUs)
# Examples: "0" for single GPU, "0,1" for two GPUs
GPU_DEVICES="0"

# Number of GPUs for tensor parallelism
TENSOR_PARALLEL_SIZE=1

# GPU memory utilization (0.0-1.0)
GPU_MEMORY_UTILIZATION=0.93

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# Maximum sequence length
MAX_MODEL_LEN=6000

# Enable optimizations
ENABLE_PREFIX_CACHING=true
ENABLE_CHUNKED_PREFILL=true

# Speculative decoding configuration (optional, can improve speed)
# Set to empty string to disable: SPECULATIVE_CONFIG=""
SPECULATIVE_CONFIG='{"method": "ngram", "prompt_lookup_min": 2, "prompt_lookup_max": 4, "num_speculative_tokens": 8}'

# =============================================================================
# Validation
# =============================================================================
if [[ -z "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH is not set!"
    echo "Please edit this script and set MODEL_PATH to your model location."
    echo ""
    echo "Examples:"
    echo '  MODEL_PATH="/path/to/your/local/model"'
    echo '  MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"'
    exit 1
fi

# =============================================================================
# Start Server
# =============================================================================
echo "============================================="
echo "Starting vLLM Server"
echo "============================================="
echo "Model Name: ${MODEL_NAME}"
echo "Model Path: ${MODEL_PATH}"
echo "Server URL: http://${HOST}:${PORT}"
echo "API Endpoint: http://${HOST}:${PORT}/v1/completions"
echo "GPU Devices: ${GPU_DEVICES}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "============================================="
echo ""
echo "Once the server is ready, run evaluation with:"
echo "  bash run_eval.sh"
echo ""
echo "Or test the server manually with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "============================================="
echo ""

export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}

# Build command
CMD="vllm serve ${MODEL_PATH} \
    --port ${PORT} \
    --host ${HOST} \
    --served-model-name ${MODEL_NAME} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"

# Add optional flags
if [[ "$ENABLE_PREFIX_CACHING" == "true" ]]; then
    CMD="$CMD --enable-prefix-caching"
fi

if [[ "$ENABLE_CHUNKED_PREFILL" == "true" ]]; then
    CMD="$CMD --enable-chunked-prefill"
fi

if [[ -n "$SPECULATIVE_CONFIG" ]]; then
    CMD="$CMD --speculative-config '${SPECULATIVE_CONFIG}'"
fi

echo "Running: $CMD"
echo ""

eval $CMD
