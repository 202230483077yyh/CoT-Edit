#!/bin/bash
# =============================================================================
# Code Edit Prediction Evaluation Script
# =============================================================================
# Usage:
#   1. Configure the variables below
#   2. Run: bash run_eval.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# Model name - used for output directory naming and model identification
#
# For API models (requires environment variable):
#   - "claude-4-sonnet-20250514" (needs ANTHROPIC_API_KEY)
#   - "Qwen/Qwen3-Coder-480B-A35B-Instruct" (needs MODELSCOPE_API_KEY)
#
# For local models (requires GEN_SERVER_URL):
#   - Use the same name as --served-model-name in test_local_model.sh
#   - Example: "zed", "my-local-model"
# MODEL_NAME="Qwen/Qwen3-Coder-480B-A35B-Instruct"
MODEL_NAME="zed"

# Prompt template ID (1-4):
#   1 = Alpaca format (simple)
#   2 = Chain-of-Thought format (with reasoning)
#   3 = Un-CoT format (direct prediction)
#   4 = Claude-style/Qwen-style format
PROMPT_ID=1

# -----------------------------------------------------------------------------
# Server Configuration (for local models only)
# -----------------------------------------------------------------------------
# URL of local generation server (leave empty for API models)
# Example: "http://127.0.0.1:8000/v1/completions"
GEN_SERVER_URL="http://127.0.0.1:8006/v1/completions"

# Path to tokenizer (required only when using --is_chat with local models)
# Can be a HuggingFace model ID or local path
# Example: "Qwen/Qwen2.5-Coder-7B-Instruct"
TOKENIZER_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
# Path to evaluation dataset (.jsonl or .json)
# Use relative path for portability
EVAL_DATASET="./data/test.jsonl"

# Output directory for results
OUTPUT_DIR="./data/results/"

# Maximum tokens for model inference
INFER_MAX_TOKENS=1024

# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------
echo "============================================="
echo "Code Edit Prediction Evaluation"
echo "============================================="
echo "MODEL_NAME: $MODEL_NAME"
echo "PROMPT_ID: $PROMPT_ID"
echo "GEN_SERVER_URL: $GEN_SERVER_URL"
echo "EVAL_DATASET: $EVAL_DATASET"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "INFER_MAX_TOKENS: $INFER_MAX_TOKENS"
echo "============================================="

# Build command with optional flags
CMD="python eval.py \
    --model_name \"$MODEL_NAME\" \
    --prompt_id $PROMPT_ID \
    --eval_dataset_path \"$EVAL_DATASET\" \
    --output_dir \"$OUTPUT_DIR\" \
    --infer_max_tokens $INFER_MAX_TOKENS"

# Add generation server URL if provided (for local models)
if [[ -n "$GEN_SERVER_URL" ]]; then
    CMD="$CMD --gen_server_url \"$GEN_SERVER_URL\""
fi

# Add tokenizer path if provided
if [[ -n "$TOKENIZER_PATH" ]]; then
    CMD="$CMD --tokenizer_path \"$TOKENIZER_PATH\""
fi

# Auto-detect chat mode for instruction-tuned models
if [[ "$MODEL_NAME" == *inst* ]] || [[ "$MODEL_NAME" == *Instruct* ]]; then
    CMD="$CMD --is_chat"
    echo "Detected instruction-tuned model, enabling --is_chat"
fi

echo "Running: $CMD"
echo "============================================="

eval $CMD

echo ""
echo "============================================="
echo "Evaluation completed!"
echo "============================================="
