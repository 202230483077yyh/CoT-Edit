# Code Edit Prediction Evaluation

This repository provides an evaluation framework for assessing LLM models on **code edit prediction** tasks — predicting the next edit a developer will make based on their recent edit history and current cursor position.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Prompt Templates](#prompt-templates)
- [Supported Models](#supported-models)
- [Command Line Arguments](#command-line-arguments)
- [Evaluation Metrics](#evaluation-metrics)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Local Model Deployment (vLLM)](#local-model-deployment-vllm)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Code edit prediction evaluates how well a model can anticipate developer actions based on:

- **Edit History**: Recent code modifications showing editing patterns and intent
- **Current Context**: Code excerpt with an editable region and cursor position
- **Task**: Predict the next logical edit within the editable region

### Key Features

- ✅ Support for external API models (Claude, Qwen)
- ✅ Support for local models via OpenAI-compatible endpoints
- ✅ Multiple prompt templates for different reasoning strategies
- ✅ Automatic checkpoint/resume for long-running evaluations
- ✅ Exact Match (EM) metric calculation

## Installation

### Prerequisites

- Python 3.8+
- API key for Claude (Anthropic) or Qwen (ModelScope)

### Install Dependencies

```bash
cd Evaluation
pip install -r requirements.txt
```

### Set API Keys

```bash
# For Qwen models (ModelScope)
export MODELSCOPE_API_KEY="your-modelscope-api-key"

# For Claude models (Anthropic)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Quick Start

### Option 1: External API Models (Recommended for Quick Testing)

```bash
# Set API key
export MODELSCOPE_API_KEY="your-key"  # For Qwen
# or
export ANTHROPIC_API_KEY="your-key"   # For Claude

# Edit configuration in run_eval.sh, then run:
bash run_eval.sh
```

### Option 2: Local Model with vLLM (Recommended for Custom Models)

**Step 1: Start the model server** (Terminal 1)
```bash
# Edit test_local_model.sh to set MODEL_PATH, then:
bash test_local_model.sh

# Wait for: "Uvicorn running on http://0.0.0.0:8006"
```

**Step 2: Run evaluation** (Terminal 2)
```bash
# Edit run_eval.sh:
#   MODEL_NAME="zed"  # Same as in test_local_model.sh
#   GEN_SERVER_URL="http://127.0.0.1:8006/v1/completions"

bash run_eval.sh
```

### Option 3: Direct Python Commands

```bash
# Evaluate with Qwen API
python eval.py \
    --model_name "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
    --prompt_id 4 \
    --eval_dataset_path ./data/test.jsonl \
    --output_dir ./results/

# Evaluate with Claude API
python eval.py \
    --model_name "claude-4-sonnet-20250514" \
    --prompt_id 4 \
    --eval_dataset_path ./data/test.jsonl \
    --output_dir ./results/

# Evaluate with local model
python eval.py \
    --model_name "zed" \
    --prompt_id 1 \
    --gen_server_url http://localhost:8006/v1/completions \
    --eval_dataset_path ./data/test.jsonl \
    --output_dir ./results/
```

## Data Format

The evaluation dataset should be in **JSONL format** (one JSON object per line):

```json
{
    "events": "User edited file:\n```diff\n-old line\n+new line\n```",
    "input": "code context with <|editable_region_start|> ... <|user_cursor_is_here|> ... <|editable_region_end|>",
    "output": "expected output with <|editable_region_start|> ... <|editable_region_end|>"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `events` | `str` or `list[str]` | Recent edit history (diff format recommended) |
| `input` | `str` or `list[str]` | Current code context with editable region markers |
| `output` | `str` or `list[str]` | Expected model output (ground truth) |

### Special Markers

| Marker | Description |
|--------|-------------|
| `<\|editable_region_start\|>` | Start of the region where edits are allowed |
| `<\|editable_region_end\|>` | End of the editable region |
| `<\|user_cursor_is_here\|>` | Current cursor position (input only) |

### Example

**Input (what the model sees):**
```
function add(a, b) {
<|editable_region_start|>
    <|user_cursor_is_here|>
<|editable_region_end|>
}
```

**Expected Output (ground truth):**
```
<|editable_region_start|>
    return a + b;
<|editable_region_end|>
```

## Prompt Templates

Four prompt templates are available, optimized for different use cases:

| ID | Name | Description | Recommended For |
|----|------|-------------|-----------------|
| 1 | **Alpaca** | Simple instruction-response format | Base models, quick testing |
| 2 | **CoT** | Chain-of-Thought with `<think>` tags | Models benefiting from reasoning |
| 3 | **Un-CoT** | Direct prediction without reasoning | Fast inference |
| 4 | **Claude-style** | Detailed guidelines and constraints | Claude/Qwen, best quality |

```bash
# Select prompt template with --prompt_id
python eval.py --prompt_id 4 ...  # Use Claude-style (recommended)
```

## Supported Models

### External API Models

| Model | Provider | Environment Variable |
|-------|----------|---------------------|
| `Qwen/Qwen3-Coder-480B-A35B-Instruct` | ModelScope | `MODELSCOPE_API_KEY` |
| `claude-4-sonnet-20250514` | Anthropic | `ANTHROPIC_API_KEY` |

### Local Models

Any model served via an OpenAI-compatible completion API:

```bash
python eval.py \
    --model_name "your-model" \
    --gen_server_url "http://localhost:8000/v1/completions" \
    --eval_dataset_path ./data/test.jsonl
```

For instruction-tuned models requiring chat templates:

```bash
python eval.py \
    --model_name "your-instruct-model" \
    --gen_server_url "http://localhost:8000/v1/completions" \
    --tokenizer_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --is_chat \
    --eval_dataset_path ./data/test.jsonl
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `--model_name` | ✅ | - | Model identifier |
| `--eval_dataset_path` | ✅ | - | Path to evaluation dataset |
| `--prompt_id` | | `1` | Prompt template (1-4) |
| `--output_dir` | | `./results/` | Output directory |
| `--gen_server_url` | * | - | Local server URL (*required for local models) |
| `--tokenizer_path` | | - | Tokenizer for chat template |
| `--infer_max_tokens` | | `1024` | Max tokens for generation |
| `--is_chat` | | `False` | Apply chat template |
| `--skip_inference` | | `False` | Only run evaluation |
| `--skip_evaluation` | | `False` | Only run inference |

### Examples

```bash
# Run full pipeline
python eval.py --model_name "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
               --prompt_id 4 \
               --eval_dataset_path ./data/test.jsonl

# Resume interrupted inference (automatically continues from checkpoint)
python eval.py --model_name "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
               --prompt_id 4 \
               --eval_dataset_path ./data/test.jsonl

# Re-run evaluation only (skip inference)
python eval.py --model_name "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
               --prompt_id 4 \
               --eval_dataset_path ./data/test.jsonl \
               --skip_inference
```

## Evaluation Metrics

### Exact Match (EM)

The primary metric measures whether the model's predicted editable region **exactly matches** the ground truth after:
1. Extracting content between `<|editable_region_start|>` and `<|editable_region_end|>`
2. Removing cursor markers (`<|user_cursor_is_here|>`)
3. Stripping whitespace

```
EM Accuracy = (Exact Matches) / (Total Samples)
```

## Output Files

Results are saved to `{output_dir}/{model_name}.pmt-{prompt_id}/`:

```
results/
└── Qwen_Qwen3-Coder-480B-A35B-Instruct.pmt-4/
    ├── inference.jsonl      # Model outputs for each sample
    ├── bad_data.jsonl       # Failed cases for analysis
    └── eval_summary.json    # Evaluation metrics
```

### eval_summary.json

```json
{
    "model_name": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "prompt_id": 4,
    "total_samples": 100,
    "em_correct": 85,
    "em_accuracy": 0.85
}
```

### inference.jsonl

Each line contains original data plus model outputs:

```json
{
    "events": "...",
    "input": "...",
    "output": "...",
    "raw_model_output": "full model response...",
    "post_model_output": "<|editable_region_start|>...<|editable_region_end|>"
}
```

## Project Structure

```
Evaluation/
├── README.md              # This documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── eval.py               # Main evaluation script
├── api.py                # API client (Claude/Qwen)
├── run_eval.sh           # Evaluation script
├── test_local_model.sh   # vLLM server startup script
└── data/
    ├── test.jsonl        # Sample evaluation data
    └── results/          # Output directory
```

## Local Model Deployment (vLLM)

For evaluating custom or fine-tuned models, use vLLM to serve the model locally.

### Prerequisites

```bash
# Install vLLM (requires CUDA)
pip install vllm
```

### Configuration

Edit `test_local_model.sh`:

```bash
# Required: Set your model path
MODEL_PATH="/path/to/your/model"  # or "Qwen/Qwen2.5-Coder-7B-Instruct"

# Optional: Customize settings
MODEL_NAME="my-model"      # Name for API requests
PORT=8006                  # Server port
GPU_DEVICES="0"            # GPU to use
TENSOR_PARALLEL_SIZE=1     # Number of GPUs for parallelism
```

### Running

**Terminal 1 - Start Server:**
```bash
bash test_local_model.sh
# Wait for: "Uvicorn running on http://0.0.0.0:8006"
```

**Terminal 2 - Run Evaluation:**
```bash
# Edit run_eval.sh:
MODEL_NAME="my-model"
GEN_SERVER_URL="http://127.0.0.1:8006/v1/completions"

# Then run:
bash run_eval.sh
```

### vLLM Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `MODEL_PATH` | Model weights path or HuggingFace ID | (required) |
| `MODEL_NAME` | Served model name | `zed` |
| `PORT` | API server port | `8006` |
| `GPU_DEVICES` | CUDA visible devices | `0` |
| `TENSOR_PARALLEL_SIZE` | GPUs for tensor parallelism | `1` |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction | `0.93` |
| `MAX_MODEL_LEN` | Maximum sequence length | `6000` |
| `ENABLE_PREFIX_CACHING` | Enable KV cache reuse | `true` |
| `ENABLE_CHUNKED_PREFILL` | Enable chunked prefill | `true` |

## Troubleshooting

### API Key Not Found
```
ValueError: API key not found. Please set the MODELSCOPE_API_KEY environment variable.
```
**Solution**: Export the required API key before running.

### Local Model Connection Error
```
requests.exceptions.ConnectionError: ...
```
**Solution**: Ensure your local model server is running at the specified URL.

### Chat Template Error
```
ValueError: Tokenizer is required for chat mode.
```
**Solution**: Provide `--tokenizer_path` when using `--is_chat`.

## License

[Add your license here]

## Citation

```bibtex
@misc{code-edit-evaluation,
    title={Code Edit Prediction Evaluation Framework},
    author={Your Name},
    year={2024},
    url={https://github.com/your-username/code-edit-evaluation}
}
```