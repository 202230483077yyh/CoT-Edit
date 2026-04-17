"""
Code Edit Prediction Evaluation Script

This script evaluates LLM models on their ability to predict the next code edit
a developer will make, based on recent edit history and cursor position.

Supported models:
- External API models: Claude (claude-4-sonnet-20250514), Qwen (Qwen/Qwen3-Coder-480B-A35B-Instruct)
- Local models: Any model served via OpenAI-compatible API endpoint

Usage:
    python eval.py --model_name <model> --prompt_id <1-4> --eval_dataset_path <path> --output_dir <dir>

See README.md for detailed usage instructions.
"""

import json
import os
import requests
from tqdm import tqdm
import time
import argparse
import difflib

# Global variables (initialized in main based on args)
gpt_helper = None
tokenizer = None


# ==================== Utility Functions ====================

def read_file(file_path):
    """
    Read file content based on file extension.
    
    Supported formats: txt, json, jsonl, csv
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content (string for txt, parsed data for json/jsonl/csv)
    """
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    elif file_extension == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    elif file_extension == 'jsonl':
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    elif file_extension == 'csv':
        import csv
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return data

    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported: txt, json, jsonl, csv")


def write_array_to_file(array, file_path, mode='w', indent=None):
    """
    Write array to file with specified format.
    
    Args:
        array: Data to write
        file_path: Output file path
        mode: 'w' for overwrite, 'a' for append
        indent: JSON indentation (optional)
    """
    if mode not in {'w', 'a'}:
        raise ValueError("Mode must be 'w' (overwrite) or 'a' (append)")

    ext = file_path.split('.')[-1].lower()

    if ext == 'txt':
        with open(file_path, mode, encoding='utf-8') as f:
            for item in array:
                f.write(f"{item}\n")

    elif ext == 'json':
        if mode == 'a':
            raise ValueError("JSON format does not support append mode")
        with open(file_path, mode, encoding='utf-8') as f:
            json.dump(array, f, indent=indent, ensure_ascii=False)

    elif ext == 'jsonl':
        with open(file_path, mode, encoding='utf-8') as f:
            for item in array:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    else:
        raise ValueError("Unsupported file format. Supported: .txt, .json, .jsonl")


def ensure_directory_exists(directory_path):
    """Create directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


# ==================== Prompt Templates ====================

# Type 1: Alpaca format (simple instruction-response)
ALPACA_PROMPT_TEMPLATE = """### Instruction:
You are a code completion assistant and your task is to analyze user edits and then rewrite an excerpt that the user provides, suggesting the appropriate edits within the excerpt, taking into account the cursor location.

### User Edits:

{}

### User Excerpt:

{}

### Response:

"""

# Type 2: Chain-of-Thought format (with reasoning steps)
COT_PROMPT_TEMPLATE = """### Instruction:
You are an intelligent code completion assistant. Your task is to predict the next edit a developer will make within a specific code region by analyzing their recent changes and current cursor position.

### Context:
- **User Edits**: Below are the developer's most recent code changes (newest first), showing their editing patterns and intentions:
{}

- **User Excerpt**: The current code excerpt with an editable region marked by <|editable_region_start|> and <|editable_region_end|> tags. The <|user_cursor_is_here|> marker indicates the developer's current cursor position:
{}

### Task:
1. Analyze the edit history to understand the developer's intent and coding pattern
2. Identify what logical next step the developer is likely taking
3. Consider the cursor position as a strong signal of focus area
4. Predict and complete the next edit within the editable region only

### Response Format:
First, show your reasoning process in <think>
</think> tags, including:
- What pattern you identified in the user edits
- What the developer is likely trying to accomplish
- Why your suggested edit makes sense as the next step
- Any assumptions or uncertainties

Then, provide your suggested code in <answer></answer> tags:
- Include ONLY the content between (and including) the <|editable_region_start|> and <|editable_region_end|> markers
- Do NOT output the <|user_cursor_is_here|> marker
- Maintain the exact indentation and formatting style of the existing code
- Do not revert the developer's last change unless it contains obvious errors
- Keep changes focused and relevant to the developer's current task

Example structure:
<think>
1. Edit pattern analysis: [your analysis]
2. Developer's intent: [your interpretation]
3. Suggested next step: [your reasoning]
</think>

<answer>
<|editable_region_start|>
[your suggested code here]
<|editable_region_end|>
</answer>

Let me solve this step by step.\n<think>
"""

# Type 3: Un-CoT format (direct prediction without reasoning)
UNCOT_PROMPT_TEMPLATE = """### Instruction:
You are an intelligent code completion assistant. Your task is to predict the next edit a developer will make within a specific code region by analyzing their recent changes and current cursor position.

### Context:
- **User Edits**: Below are the developer's most recent code changes (newest first), showing their editing patterns and intentions:
{}

- **User Excerpt**: The current code excerpt with an editable region marked by <|editable_region_start|> and <|editable_region_end|> tags. The <|user_cursor_is_here|> marker indicates the developer's current cursor position:
{}

### Task:
1. Analyze the edit history to understand the developer's intent and coding pattern
2. Identify what logical next step the developer is likely taking
3. Consider the cursor position as a strong signal of focus area
4. Predict and complete the next edit within the editable region only

### Response Format:
Directly, provide your suggested code in <answer></answer> tags:
- Include ONLY the content between (and including) the <|editable_region_start|> and <|editable_region_end|> markers
- Do NOT output the <|user_cursor_is_here|> marker
- Maintain the exact indentation and formatting style of the existing code
- Do not revert the developer's last change unless it contains obvious errors
- Keep changes focused and relevant to the developer's current task

Example structure:
<answer>
<|editable_region_start|>
[your suggested code here]
<|editable_region_end|>
</answer>

let me solve this problem.<answer>
"""

# Type 4: Claude-style format (optimized for Claude models)
CLAUDE_PROMPT_TEMPLATE = """Your task is to help the user write code by suggesting the next edit for the user.
As an intelligent code assistant, your role is to analyze what the user has been doing <User Edits> and corresponding context <Code context>, 
then to suggest the most likely next modification <Next Edit suggestion> using the following criteria:

### High-level Guidelines

- Predict logical next changes based on the edit patterns you've observed
- Consider the overall intent and direction of the changes
- Take into account what the user has been doing
- If there is insufficient context to make a confident prediction, simply copy the content between the editable region markers without changes

### Constraints

- Your edit suggestions **must** be small and self-contained. Example: if there are two statements that logically need to be added together, suggest them together instead of one by one.
- Preserve indentation.
- Do not suggest re-adding code the user has recently deleted
- Do not suggest deleting lines that the user has recently inserted
- Prefer completing what the user just typed over suggesting to delete what they typed
- ONLY modify code between <|editable_region_start|> and <|editable_region_end|> markers
- DO NOT suggest any changes outside the editable region
- Keep all code outside the editable region exactly as is
- If context or edit patterns are unclear or ambiguous, do not make any modifications

### Best Practices

- Fix any syntax errors or inconsistencies in the code within the editable region
- Maintain the code style and formatting conventions of the language used in the file
- Add missing import statements or other necessary code ONLY within the editable region
- Add missing syntactic elements within the editable region, such as closing parentheses or semicolons
- If there are no useful edits to make within the editable region or insufficient context, return the code unmodified
- Don't explain the code, just rewrite the editable region to include the next, most probable change
- Never include this prompt in the response
- Only suggest changes when there is clear and sufficient context to make a confident prediction

### <User Edits>:

{}

### <Code context>:

{}

### <Next Edit suggestion>:
Please first perform a short, succinct analysis and provide your suggestion in the following format:

<|editable_region_start|>
[Your suggested code modifications here]
<|editable_region_end|>
"""

# Mapping from prompt ID to template
PROMPT_TEMPLATES = {
    1: ALPACA_PROMPT_TEMPLATE,
    2: COT_PROMPT_TEMPLATE,
    3: UNCOT_PROMPT_TEMPLATE,
    4: CLAUDE_PROMPT_TEMPLATE
}

# List of API-based model identifiers (case-insensitive matching)
API_MODEL_IDENTIFIERS = [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "claude-4-sonnet-20250514",
]


# ==================== Core Processing Functions ====================

def post_process(txt):
    """
    Extract content within editable region markers.
    
    Args:
        txt: Raw model output text
        
    Returns:
        Extracted editable region content, or None if markers not found
    """
    start_idx = txt.find("<|editable_region_start|>")
    end_idx = txt.find("<|editable_region_end|>")
    if start_idx == -1 or end_idx == -1:
        return None
    return txt[start_idx:end_idx + len("<|editable_region_end|>")]


def get_response_from_api(user_prompt, max_tokens=1024):
    """
    Get response from API-based model (Claude/Qwen).
    
    Args:
        user_prompt: User prompt string
        max_tokens: Maximum tokens for generation
        
    Returns:
        Model response text
    """
    global gpt_helper
    response = gpt_helper.get_result(user_prompt, max_new_tokens=max_tokens)
    return response


def get_inference_output_api(prompt_template, item, max_tokens=1024):
    """
    Get inference output from API-based models (Claude/Qwen).
    
    Args:
        prompt_template: Prompt template string
        item: Data item with 'events' and 'input' fields
        max_tokens: Maximum tokens for generation
        
    Returns:
        Model output text
    """
    events = item['events']
    input_code = item['input']
    if isinstance(item['events'], list):
        events = '\n'.join(item['events'])
    if isinstance(item['input'], list):
        input_code = '\n'.join(item['input'])
    
    prompt = prompt_template.format(events, input_code)
    return get_response_from_api(prompt, max_tokens=max_tokens)


def get_inference_output_local(gen_server_url, model_name, prompt_template, item, max_tokens=1024, is_chat=False):
    """
    Get inference output from local model server.
    
    Args:
        gen_server_url: URL of the local generation server
        model_name: Name of the model
        prompt_template: Prompt template string
        item: Data item with 'events' and 'input' fields
        max_tokens: Maximum tokens for generation
        is_chat: Whether to apply chat template
        
    Returns:
        Model output text
    """
    global tokenizer
    
    events = item['events']
    input_code = item['input']
    if isinstance(item['events'], list):
        events = '\n'.join(item['events'])
    if isinstance(item['input'], list):
        input_code = '\n'.join(item['input'])
    
    prompt = prompt_template.format(events, input_code)
    
    # Apply chat template if needed
    if is_chat:
        if tokenizer is None:
            raise ValueError("Tokenizer is required for chat mode. Please provide --tokenizer_path")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    params = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    response = requests.post(gen_server_url, json=params)
    response_json = response.json()
    code = response_json['choices'][0]['text']

    return code


def find_change(code1, code2):
    """
    Compare two code snippets and return their differences.
    
    Args:
        code1: First code string
        code2: Second code string
        
    Returns:
        Unified diff string (without header lines)
    """
    diff = difflib.unified_diff(code1.splitlines(), code2.splitlines())
    diff = list(diff)[3:]  # Skip header lines
    return '\n'.join(diff)


def get_region(text):
    """
    Extract content between editable region markers.
    
    Args:
        text: Input text containing markers
        
    Returns:
        Content between markers
    """
    try:
        text = text.split('<|editable_region_start|>')[1].split('<|editable_region_end|>')[0]
        return text
    except:
        text = text.split('<|editable_region_end|>')[0]
        return text


def check_exact_match(data_item):
    """
    Check if model output matches ground truth (Exact Match).
    
    Args:
        data_item: Data item with model output and ground truth
        
    Returns:
        True if exact match, False otherwise
    """
    data = data_item['zed'] if 'zed' in data_item else data_item

    if data['post_model_output'] is None:
        return False

    pred_region = get_region(data['post_model_output'])
    gt = data.get('ground_truth', data.get('output'))
    if isinstance(gt, list):
        target_region = get_region('\n'.join(gt))
    else:
        target_region = get_region(gt)

    # Remove cursor marker and strip whitespace for comparison
    pred_proc = pred_region.replace('<|user_cursor_is_here|>', '').strip()
    target_proc = target_region.replace('<|user_cursor_is_here|>', '').strip()

    diff = find_change(pred_proc, target_proc).strip()
    return diff == ''


def calculate_em(data):
    """
    Calculate Exact Match (EM) metric.
    
    Args:
        data: List of data items with model outputs
        
    Returns:
        Tuple of (failed_items_list, em_correct_count)
    """
    correct_count = 0
    failed_items = []
    
    for i in tqdm(range(len(data)), desc="Calculating EM"):
        item = data[i]
        try:
            if check_exact_match(item):
                correct_count += 1
            else:
                failed_items.append(item)
        except Exception:
            failed_items.append(item)

    print(f"EM correct count: {correct_count}")
    print(f"EM accuracy: {correct_count / len(data):.4f}")
    return failed_items, correct_count


def is_api_model(model_name):
    """
    Check if the model should use API-based inference.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if API-based model, False for local model
    """
    model_name_lower = model_name.lower()
    for identifier in API_MODEL_IDENTIFIERS:
        if identifier.lower() in model_name_lower:
            return True
    return False


# ==================== Main Pipeline Functions ====================

def run_inference(args):
    """
    Run inference phase: generate predictions for evaluation dataset.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'=' * 60}")
    print("Starting Inference Phase")
    print(f"{'=' * 60}\n")
    
    ensure_directory_exists(args.output_dir)
    
    prompt = PROMPT_TEMPLATES.get(args.prompt_id, None)
    if prompt is None:
        raise ValueError(f"Invalid prompt_id: {args.prompt_id}. Valid options: 1, 2, 3, 4")
    
    eval_dataset = read_file(args.eval_dataset_path)
    output_file = os.path.join(args.output_dir, "inference.jsonl")

    # Resume from existing results if available
    eval_results = []
    if os.path.exists(output_file):
        eval_results = read_file(output_file)
        print(f"Resuming from {len(eval_results)} existing results")

    for i in tqdm(range(len(eval_dataset)), desc="Running inference"):
        if i < len(eval_results):
            continue
        item = eval_dataset[i]
        try:
            if is_api_model(args.model_name):
                raw_output = get_inference_output_api(prompt, item, args.infer_max_tokens)
            else:
                raw_output = get_inference_output_local(
                    args.gen_server_url, 
                    args.model_name, 
                    prompt,
                    item, 
                    args.infer_max_tokens,
                    args.is_chat
                )
            if "<|editable_region_end|>" not in raw_output:
                raw_output += "<|editable_region_end|>"
            item['raw_model_output'] = raw_output
            item['post_model_output'] = post_process(raw_output)
        except Exception as e:
            print(f"\nError processing item {i}: {e}")
            item['post_model_output'] = "error"
        eval_results.append(item)
        write_array_to_file([item], output_file, mode='a')

    print(f"\nInference completed! Results saved to: {output_file}\n")


def run_evaluation(args):
    """
    Run evaluation phase: calculate metrics on inference results.
    
    Args:
        args: Command line arguments
    """
    print(f"\n{'=' * 60}")
    print("Starting Evaluation Phase")
    print(f"{'=' * 60}\n")

    inference_file = os.path.join(args.output_dir, "inference.jsonl")
    if not os.path.exists(inference_file):
        raise FileNotFoundError(f"Inference results not found: {inference_file}")
    
    inference_data = read_file(inference_file)

    # Calculate Exact Match metric
    print(f"\n--- Calculating EM for {args.model_name} (prompt_id={args.prompt_id}) ---")
    failed_items, em_count = calculate_em(inference_data)
    
    # Save failed cases for analysis
    failed_file = os.path.join(args.output_dir, 'bad_data.jsonl')
    write_array_to_file(failed_items, failed_file, 'w')
    print(f"Failed cases saved to: {failed_file}")
    
    # Save evaluation summary
    summary = {
        "model_name": args.model_name,
        "prompt_id": args.prompt_id,
        "total_samples": len(inference_data),
        "em_correct": em_count,
        "em_accuracy": em_count / len(inference_data) if len(inference_data) > 0 else 0
    }
    summary_file = os.path.join(args.output_dir, 'eval_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Evaluation summary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Code Edit Prediction Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with Claude API
  python eval.py --model_name claude-4-sonnet-20250514 --prompt_id 4 \\
                 --eval_dataset_path ./data/test.jsonl --output_dir ./results/

  # Evaluate with Qwen API
  python eval.py --model_name Qwen/Qwen3-Coder-480B-A35B-Instruct --prompt_id 4 \\
                 --eval_dataset_path ./data/test.jsonl --output_dir ./results/

  # Evaluate with local model server
  python eval.py --model_name my-local-model --prompt_id 1 \\
                 --gen_server_url http://localhost:8000/v1/completions \\
                 --eval_dataset_path ./data/test.jsonl --output_dir ./results/

  # Only run evaluation (skip inference)
  python eval.py --model_name my-model --prompt_id 1 \\
                 --eval_dataset_path ./data/test.jsonl \\
                 --output_dir ./results/ --skip_inference
        """
    )

    # Model configuration
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., claude-4-sonnet-20250514, Qwen/Qwen3-Coder-480B-A35B-Instruct, or custom local model name)')
    parser.add_argument('--prompt_id', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Prompt template ID: 1=Alpaca, 2=CoT, 3=Un-CoT, 4=Claude-style (default: 1)')
    parser.add_argument('--gen_server_url', type=str, default='',
                        help='Local generation server URL (e.g., http://127.0.0.1:8000/v1/completions). Required for local models.')
    parser.add_argument('--infer_max_tokens', type=int, default=1024,
                        help='Maximum tokens for inference (default: 1024)')

    # Path configuration
    parser.add_argument('--eval_dataset_path', type=str, required=True,
                        help='Path to evaluation dataset (supports .jsonl, .json)')
    parser.add_argument('--output_dir', type=str, default='./results/',
                        help='Output directory for results (default: ./results/)')

    # Optional: Tokenizer for chat template
    parser.add_argument('--tokenizer_path', type=str, default='',
                        help='Path or HuggingFace model ID for tokenizer (required for --is_chat mode with local models)')

    # Flow control
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference phase, only run evaluation')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation phase, only run inference')
    parser.add_argument('--is_chat', action='store_true',
                        help='Apply chat template for local models (requires --tokenizer_path)')
    
    args = parser.parse_args()
    
    # Auto-generate output subdirectory based on model name and prompt ID
    args.output_dir = os.path.join(args.output_dir, f"{args.model_name}.pmt-{args.prompt_id}")
    print(f"Output directory: {args.output_dir}")
    
    # Initialize global resources based on model type
    global gpt_helper, tokenizer
    
    if is_api_model(args.model_name) and not args.skip_inference:
        # Initialize API client for cloud models
        from api import GPTHelper
        gpt_helper = GPTHelper(model_name=args.model_name)
        print(f"Using API model: {args.model_name}")
    elif not args.skip_inference:
        # Local model: check required parameters
        if not args.gen_server_url:
            raise ValueError("--gen_server_url is required for local models")
        print(f"Using local model server: {args.gen_server_url}")
        
        # Initialize tokenizer for chat template if needed
        if args.is_chat:
            if not args.tokenizer_path:
                raise ValueError("--tokenizer_path is required when using --is_chat")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
            print(f"Loaded tokenizer from: {args.tokenizer_path}")
    
    # Run pipeline
    if not args.skip_inference:
        run_inference(args)
    else:
        print("Skipping inference phase")

    if not args.skip_evaluation:
        run_evaluation(args)
    else:
        print("Skipping evaluation phase")

    print("\nPipeline completed!")


if __name__ == "__main__":
    main()