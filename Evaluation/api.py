"""
API Client for External LLM Services

Supported models:
1. Qwen/Qwen3-Coder-480B-A35B-Instruct - via ModelScope API
   - Requires: MODELSCOPE_API_KEY environment variable
   - Base URL: https://api-inference.modelscope.cn

2. claude-4-sonnet-20250514 - via Anthropic API  
   - Requires: ANTHROPIC_API_KEY environment variable

Usage:
    export MODELSCOPE_API_KEY="your-modelscope-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
"""

import anthropic
import os
from typing import Optional


class APIClient:
    """
    Unified API client supporting both Anthropic and ModelScope endpoints.
    Uses the Anthropic SDK which is compatible with ModelScope's API format.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            api_key: API key for authentication
            base_url: Custom API endpoint (optional, for ModelScope or proxy services)
        """
        if base_url:
            self.client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    def chat(self, model_name: str, max_token: int, message: str):
        """
        Send a chat request to the API.
        
        Args:
            model_name: Name of the model to use
            max_token: Maximum tokens for response
            message: User message content
        
        Returns:
            Model response text, or None if error occurs
        """
        try:
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_token,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=0.01  # Use small value instead of 0 for compatibility
            )
            return response.content[0].text
        except Exception as e:
            print(f"API Error: {e}")
            return None


class GPTHelper:
    """
    High-level helper class for model inference.
    Automatically selects the appropriate API configuration based on model name.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        "Qwen/Qwen3-Coder-480B-A35B-Instruct": {
            "env_key": "MODELSCOPE_API_KEY",
            "base_url": "https://api-inference.modelscope.cn"
        },
        "claude-4-sonnet-20250514": {
            "env_key": "ANTHROPIC_API_KEY",
            "base_url": None  # Use default Anthropic endpoint
        }
    }
    
    def __init__(self, model_name: str):
        """
        Initialize GPTHelper with specified model.
        
        Args:
            model_name: Name of the model to use
            
        Raises:
            ValueError: If model is not supported or API key is not set
        """
        self.model_name = model_name
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}. "
                           f"Supported models: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_name]
        api_key = os.environ.get(config["env_key"], "")
        
        if not api_key:
            raise ValueError(f"API key not found. Please set the {config['env_key']} environment variable.\n"
                           f"Example: export {config['env_key']}='your-api-key'")
        
        self.client = APIClient(api_key=api_key, base_url=config["base_url"])
        print(f"Initialized API client for: {model_name}")
    
    def get_result(self, message: str, max_new_tokens: int = 1024, temperature: float = 0):
        """
        Get model response for the given message.
        
        Args:
            message: Input message/prompt
            max_new_tokens: Maximum tokens for response
            temperature: Sampling temperature (not used, kept for interface compatibility)
        
        Returns:
            Model response text
        """
        return self.client.chat(self.model_name, max_token=max_new_tokens, message=message)
