"""
vLLM client implementation using OpenAI-compatible API.

vLLM provides high-throughput and memory-efficient inference for LLMs
with an OpenAI-compatible HTTP API. This client allows integration with
locally hosted or cloud-deployed vLLM servers.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

from echo_roots.pipelines.extraction import ExtractionError


logger = logging.getLogger(__name__)


class VLLMClient:
    """vLLM client using OpenAI-compatible API.
    
    Connects to vLLM server deployments that expose OpenAI-compatible endpoints.
    Supports both local and remote vLLM instances with custom models.
    """
    
    def __init__(self, 
                 base_url: str,
                 model_name: str,
                 api_key: str = "dummy-key",
                 timeout: int = 60,
                 max_retries: int = 3):
        """Initialize vLLM client.
        
        Args:
            base_url: vLLM server base URL (e.g., 'http://localhost:8000/v1')
            model_name: Model name as configured in vLLM server
            api_key: API key (often not required for local vLLM, use dummy value)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK is required for vLLM client. Install with: "
                "pip install openai"
            )
        
        if not base_url:
            raise ValueError("base_url is required for vLLM client")
        
        if not model_name:
            raise ValueError("model_name is required for vLLM client")
        
        # Ensure base_url ends with /v1 for OpenAI compatibility
        if not base_url.endswith('/v1'):
            if base_url.endswith('/'):
                base_url = base_url + 'v1'
            else:
                base_url = base_url + '/v1'
        
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        
        # Initialize OpenAI client pointing to vLLM server
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        
        logger.info(f"Initialized vLLM client: {base_url} with model {model_name}")
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt using vLLM server.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response text
            
        Raises:
            ExtractionError: If API call fails
        """
        try:
            # Prepare generation parameters
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 2000),
                "top_p": kwargs.get("top_p", 0.9),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "stream": False
            }
            
            # Add vLLM-specific parameters if provided
            if "top_k" in kwargs:
                params["extra_body"] = {"top_k": kwargs["top_k"]}
            
            # Call vLLM server
            response = await self.client.chat.completions.create(**params)
            
            # Extract response text
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content.strip()
            
            raise ExtractionError("Empty response from vLLM server")
            
        except openai.APIError as e:
            logger.error(f"vLLM API error: {e}")
            raise ExtractionError(f"vLLM API call failed: {e}")
        except Exception as e:
            logger.error(f"vLLM client error: {e}")
            raise ExtractionError(f"vLLM client error: {e}")
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters for generation
            
        Returns:
            List of response texts in same order as prompts
        """
        tasks = [self.complete(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        try:
            models = await self.client.models.list()
            for model in models.data:
                if model.id == self.model_name:
                    return {
                        "id": model.id,
                        "object": model.object,
                        "created": getattr(model, 'created', None),
                        "owned_by": getattr(model, 'owned_by', 'vllm')
                    }
            
            # If exact match not found, return basic info
            return {
                "id": self.model_name,
                "object": "model",
                "owned_by": "vllm"
            }
            
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {
                "id": self.model_name,
                "object": "model",
                "owned_by": "vllm",
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """Check if vLLM server is responding.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try a simple completion
            response = await self.complete("Hello", max_tokens=10)
            return len(response) > 0
            
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False
    
    def update_config(self, **config_updates):
        """Update client configuration.
        
        Args:
            **config_updates: Configuration parameters to update
        """
        if "timeout" in config_updates:
            # Note: OpenAI client doesn't support runtime timeout updates
            logger.warning("Timeout updates require client recreation")
        
        if "model_name" in config_updates:
            self.model_name = config_updates["model_name"]
            logger.info(f"Updated vLLM model to: {self.model_name}")
    
    async def close(self):
        """Close client connections."""
        try:
            await self.client.close()
        except Exception as e:
            logger.warning(f"Error closing vLLM client: {e}")
        
        logger.debug("vLLM client closed")
    
    def __repr__(self) -> str:
        return f"VLLMClient(base_url='{self.base_url}', model='{self.model_name}')"


class VLLMLocalClient(VLLMClient):
    """vLLM client optimized for local deployments.
    
    Pre-configured for common local vLLM setups with reasonable defaults.
    """
    
    def __init__(self, 
                 model_name: str,
                 port: int = 8000,
                 host: str = "localhost",
                 **kwargs):
        """Initialize local vLLM client.
        
        Args:
            model_name: Model name as configured in vLLM server
            port: vLLM server port (default: 8000)
            host: vLLM server host (default: localhost)
            **kwargs: Additional VLLMClient parameters
        """
        base_url = f"http://{host}:{port}/v1"
        
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key="dummy-key",  # Local vLLM usually doesn't need auth
            **kwargs
        )


class VLLMCloudClient(VLLMClient):
    """vLLM client for cloud deployments.
    
    Configured for remote vLLM deployments with authentication and 
    higher timeout values.
    """
    
    def __init__(self,
                 base_url: str, 
                 model_name: str,
                 api_key: Optional[str] = None,
                 **kwargs):
        """Initialize cloud vLLM client.
        
        Args:
            base_url: Remote vLLM server URL
            model_name: Model name
            api_key: API key for authentication (uses VLLM_API_KEY env var if None)
            **kwargs: Additional VLLMClient parameters
        """
        if api_key is None:
            api_key = os.getenv('VLLM_API_KEY', 'dummy-key')
        
        # Cloud deployments typically need longer timeouts
        kwargs.setdefault('timeout', 120)
        kwargs.setdefault('max_retries', 5)
        
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )


def create_vllm_from_config(config: Dict[str, Any]) -> VLLMClient:
    """Create vLLM client from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured VLLMClient
        
    Example config:
        {
            "vllm": {
                "base_url": "http://localhost:8000/v1",
                "model_name": "meta-llama/Llama-2-7b-chat-hf",
                "api_key": "dummy-key",
                "deployment_type": "local"  # or "cloud"
            }
        }
    """
    vllm_config = config.get("vllm", {})
    
    deployment_type = vllm_config.get("deployment_type", "local")
    model_name = vllm_config.get("model_name") or vllm_config.get("model")
    
    if not model_name:
        raise ValueError("model_name is required in vLLM config")
    
    if deployment_type == "local":
        port = vllm_config.get("port", 8000)
        host = vllm_config.get("host", "localhost")
        
        return VLLMLocalClient(
            model_name=model_name,
            port=port,
            host=host,
            timeout=vllm_config.get("timeout", 60),
            max_retries=vllm_config.get("max_retries", 3)
        )
    
    elif deployment_type == "cloud":
        base_url = vllm_config.get("base_url") or vllm_config.get("endpoint")
        if not base_url:
            raise ValueError("base_url is required for cloud vLLM deployment")
        
        return VLLMCloudClient(
            base_url=base_url,
            model_name=model_name,
            api_key=vllm_config.get("api_key"),
            timeout=vllm_config.get("timeout", 120),
            max_retries=vllm_config.get("max_retries", 5)
        )
    
    else:
        # Generic client
        base_url = vllm_config.get("base_url") or vllm_config.get("endpoint")
        if not base_url:
            raise ValueError("base_url is required for vLLM client")
        
        return VLLMClient(
            base_url=base_url,
            model_name=model_name,
            api_key=vllm_config.get("api_key", "dummy-key"),
            timeout=vllm_config.get("timeout", 60),
            max_retries=vllm_config.get("max_retries", 3)
        )


def check_vllm_availability() -> tuple[bool, str]:
    """Check if vLLM integration is available.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    if not OPENAI_AVAILABLE:
        return False, "OpenAI SDK not installed (required for vLLM client)"
    
    # Check for vLLM configuration
    base_url = os.getenv("VLLM_BASE_URL")
    model_name = os.getenv("VLLM_MODEL_NAME") 
    
    if base_url and model_name:
        return True, f"vLLM configured: {base_url} with {model_name}"
    elif base_url:
        return False, "VLLM_MODEL_NAME environment variable not set"
    elif model_name:
        return False, "VLLM_BASE_URL environment variable not set" 
    else:
        return False, "VLLM_BASE_URL and VLLM_MODEL_NAME environment variables not set"


# Export main classes and functions
__all__ = [
    "VLLMClient",
    "VLLMLocalClient", 
    "VLLMCloudClient",
    "create_vllm_from_config",
    "check_vllm_availability"
]
