"""
LLM client factory for creating different provider clients.

Provides a unified interface for creating LLM clients from different providers
(OpenAI, Gemini, Anthropic, etc.) with consistent configuration handling.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml

from echo_roots.pipelines.extraction import LLMClient, ExtractionError


logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Factory for creating LLM clients from different providers."""
    
    # Supported providers
    SUPPORTED_PROVIDERS = {
        "openai": "OpenAI GPT models",
        "gemini": "Google Gemini models", 
        "anthropic": "Anthropic Claude models",
        "azure": "Azure OpenAI Service",
        "vllm": "vLLM OpenAI-compatible API",
        "mock": "Mock client for testing"
    }
    
    @classmethod
    def create_client(cls, 
                     provider: str,
                     config: Optional[Dict[str, Any]] = None,
                     **kwargs) -> LLMClient:
        """Create LLM client for specified provider.
        
        Args:
            provider: Provider name (openai, gemini, anthropic, azure, vllm, mock)
            config: Configuration dictionary
            **kwargs: Additional parameters for client creation
            
        Returns:
            Configured LLM client
            
        Raises:
            ValueError: If provider is not supported
            ImportError: If required SDK is not installed
            ExtractionError: If client creation fails
        """
        provider = provider.lower().strip()
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{provider}'. "
                f"Supported providers: {list(cls.SUPPORTED_PROVIDERS.keys())}"
            )
        
        config = config or {}
        
        try:
            if provider == "openai":
                return cls._create_openai_client(config, **kwargs)
            elif provider == "gemini":
                return cls._create_gemini_client(config, **kwargs)
            elif provider == "anthropic":
                return cls._create_anthropic_client(config, **kwargs)
            elif provider == "azure":
                return cls._create_azure_client(config, **kwargs)
            elif provider == "vllm":
                return cls._create_vllm_client(config, **kwargs)
            elif provider == "mock":
                return cls._create_mock_client(config, **kwargs)
            else:
                raise ValueError(f"Provider '{provider}' not implemented")
                
        except ImportError as e:
            raise ImportError(
                f"Failed to create {provider} client: {e}. "
                f"Install required dependencies with: pip install 'echo-roots[llm]'"
            )
        except Exception as e:
            raise ExtractionError(f"Failed to create {provider} client: {e}")
    
    @classmethod
    def _create_openai_client(cls, config: Dict[str, Any], **kwargs):
        """Create OpenAI client."""
        from echo_roots.pipelines.openai_client import OpenAIClient
        
        openai_config = config.get("openai", {})
        
        api_key = kwargs.get("api_key") or openai_config.get("api_key") or config.get("api_key")
        organization = kwargs.get("organization") or openai_config.get("organization")
        base_url = kwargs.get("base_url") or openai_config.get("base_url")
        
        return OpenAIClient(
            api_key=api_key,
            organization=organization,
            base_url=base_url
        )
    
    @classmethod  
    def _create_gemini_client(cls, config: Dict[str, Any], **kwargs):
        """Create Gemini client."""
        from echo_roots.pipelines.gemini_client import create_gemini_from_config
        
        # Merge config with kwargs
        merged_config = {**config, **kwargs}
        
        return create_gemini_from_config(merged_config)
    
    @classmethod
    def _create_anthropic_client(cls, config: Dict[str, Any], **kwargs):
        """Create Anthropic client."""
        try:
            from echo_roots.pipelines.anthropic_client import AnthropicClient
        except ImportError:
            # Create a basic Anthropic client if our module doesn't exist
            import anthropic
            
            class BasicAnthropicClient:
                def __init__(self, api_key=None, model_name="claude-3-sonnet-20240229"):
                    self.client = anthropic.AsyncAnthropic(
                        api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
                    )
                    self.model_name = model_name
                
                async def complete(self, prompt: str, **kwargs) -> str:
                    response = await self.client.messages.create(
                        model=self.model_name,
                        max_tokens=kwargs.get("max_tokens", 2000),
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text
            
            anthropic_config = config.get("anthropic", {})
            api_key = kwargs.get("api_key") or anthropic_config.get("api_key") or config.get("api_key")
            model_name = kwargs.get("model_name") or anthropic_config.get("model_name", "claude-3-sonnet-20240229")
            
            return BasicAnthropicClient(api_key=api_key, model_name=model_name)
    
    @classmethod
    def _create_azure_client(cls, config: Dict[str, Any], **kwargs):
        """Create Azure OpenAI client."""
        from echo_roots.pipelines.openai_client import AzureOpenAIClient
        
        azure_config = config.get("azure", {})
        
        api_key = kwargs.get("api_key") or azure_config.get("api_key") or config.get("api_key")
        endpoint = kwargs.get("endpoint") or azure_config.get("endpoint")
        deployment_name = kwargs.get("deployment_name") or azure_config.get("deployment_name")
        api_version = kwargs.get("api_version") or azure_config.get("api_version", "2024-02-15-preview")
        
        if not endpoint:
            raise ValueError("Azure endpoint is required")
        if not deployment_name:
            raise ValueError("Azure deployment_name is required")
        
        return AzureOpenAIClient(
            azure_endpoint=endpoint,
            deployment_name=deployment_name,
            api_version=api_version,
            api_key=api_key
        )
    
    @classmethod
    def _create_vllm_client(cls, config: Dict[str, Any], **kwargs):
        """Create vLLM client (OpenAI-compatible API)."""
        from echo_roots.pipelines.vllm_client import create_vllm_from_config
        
        # Merge config with kwargs
        merged_config = {**config, **kwargs}
        
        return create_vllm_from_config(merged_config)
    
    @classmethod
    def _create_mock_client(cls, config: Dict[str, Any], **kwargs):
        """Create mock client for testing."""
        from echo_roots.pipelines.extraction import MockLLMClient
        return MockLLMClient()
    
    @classmethod
    def create_from_config_file(cls, 
                               config_path: Union[str, Path],
                               provider: Optional[str] = None) -> LLMClient:
        """Create client from YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            provider: Override provider from config file
            
        Returns:
            Configured LLM client
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Get provider from parameter or config
        if provider is None:
            llm_config = config.get("llm", {})
            provider = llm_config.get("provider")
        
        if not provider:
            raise ValueError("Provider must be specified in config file or as parameter")
        
        return cls.create_client(provider, config)
    
    @classmethod
    def create_from_environment(cls, provider: Optional[str] = None) -> LLMClient:
        """Create client from environment variables.
        
        Args:
            provider: LLM provider (uses LLM_PROVIDER env var if None)
            
        Returns:
            Configured LLM client
        """
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", "openai")
        
        # Create minimal config from environment
        config = {
            "llm": {
                "provider": provider
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model_name": os.getenv("OPENAI_MODEL", "gpt-4")
            },
            "gemini": {
                "api_key": os.getenv("GOOGLE_API_KEY"),
                "model_name": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                "project_id": os.getenv("GOOGLE_PROJECT_ID")
            },
            "anthropic": {
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "model_name": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            },
            "azure": {
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            },
            "vllm": {
                "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                "model_name": os.getenv("VLLM_MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf"),
                "api_key": os.getenv("VLLM_API_KEY", "dummy-key"),
                "deployment_type": os.getenv("VLLM_DEPLOYMENT_TYPE", "local")
            }
        }
        
        return cls.create_client(provider, config)
    
    @classmethod
    def list_providers(cls) -> Dict[str, str]:
        """List available providers and their descriptions.
        
        Returns:
            Dictionary of provider names and descriptions
        """
        return cls.SUPPORTED_PROVIDERS.copy()
    
    @classmethod
    def check_provider_availability(cls, provider: str) -> tuple[bool, str]:
        """Check if a provider is available and properly configured.
        
        Args:
            provider: Provider name to check
            
        Returns:
            Tuple of (is_available, status_message)
        """
        provider = provider.lower().strip()
        
        if provider not in cls.SUPPORTED_PROVIDERS:
            return False, f"Unsupported provider: {provider}"
        
        try:
            if provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    return False, "OPENAI_API_KEY environment variable not set"
                return True, "OpenAI client ready"
            
            elif provider == "gemini":
                from echo_roots.pipelines.gemini_client import check_gemini_availability
                return check_gemini_availability()
            
            elif provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    return False, "ANTHROPIC_API_KEY environment variable not set"
                return True, "Anthropic client ready"
            
            elif provider == "azure":
                import openai
                required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
                missing = [var for var in required_vars if not os.getenv(var)]
                if missing:
                    return False, f"Missing environment variables: {missing}"
                return True, "Azure OpenAI client ready"
            
            elif provider == "vllm":
                from echo_roots.pipelines.vllm_client import check_vllm_availability
                return check_vllm_availability()
            
            elif provider == "mock":
                return True, "Mock client always available"
            
            else:
                return False, f"Provider {provider} not implemented"
                
        except ImportError as e:
            return False, f"Required SDK not installed: {e}"
        except Exception as e:
            return False, f"Configuration error: {e}"


# Convenience functions
def create_llm_client(provider: str = None, config_file: str = None, **kwargs) -> LLMClient:
    """Convenience function to create LLM client.
    
    Args:
        provider: Provider name (if None, uses environment or config file)
        config_file: Path to config file (if None, uses environment)
        **kwargs: Additional client parameters
        
    Returns:
        Configured LLM client
    """
    if config_file:
        return LLMClientFactory.create_from_config_file(config_file, provider)
    elif provider:
        return LLMClientFactory.create_from_environment(provider)
    else:
        return LLMClientFactory.create_from_environment()


def get_available_providers() -> Dict[str, tuple[bool, str]]:
    """Get status of all providers.
    
    Returns:
        Dict mapping provider names to (is_available, status_message) tuples
    """
    status = {}
    for provider in LLMClientFactory.SUPPORTED_PROVIDERS:
        status[provider] = LLMClientFactory.check_provider_availability(provider)
    return status


# Export main classes and functions
__all__ = [
    "LLMClientFactory",
    "create_llm_client", 
    "get_available_providers"
]
