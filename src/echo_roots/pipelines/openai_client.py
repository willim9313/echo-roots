"""OpenAI client implementation for LLM extraction.

Provides integration with OpenAI's API for real LLM-based extraction.
Supports both synchronous and asynchronous calls with proper error handling.
"""

import os
import asyncio
from typing import Optional, Dict, Any
import openai
from openai import AsyncOpenAI

from echo_roots.pipelines.extraction import LLMClient, ExtractionError


class OpenAIClient:
    """OpenAI client implementation for LLM extraction.
    
    Provides integration with OpenAI's GPT models for attribute and term extraction.
    Handles API authentication, rate limiting, and error recovery.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 organization: Optional[str] = None,
                 base_url: Optional[str] = None):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            organization: OpenAI organization ID (optional)
            base_url: Custom API base URL (for API proxies)
        """
        # Use environment variable if no API key provided
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize async client
        client_kwargs = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url
            
        self.client = AsyncOpenAI(**client_kwargs)
        self.api_key = api_key
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt using OpenAI's API.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (model, temperature, max_tokens, etc.)
            
        Returns:
            LLM response text
            
        Raises:
            ExtractionError: If API call fails
        """
        try:
            # Default parameters
            params = {
                "model": kwargs.get("model", "gpt-4"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.1),
                "max_tokens": kwargs.get("max_tokens", 2000),
            }
            
            # Add any additional OpenAI-specific parameters
            if "top_p" in kwargs:
                params["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs:
                params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                params["presence_penalty"] = kwargs["presence_penalty"]
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            # Extract response text
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content or ""
            else:
                raise ExtractionError("Empty response from OpenAI API")
                
        except openai.APIError as e:
            raise ExtractionError(f"OpenAI API error: {e}")
        except openai.RateLimitError as e:
            raise ExtractionError(f"OpenAI rate limit exceeded: {e}")
        except openai.AuthenticationError as e:
            raise ExtractionError(f"OpenAI authentication failed: {e}")
        except Exception as e:
            raise ExtractionError(f"Unexpected error calling OpenAI API: {e}")
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()
    
    def __str__(self) -> str:
        return f"OpenAIClient(api_key={'***' if self.api_key else None})"


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI client implementation.
    
    Specialized client for Azure's OpenAI service with endpoint and deployment handling.
    """
    
    def __init__(self,
                 azure_endpoint: str,
                 deployment_name: str,
                 api_version: str = "2024-02-15-preview",
                 api_key: Optional[str] = None):
        """Initialize Azure OpenAI client.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            deployment_name: Azure deployment name
            api_version: Azure API version
            api_key: Azure OpenAI API key (uses AZURE_OPENAI_API_KEY env var if None)
        """
        # Use environment variable if no API key provided
        if api_key is None:
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        # Initialize client with Azure configuration
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{azure_endpoint}/openai/deployments/{deployment_name}",
            api_version=api_version
        )
        self.api_key = api_key
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt using Azure OpenAI.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
                     Note: model parameter is ignored (uses deployment)
            
        Returns:
            LLM response text
        """
        # Remove model parameter since Azure uses deployments
        azure_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
        
        # Call parent implementation
        return await super().complete(prompt, **azure_kwargs)
    
    def __str__(self) -> str:
        return f"AzureOpenAIClient(endpoint={self.azure_endpoint}, deployment={self.deployment_name})"


# Convenience functions for creating clients
def create_openai_client(api_key: Optional[str] = None) -> OpenAIClient:
    """Create a standard OpenAI client.
    
    Args:
        api_key: OpenAI API key (uses environment variable if None)
        
    Returns:
        Configured OpenAIClient
    """
    return OpenAIClient(api_key=api_key)


def create_azure_client(azure_endpoint: str,
                       deployment_name: str,
                       api_version: str = "2024-02-15-preview",
                       api_key: Optional[str] = None) -> AzureOpenAIClient:
    """Create an Azure OpenAI client.
    
    Args:
        azure_endpoint: Azure OpenAI endpoint URL
        deployment_name: Azure deployment name  
        api_version: Azure API version
        api_key: Azure API key (uses environment variable if None)
        
    Returns:
        Configured AzureOpenAIClient
    """
    return AzureOpenAIClient(
        azure_endpoint=azure_endpoint,
        deployment_name=deployment_name,
        api_version=api_version,
        api_key=api_key
    )


class MockLLMClient:
    """Mock LLM client for testing purposes.
    
    Provides realistic but deterministic responses for testing
    the extraction pipeline without making actual API calls.
    """
    
    def __init__(self):
        """Initialize mock client."""
        self.call_count = 0
    
    async def extract_structured_data(self, prompt: str, model: str, 
                                     temperature: float = 0.1, max_tokens: int = 1500,
                                     timeout: float = 30.0) -> Dict[str, Any]:
        """Generate mock structured data response.
        
        Args:
            prompt: The extraction prompt (analyzed for content)
            model: Model name (not used in mock)
            temperature: Temperature (affects randomness simulation)
            max_tokens: Max tokens (affects response size)
            timeout: Timeout (not used in mock)
            
        Returns:
            Mock structured data response
        """
        self.call_count += 1
        
        # Simulate processing delay
        await asyncio.sleep(0.01)
        
        # Analyze prompt to generate realistic responses
        prompt_lower = prompt.lower()
        
        # Generate mock attributes based on prompt content
        attributes = []
        if "title" in prompt_lower or "name" in prompt_lower:
            attributes.append({
                "name": "title",
                "value": self._extract_mock_title(prompt),
                "evidence": "Extracted from product description",
                "confidence": 0.95
            })
        
        if "price" in prompt_lower or "cost" in prompt_lower or "$" in prompt:
            attributes.append({
                "name": "price",
                "value": "$99.99",
                "evidence": "Found price information in text",
                "confidence": 0.90
            })
        
        if "category" in prompt_lower or "type" in prompt_lower:
            attributes.append({
                "name": "category",
                "value": "electronics",
                "evidence": "Categorized based on product description",
                "confidence": 0.85
            })
        
        if "availability" in prompt_lower or "available" in prompt_lower:
            attributes.append({
                "name": "availability",
                "value": "true",
                "evidence": "Availability mentioned in text",
                "confidence": 0.80
            })
        
        # Generate mock terms based on prompt content
        terms = []
        if "iphone" in prompt_lower:
            terms.extend([
                {
                    "term": "smartphone",
                    "context": "mobile device category",
                    "confidence": 0.90
                },
                {
                    "term": "apple",
                    "context": "brand name",
                    "confidence": 0.95
                }
            ])
        
        # Add some generic terms
        terms.extend([
            {
                "term": "product",
                "context": "general item classification",
                "confidence": 0.75
            },
            {
                "term": "description",
                "context": "textual content",
                "confidence": 0.70
            }
        ])
        
        # Ensure we have at least some data
        if not attributes:
            attributes.append({
                "name": "description",
                "value": "Mock product description",
                "evidence": "Generated from text content",
                "confidence": 0.60
            })
        
        if not terms:
            terms.append({
                "term": "item",
                "context": "general product term",
                "confidence": 0.65
            })
        
        return {
            "attributes": attributes,
            "terms": terms
        }
    
    def _extract_mock_title(self, prompt: str) -> str:
        """Extract a mock title from the prompt text."""
        # Look for common title patterns
        lines = prompt.split('\n')
        for line in lines:
            line = line.strip()
            # Skip instruction lines
            if any(word in line.lower() for word in ['extract', 'required', 'respond', 'json']):
                continue
            # Look for product-like names
            if len(line) > 5 and len(line) < 100:
                # Check if it looks like a product name
                if any(char.isupper() for char in line) and not line.endswith(':'):
                    return line
        
        # Fallback to generic title
        return "Mock Product Title"
