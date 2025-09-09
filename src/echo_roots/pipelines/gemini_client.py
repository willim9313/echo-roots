"""
Google Gemini client implementation using official google.genai SDK.

Provides integration with Google's Gemini API for LLM-based extraction
in the echo-roots taxonomy management system.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

from echo_roots.pipelines.extraction import ExtractionError


logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini client implementation using official SDK.
    
    Provides integration with Google's Gemini models for attribute and term extraction.
    Implements the LLMClient protocol for use with the extraction pipeline.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash",
                 project_id: Optional[str] = None):
        """Initialize Gemini client.
        
        Args:
            api_key: Google AI API key (uses GOOGLE_API_KEY env var if None)
            model_name: Gemini model name to use
            project_id: Google Cloud project ID (optional)
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "Google Generative AI SDK is required. Install with: "
                "pip install google-genai"
            )
        
        # Use environment variable if no API key provided
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_key = api_key
        self.model_name = model_name
        self.project_id = project_id
        
        # Initialize the client
        self.client = genai.Client(api_key=api_key)
        
        # Default generation config optimized for structured extraction
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2000,
        }
        
        logger.info(f"Initialized Gemini client with model: {model_name}")

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

from echo_roots.pipelines.extraction import ExtractionError


logger = logging.getLogger(__name__)

import os
import asyncio
from typing import Optional, Dict, Any, List
import logging

try:
    import google.genai as genai
    from google.genai.types import GenerateContentConfig, SafetySetting, HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from echo_roots.pipelines.extraction import LLMClient, ExtractionError


logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini client implementation using official SDK.
    
    Provides integration with Google's Gemini models for attribute and term extraction.
    Uses the official google.genai SDK for optimal performance and feature support.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-1.5-flash",
                 project_id: Optional[str] = None):
        """Initialize Gemini client.
        
        Args:
            api_key: Google AI API key (uses GOOGLE_API_KEY env var if None)
            model_name: Gemini model name to use
            project_id: Google Cloud project ID (optional)
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "Google Generative AI SDK is required. Install with: "
                "pip install google-genai"
            )
        
        # Use environment variable if no API key provided
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.api_key = api_key
        self.model_name = model_name
        self.project_id = project_id
        
        # Initialize the client
        self.client = genai.Client(api_key=api_key)
        
        # Default generation config optimized for structured extraction
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2000,
        }
        
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt using Gemini API.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLM response text
            
        Raises:
            ExtractionError: If API call fails
        """
        try:
            # Merge generation config with kwargs
            config = self.generation_config.copy()
            if 'temperature' in kwargs:
                config['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                config['max_output_tokens'] = kwargs['max_tokens']
            
            # Enhance prompt for structured output
            enhanced_prompt = self._enhance_prompt_for_extraction(prompt)
            
            # Generate content using the client
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=enhanced_prompt,
                    config=config
                )
            )
            
            # Extract text from response
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # Try alternative response format
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                        if text_parts:
                            return ''.join(text_parts).strip()
            
            raise ExtractionError("Empty response from Gemini API")
            
        except Exception as e:
            if isinstance(e, ExtractionError):
                raise
            
            logger.error(f"Gemini API error: {e}")
            raise ExtractionError(f"Gemini API call failed: {e}")
    
    def _enhance_prompt_for_extraction(self, prompt: str) -> str:
        """Enhance prompt for better structured extraction results.
        
        Args:
            prompt: Original prompt text
            
        Returns:
            Enhanced prompt with JSON formatting instructions
        """
        enhanced = f"""You are an expert at extracting structured data from text. 

IMPORTANT: You must respond with ONLY valid JSON, no other text or explanations.

{prompt}

CRITICAL: Your response must be valid JSON that can be parsed by json.loads(). 
Do not include any markdown formatting, backticks, or explanatory text.
Start your response directly with {{ and end with }}"""
        
        return enhanced
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts efficiently.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional parameters for generation
            
        Returns:
            List of response texts in same order as prompts
        """
        tasks = [self.complete(prompt, **kwargs) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to empty strings or handle as needed
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch generation failed for prompt {i}: {result}")
                processed_results.append("")  # or raise exception
            else:
                processed_results.append(result)
        
        return processed_results
    
    def update_model_config(self, **config_updates):
        """Update the generation configuration.
        
        Args:
            **config_updates: Configuration parameters to update
        """
        # Update generation config
        current_config = {
            "temperature": self.generation_config.temperature,
            "top_p": self.generation_config.top_p,
            "top_k": self.generation_config.top_k,
            "max_output_tokens": self.generation_config.max_output_tokens,
        }
        
        current_config.update(config_updates)
        
        self.generation_config = GenerateContentConfig(**current_config)
        logger.info(f"Updated Gemini model config: {config_updates}")
    
    async def close(self):
        """Close the client (cleanup if needed)."""
        # Gemini SDK doesn't require explicit cleanup
        logger.debug("Gemini client closed")
    
    def __str__(self) -> str:
        return f"GeminiClient(model={self.model_name}, project_id={self.project_id})"


class GeminiProClient(GeminiClient):
    """Gemini Pro client with optimized settings for complex tasks.
    
    Uses Gemini Pro model with enhanced configuration for demanding
    extraction tasks requiring higher accuracy.
    """
    
    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None):
        """Initialize Gemini Pro client.
        
        Args:
            api_key: Google AI API key
            project_id: Google Cloud project ID
        """
        super().__init__(
            api_key=api_key,
            model_name="gemini-1.5-pro",
            project_id=project_id
        )
        
        # Enhanced config for Pro model
        self.generation_config = GenerateContentConfig(
            temperature=0.1,  # Lower temperature for more consistent results
            top_p=0.9,
            top_k=20,        # More focused sampling
            max_output_tokens=4000,  # Higher token limit
        )


class GeminiFlashClient(GeminiClient):
    """Gemini Flash client optimized for speed and efficiency.
    
    Uses Gemini Flash model with optimized settings for high-throughput
    extraction tasks where speed is prioritized over maximum accuracy.
    """
    
    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None):
        """Initialize Gemini Flash client.
        
        Args:
            api_key: Google AI API key
            project_id: Google Cloud project ID
        """
        super().__init__(
            api_key=api_key,
            model_name="gemini-1.5-flash",
            project_id=project_id
        )
        
        # Optimized config for speed
        self.generation_config = GenerateContentConfig(
            temperature=0.3,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2000,
        )


# Convenience functions for creating Gemini clients
def create_gemini_client(model_type: str = "flash", 
                        api_key: Optional[str] = None,
                        project_id: Optional[str] = None) -> GeminiClient:
    """Create a Gemini client of the specified type.
    
    Args:
        model_type: Type of model ("flash", "pro", or specific model name)
        api_key: Google AI API key (uses environment variable if None)
        project_id: Google Cloud project ID
        
    Returns:
        Configured Gemini client
        
    Raises:
        ValueError: If model_type is not supported
    """
    if model_type.lower() == "flash":
        return GeminiFlashClient(api_key=api_key, project_id=project_id)
    elif model_type.lower() == "pro":
        return GeminiProClient(api_key=api_key, project_id=project_id)
    elif model_type.startswith("gemini-"):
        return GeminiClient(
            api_key=api_key,
            model_name=model_type,
            project_id=project_id
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_gemini_from_config(config: Dict[str, Any]) -> GeminiClient:
    """Create Gemini client from configuration dictionary.
    
    Args:
        config: Configuration dictionary with Gemini settings
        
    Returns:
        Configured Gemini client
    """
    gemini_config = config.get("gemini", {})
    
    api_key = gemini_config.get("api_key") or config.get("api_key")
    model_name = gemini_config.get("model_name", "gemini-1.5-flash")
    project_id = gemini_config.get("project_id")
    
    client = GeminiClient(
        api_key=api_key,
        model_name=model_name,
        project_id=project_id
    )
    
    # Apply any additional configuration
    if "generation_config" in gemini_config:
        client.update_model_config(**gemini_config["generation_config"])
    
    return client


# Check SDK availability
def check_gemini_availability() -> tuple[bool, str]:
    """Check if Gemini SDK is available and properly configured.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    if not GENAI_AVAILABLE:
        return False, "google-genai SDK not installed. Install with: pip install google-genai"
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return False, "GOOGLE_API_KEY environment variable not set"
    
    try:
        # Test basic configuration
        genai.configure(api_key=api_key)
        return True, "Gemini SDK ready"
    except Exception as e:
        return False, f"Gemini configuration failed: {e}"


# Export the main classes for easier imports
__all__ = [
    "GeminiClient",
    "GeminiProClient", 
    "GeminiFlashClient",
    "create_gemini_client",
    "create_gemini_from_config",
    "check_gemini_availability",
    "GENAI_AVAILABLE"
]
