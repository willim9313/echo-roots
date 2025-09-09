"""
Embedding providers for semantic vector generation.

This module provides various embedding providers that generate vector
representations for text content in the semantic processing pipeline.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np

from ..semantic import EmbeddingProvider, EmbeddingModel

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._dimensions = None
        
    async def initialize(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load model in executor to avoid blocking
            self.model = await asyncio.get_event_loop().run_in_executor(
                None, lambda: SentenceTransformer(self.model_name, device=self.device)
            )
            
            # Get dimensions
            self._dimensions = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"SentenceTransformers model loaded: {self.model_name} ({self._dimensions}d)")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: "
                "pip install sentence-transformers"
            )
    
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embedding for given text."""
        if not self.model:
            await self.initialize()
        
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimensions
        
        try:
            # Generate embedding in executor
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.encode([text], show_progress_bar=False)[0]
            )
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector on error
            return [0.0] * self._dimensions
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        model: str = None,
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if not self.model:
            await self.initialize()
        
        if not texts:
            return []
        
        # Filter out empty texts and keep track of original indices
        filtered_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if text.strip():
                filtered_texts.append(text)
                original_indices.append(i)
        
        if not filtered_texts:
            # All texts are empty
            return [[0.0] * self._dimensions] * len(texts)
        
        try:
            # Process in batches to avoid memory issues
            all_embeddings = []
            
            for i in range(0, len(filtered_texts), batch_size):
                batch_texts = filtered_texts[i:i + batch_size]
                
                # Generate embeddings in executor
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.model.encode(batch_texts, show_progress_bar=False)
                )
                
                all_embeddings.extend(batch_embeddings.tolist())
            
            # Reconstruct full results with zero vectors for empty texts
            results = []
            embedding_idx = 0
            
            for i in range(len(texts)):
                if i in original_indices:
                    results.append(all_embeddings[embedding_idx])
                    embedding_idx += 1
                else:
                    results.append([0.0] * self._dimensions)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors on error
            return [[0.0] * self._dimensions] * len(texts)
    
    def get_embedding_dimensions(self, model: str = None) -> int:
        """Get dimensions for specific model."""
        return self._dimensions if self._dimensions else 384  # Default for MiniLM
    
    async def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in valid range
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding API provider."""
    
    def __init__(
        self, 
        api_key: str,
        model_name: str = "text-embedding-3-small",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = None
        
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
            
            logger.info(f"OpenAI embedding client initialized: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "OpenAI client not installed. Install with: pip install openai"
            )
    
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate embedding for given text."""
        if not self.client:
            await self.initialize()
        
        if not text.strip():
            # Return zero vector for empty text
            dimensions = self.get_embedding_dimensions(model)
            return [0.0] * dimensions
        
        effective_model = model or self.model_name
        
        try:
            response = await self.client.embeddings.create(
                input=[text],
                model=effective_model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            dimensions = self.get_embedding_dimensions(model)
            return [0.0] * dimensions
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        model: str = None,
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for batch of texts."""
        if not self.client:
            await self.initialize()
        
        if not texts:
            return []
        
        effective_model = model or self.model_name
        dimensions = self.get_embedding_dimensions(model)
        
        # Filter empty texts
        non_empty_texts = [text if text.strip() else " " for text in texts]
        
        try:
            # Process in batches
            all_embeddings = []
            
            for i in range(0, len(non_empty_texts), batch_size):
                batch_texts = non_empty_texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    input=batch_texts,
                    model=effective_model
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI batch embeddings: {e}")
            return [[0.0] * dimensions] * len(texts)
    
    def get_embedding_dimensions(self, model: str = None) -> int:
        """Get dimensions for specific model."""
        effective_model = model or self.model_name
        
        # OpenAI model dimensions
        dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        return dimensions_map.get(effective_model, 1536)
    
    async def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


class MultiModalEmbeddingProvider(EmbeddingProvider):
    """Multi-model embedding aggregation provider."""
    
    def __init__(self, providers: List[EmbeddingProvider], weights: Optional[List[float]] = None):
        self.providers = providers
        self.weights = weights or [1.0] * len(providers)
        
        if len(self.weights) != len(self.providers):
            raise ValueError("Number of weights must match number of providers")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    async def initialize(self) -> None:
        """Initialize all providers."""
        for provider in self.providers:
            await provider.initialize()
        
        logger.info(f"MultiModal provider initialized with {len(self.providers)} providers")
    
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate weighted average embedding from all providers."""
        if not text.strip():
            dimensions = self.get_embedding_dimensions(model)
            return [0.0] * dimensions
        
        embeddings = []
        valid_weights = []
        
        for i, provider in enumerate(self.providers):
            try:
                embedding = await provider.generate_embedding(text, model)
                embeddings.append(np.array(embedding))
                valid_weights.append(self.weights[i])
            except Exception as e:
                logger.warning(f"Provider {i} failed: {e}")
                continue
        
        if not embeddings:
            dimensions = self.get_embedding_dimensions(model)
            return [0.0] * dimensions
        
        # Normalize valid weights
        total_valid_weight = sum(valid_weights)
        normalized_weights = [w / total_valid_weight for w in valid_weights]
        
        # Calculate weighted average
        result = np.zeros_like(embeddings[0])
        for embedding, weight in zip(embeddings, normalized_weights):
            result += embedding * weight
        
        return result.tolist()
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        model: str = None,
        batch_size: int = 32
    ) -> List[List[float]]:
        """Generate batch embeddings using all providers."""
        if not texts:
            return []
        
        all_embeddings = []
        valid_weights = []
        
        for i, provider in enumerate(self.providers):
            try:
                embeddings = await provider.generate_batch_embeddings(texts, model, batch_size)
                all_embeddings.append(embeddings)
                valid_weights.append(self.weights[i])
            except Exception as e:
                logger.warning(f"Provider {i} failed for batch: {e}")
                continue
        
        if not all_embeddings:
            dimensions = self.get_embedding_dimensions(model)
            return [[0.0] * dimensions] * len(texts)
        
        # Normalize weights
        total_valid_weight = sum(valid_weights)
        normalized_weights = [w / total_valid_weight for w in valid_weights]
        
        # Calculate weighted average for each text
        results = []
        for text_idx in range(len(texts)):
            text_embeddings = [emb_list[text_idx] for emb_list in all_embeddings]
            
            result = np.zeros(len(text_embeddings[0]))
            for embedding, weight in zip(text_embeddings, normalized_weights):
                result += np.array(embedding) * weight
            
            results.append(result.tolist())
        
        return results
    
    def get_embedding_dimensions(self, model: str = None) -> int:
        """Get dimensions from first provider."""
        if self.providers:
            return self.providers[0].get_embedding_dimensions(model)
        return 384  # Default
    
    async def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0


class EmbeddingProviderFactory:
    """Factory for creating embedding providers."""
    
    @staticmethod
    def create_provider(
        provider_type: str,
        config: Dict[str, Any]
    ) -> EmbeddingProvider:
        """Create embedding provider based on type and config."""
        
        if provider_type == "sentence_transformers":
            return SentenceTransformersProvider(
                model_name=config.get("model_name", "all-MiniLM-L6-v2"),
                device=config.get("device", "cpu")
            )
        
        elif provider_type == "openai":
            return OpenAIEmbeddingProvider(
                api_key=config["api_key"],
                model_name=config.get("model_name", "text-embedding-3-small"),
                max_retries=config.get("max_retries", 3),
                timeout=config.get("timeout", 30.0)
            )
        
        elif provider_type == "multimodal":
            providers = []
            for provider_config in config["providers"]:
                provider = EmbeddingProviderFactory.create_provider(
                    provider_config["type"],
                    provider_config["config"]
                )
                providers.append(provider)
            
            return MultiModalEmbeddingProvider(
                providers=providers,
                weights=config.get("weights")
            )
        
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def create_default_provider() -> EmbeddingProvider:
        """Create default sentence-transformers provider."""
        return SentenceTransformersProvider()
