"""
Semantic processing pipeline for Layer D operations.

This module provides the SemanticProcessingPipeline that orchestrates
embedding generation, candidate processing, and vector storage operations.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, UTC
from uuid import uuid4

from ..models.taxonomy import SemanticCandidate
from ..semantic import (
    SemanticEmbedding, EmbeddingProvider, SemanticRepository,
    TextProcessor, SemanticEnrichmentEngine
)
from ..storage.interfaces import SemanticVectorRepository, StorageError
from .embedding_providers import EmbeddingProviderFactory

logger = logging.getLogger(__name__)


class SemanticProcessingPipeline:
    """
    Pipeline for processing semantic candidates into vector storage.
    
    Coordinates between embedding generation, candidate processing,
    and storage across DuckDB and Qdrant backends.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_repository: Optional[SemanticVectorRepository] = None,
        batch_size: int = 50,
        max_retries: int = 3
    ):
        self.embedding_provider = embedding_provider
        self.vector_repository = vector_repository
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.text_processor = TextProcessor()
        
        # Statistics tracking
        self.stats = {
            "candidates_processed": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    async def initialize(self) -> None:
        """Initialize the pipeline and embedding provider."""
        await self.embedding_provider.initialize()
        logger.info("Semantic processing pipeline initialized")
    
    async def process_semantic_candidates(
        self, 
        candidates: List[SemanticCandidate]
    ) -> List[str]:
        """
        Process semantic candidates into vector storage.
        
        Args:
            candidates: List of semantic candidates to process
            
        Returns:
            List of embedding IDs that were successfully stored
        """
        if not candidates:
            return []
        
        self.stats["start_time"] = datetime.now(UTC)
        self.stats["candidates_processed"] = len(candidates)
        
        logger.info(f"Processing {len(candidates)} semantic candidates")
        
        try:
            # Process in batches
            all_embedding_ids = []
            
            for i in range(0, len(candidates), self.batch_size):
                batch = candidates[i:i + self.batch_size]
                batch_ids = await self._process_batch(batch)
                all_embedding_ids.extend(batch_ids)
                
                logger.debug(f"Processed batch {i//self.batch_size + 1}/{(len(candidates)-1)//self.batch_size + 1}")
            
            self.stats["end_time"] = datetime.now(UTC)
            self._log_statistics()
            
            return all_embedding_ids
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            self.stats["errors"] += 1
            raise StorageError(f"Failed to process candidates: {e}")
    
    async def _process_batch(self, candidates: List[SemanticCandidate]) -> List[str]:
        """Process a batch of candidates."""
        if not candidates:
            return []
        
        try:
            # Prepare texts for embedding generation
            texts = []
            for candidate in candidates:
                # Combine term with context for richer embeddings
                text = self._prepare_candidate_text(candidate)
                texts.append(text)
            
            # Generate embeddings
            embeddings_vectors = await self.embedding_provider.generate_batch_embeddings(
                texts, batch_size=self.batch_size
            )
            
            self.stats["embeddings_generated"] += len(embeddings_vectors)
            
            # Create SemanticEmbedding objects
            embeddings = []
            for candidate, vector in zip(candidates, embeddings_vectors):
                embedding = self._create_embedding_from_candidate(candidate, vector)
                embeddings.append(embedding)
            
            # Store embeddings if vector repository is available
            stored_ids = []
            if self.vector_repository:
                try:
                    stored_ids = await self.vector_repository.batch_store_embeddings(embeddings)
                    self.stats["embeddings_stored"] += len(stored_ids)
                    logger.debug(f"Stored {len(stored_ids)} embeddings in vector storage")
                except Exception as e:
                    logger.error(f"Failed to store batch in vector storage: {e}")
                    self.stats["errors"] += 1
            
            return stored_ids
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.stats["errors"] += 1
            return []
    
    def _prepare_candidate_text(self, candidate: SemanticCandidate) -> str:
        """Prepare candidate text for embedding generation."""
        # Start with the main term
        text_parts = [candidate.term]
        
        # Add normalized term if different
        if candidate.normalized_term and candidate.normalized_term != candidate.term:
            text_parts.append(candidate.normalized_term)
        
        # Add context samples if available
        if candidate.contexts:
            # Take up to 3 context samples
            contexts = candidate.contexts[:3] if isinstance(candidate.contexts, list) else [candidate.contexts]
            text_parts.extend(contexts)
        
        # Combine all parts
        combined_text = " | ".join(text_parts)
        
        # Clean the text
        return self.text_processor.clean_text_for_embedding(combined_text)
    
    def _create_embedding_from_candidate(
        self, 
        candidate: SemanticCandidate, 
        vector: List[float]
    ) -> SemanticEmbedding:
        """Create SemanticEmbedding from candidate and vector."""
        return SemanticEmbedding(
            embedding_id=f"emb_{candidate.candidate_id}_{uuid4().hex[:8]}",
            entity_id=candidate.candidate_id,
            entity_type="semantic_candidate",
            embedding_vector=vector,
            model_name=getattr(self.embedding_provider, 'model_name', 'default'),
            model_version="1.0",
            dimensions=len(vector),
            created_at=datetime.now(UTC),
            quality_score=candidate.score,
            is_active=candidate.status == "active",
            metadata={
                "term": candidate.term,
                "normalized_term": candidate.normalized_term,
                "frequency": candidate.frequency,
                "language": candidate.language,
                "domain": candidate.domain,
                "status": candidate.status,
                "cluster_id": candidate.cluster_id,
                "source_text": self._prepare_candidate_text(candidate)
            }
        )
    
    async def update_candidate_embeddings(
        self, 
        entity_ids: List[str],
        force_regenerate: bool = False
    ) -> int:
        """
        Re-generate embeddings for existing candidates.
        
        Args:
            entity_ids: List of candidate entity IDs to update
            force_regenerate: Whether to regenerate even if embedding exists
            
        Returns:
            Number of embeddings successfully updated
        """
        if not entity_ids:
            return 0
        
        logger.info(f"Updating embeddings for {len(entity_ids)} candidates")
        
        updated_count = 0
        
        try:
            # TODO: Retrieve candidates from storage (needs semantic repository implementation)
            # For now, log that this feature is not fully implemented
            logger.warning("Candidate embedding updates not fully implemented - needs semantic candidate repository")
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Failed to update candidate embeddings: {e}")
            raise StorageError(f"Failed to update embeddings: {e}")
    
    async def generate_embeddings_for_texts(
        self, 
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[SemanticEmbedding]:
        """
        Generate embeddings for arbitrary texts.
        
        Args:
            texts: List of texts to embed
            metadata_list: Optional metadata for each text
            
        Returns:
            List of SemanticEmbedding objects
        """
        if not texts:
            return []
        
        try:
            # Generate embeddings
            vectors = await self.embedding_provider.generate_batch_embeddings(texts)
            
            # Create embedding objects
            embeddings = []
            for i, (text, vector) in enumerate(zip(texts, vectors)):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                
                embedding = SemanticEmbedding(
                    embedding_id=f"emb_{uuid4().hex}",
                    entity_id=metadata.get("entity_id", f"text_{i}"),
                    entity_type=metadata.get("entity_type", "text"),
                    embedding_vector=vector,
                    model_name=getattr(self.embedding_provider, 'model_name', 'default'),
                    model_version="1.0",
                    dimensions=len(vector),
                    metadata={
                        "source_text": text,
                        **metadata
                    }
                )
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise StorageError(f"Failed to generate embeddings: {e}")
    
    async def find_similar_candidates(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float = 0.7,
        domains: Optional[List[str]] = None
    ) -> List[Tuple[SemanticCandidate, float]]:
        """
        Find semantically similar candidates using vector search.
        
        Args:
            query_text: Text to find similar candidates for
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            domains: Optional domain filter
            
        Returns:
            List of (candidate, similarity_score) tuples
        """
        if not self.vector_repository:
            logger.warning("Vector repository not available for similarity search")
            return []
        
        try:
            # Generate query embedding
            query_vector = await self.embedding_provider.generate_embedding(query_text)
            
            # Search for similar embeddings
            similar_embeddings = await self.vector_repository.find_similar_embeddings(
                query_vector=query_vector,
                limit=limit,
                threshold=threshold,
                entity_types=["semantic_candidate"],
                domains=domains,
                active_only=True
            )
            
            # Convert embeddings back to candidates (simplified)
            results = []
            for embedding, similarity in similar_embeddings:
                # Create candidate from embedding metadata
                candidate = self._create_candidate_from_embedding(embedding)
                results.append((candidate, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _create_candidate_from_embedding(self, embedding: SemanticEmbedding) -> SemanticCandidate:
        """Create SemanticCandidate from embedding metadata."""
        metadata = embedding.metadata
        
        return SemanticCandidate(
            candidate_id=embedding.entity_id,
            term=metadata.get("term", ""),
            normalized_term=metadata.get("normalized_term", ""),
            frequency=metadata.get("frequency", 1),
            contexts=metadata.get("contexts", []),
            cluster_id=metadata.get("cluster_id"),
            score=metadata.get("score", embedding.quality_score),
            language=metadata.get("language", "auto"),
            domain=metadata.get("domain", "unknown"),
            status=metadata.get("status", "active")
        )
    
    async def calculate_similarity_matrix(
        self, 
        candidates: List[SemanticCandidate]
    ) -> List[List[float]]:
        """
        Calculate similarity matrix for a list of candidates.
        
        Useful for clustering and duplicate detection.
        """
        if not candidates:
            return []
        
        try:
            # Generate embeddings for all candidates
            texts = [self._prepare_candidate_text(c) for c in candidates]
            embeddings = await self.embedding_provider.generate_batch_embeddings(texts)
            
            # Calculate pairwise similarities
            n = len(embeddings)
            similarity_matrix = [[0.0] * n for _ in range(n)]
            
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        similarity = await self.embedding_provider.calculate_similarity(
                            embeddings[i], embeddings[j]
                        )
                        similarity_matrix[i][j] = similarity
                        similarity_matrix[j][i] = similarity  # Symmetric
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity matrix: {e}")
            return []
    
    def _log_statistics(self) -> None:
        """Log processing statistics."""
        if self.stats["start_time"] and self.stats["end_time"]:
            duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
            
            logger.info(
                f"Pipeline completed in {duration:.2f}s: "
                f"{self.stats['candidates_processed']} candidates processed, "
                f"{self.stats['embeddings_generated']} embeddings generated, "
                f"{self.stats['embeddings_stored']} stored in vector storage, "
                f"{self.stats['errors']} errors"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        stats = self.stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            stats["duration_seconds"] = (
                stats["end_time"] - stats["start_time"]
            ).total_seconds()
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            "candidates_processed": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }


class SemanticPipelineFactory:
    """Factory for creating semantic processing pipelines."""
    
    @staticmethod
    async def create_pipeline(
        config: Dict[str, Any],
        vector_repository: Optional[SemanticVectorRepository] = None
    ) -> SemanticProcessingPipeline:
        """
        Create and initialize a semantic processing pipeline.
        
        Args:
            config: Pipeline configuration
            vector_repository: Optional vector storage repository
            
        Returns:
            Initialized pipeline
        """
        # Create embedding provider
        embedding_config = config.get("embedding", {})
        provider_type = embedding_config.get("type", "sentence_transformers")
        provider_config = embedding_config.get("config", {})
        
        provider = EmbeddingProviderFactory.create_provider(provider_type, provider_config)
        
        # Create pipeline
        pipeline = SemanticProcessingPipeline(
            embedding_provider=provider,
            vector_repository=vector_repository,
            batch_size=config.get("batch_size", 50),
            max_retries=config.get("max_retries", 3)
        )
        
        # Initialize
        await pipeline.initialize()
        
        return pipeline
    
    @staticmethod
    async def create_default_pipeline(
        vector_repository: Optional[SemanticVectorRepository] = None
    ) -> SemanticProcessingPipeline:
        """Create pipeline with default configuration."""
        provider = EmbeddingProviderFactory.create_default_provider()
        
        pipeline = SemanticProcessingPipeline(
            embedding_provider=provider,
            vector_repository=vector_repository
        )
        
        await pipeline.initialize()
        return pipeline
