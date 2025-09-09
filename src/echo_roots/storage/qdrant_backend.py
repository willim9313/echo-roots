"""
Qdrant vector database backend implementation.

This module provides Qdrant-based vector storage for Layer D semantic
candidates in the echo-roots taxonomy system. Qdrant serves as the
specialized backend for semantic similarity search and vector operations.

Features:
- High-performance vector similarity search
- Semantic embedding storage and retrieval
- Batch operations for efficient processing
- Configurable distance metrics (Cosine, Euclidean, etc.)
- Async operations with connection pooling
"""

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import uuid4

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, CollectionStatus,
    PointStruct, SearchRequest, Filter, FieldCondition,
    MatchValue, UpdateStatus, CollectionInfo
)
from qdrant_client.http.exceptions import ResponseHandlingException
import numpy as np

from .interfaces import StorageBackend, StorageError, ConnectionError, NotFoundError
from ..models.taxonomy import SemanticCandidate
from ..semantic import SemanticEmbedding


logger = logging.getLogger(__name__)


class QdrantConnectionConfig:
    """Configuration for Qdrant connection."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        https: bool = False,
        timeout: float = 30.0,
        collection_name: str = "semantic_candidates",
        vector_size: int = 768,
        distance: str = "Cosine"
    ):
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.api_key = api_key
        self.https = https
        self.timeout = timeout
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = self._parse_distance(distance)
    
    def _parse_distance(self, distance_str: str) -> Distance:
        """Parse distance metric string to Qdrant Distance enum."""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "manhattan": Distance.MANHATTAN,
            "dot": Distance.DOT
        }
        return distance_map.get(distance_str.lower(), Distance.COSINE)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QdrantConnectionConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


class QdrantBackend(StorageBackend):
    """Qdrant vector database backend for Layer D semantic storage."""
    
    def __init__(self, config: QdrantConnectionConfig):
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Qdrant connection and create collection."""
        try:
            # Create async client
            self.client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                api_key=self.config.api_key,
                https=self.config.https,
                timeout=self.config.timeout
            )
            
            # Test connection
            await self._test_connection()
            
            # Create collection if it doesn't exist
            await self._ensure_collection_exists()
            
            self._initialized = True
            logger.info(f"Qdrant backend initialized: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant backend: {e}")
            raise ConnectionError(f"Failed to initialize Qdrant: {e}")
    
    async def _test_connection(self) -> None:
        """Test Qdrant connection."""
        try:
            collections = await self.client.get_collections()
            logger.debug(f"Qdrant connection test successful, found {len(collections.collections)} collections")
        except Exception as e:
            raise ConnectionError(f"Qdrant connection test failed: {e}")
    
    async def _ensure_collection_exists(self) -> None:
        """Ensure the semantic candidates collection exists."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.config.collection_name not in collection_names:
                # Create collection
                await self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=self.config.distance
                    )
                )
                logger.info(f"Created Qdrant collection: {self.config.collection_name}")
            else:
                logger.debug(f"Qdrant collection already exists: {self.config.collection_name}")
                
        except Exception as e:
            raise StorageError(f"Failed to ensure collection exists: {e}")
    
    async def store_embedding(self, embedding: SemanticEmbedding) -> str:
        """Store semantic embedding in Qdrant."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        try:
            # Prepare point data
            point = PointStruct(
                id=embedding.embedding_id,
                vector=embedding.embedding_vector,
                payload={
                    "entity_id": embedding.entity_id,
                    "entity_type": embedding.entity_type,
                    "model_name": embedding.model_name,
                    "model_version": embedding.model_version,
                    "dimensions": embedding.dimensions,
                    "created_at": embedding.created_at.isoformat(),
                    "quality_score": embedding.quality_score,
                    "is_active": embedding.is_active,
                    **embedding.metadata
                }
            )
            
            # Store point
            result = await self.client.upsert(
                collection_name=self.config.collection_name,
                points=[point]
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.debug(f"Stored embedding: {embedding.embedding_id}")
                return embedding.embedding_id
            else:
                raise StorageError(f"Failed to store embedding: {result}")
                
        except Exception as e:
            logger.error(f"Error storing embedding {embedding.embedding_id}: {e}")
            raise StorageError(f"Failed to store embedding: {e}")
    
    async def batch_upsert(self, embeddings: List[SemanticEmbedding]) -> List[str]:
        """Batch upsert multiple embeddings."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        if not embeddings:
            return []
        
        try:
            # Prepare points
            points = []
            for embedding in embeddings:
                point = PointStruct(
                    id=embedding.embedding_id,
                    vector=embedding.embedding_vector,
                    payload={
                        "entity_id": embedding.entity_id,
                        "entity_type": embedding.entity_type,
                        "model_name": embedding.model_name,
                        "model_version": embedding.model_version,
                        "dimensions": embedding.dimensions,
                        "created_at": embedding.created_at.isoformat(),
                        "quality_score": embedding.quality_score,
                        "is_active": embedding.is_active,
                        **embedding.metadata
                    }
                )
                points.append(point)
            
            # Batch upsert
            result = await self.client.upsert(
                collection_name=self.config.collection_name,
                points=points
            )
            
            if result.status == UpdateStatus.COMPLETED:
                embedding_ids = [emb.embedding_id for emb in embeddings]
                logger.info(f"Batch upserted {len(embedding_ids)} embeddings")
                return embedding_ids
            else:
                raise StorageError(f"Batch upsert failed: {result}")
                
        except Exception as e:
            logger.error(f"Error in batch upsert: {e}")
            raise StorageError(f"Batch upsert failed: {e}")
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SemanticEmbedding, float]]:
        """Search for similar embeddings using vector similarity."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        try:
            # Prepare search filters
            search_filter = None
            if filters:
                conditions = []
                
                # Add filter conditions
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values - use match any
                        for v in value:
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=v))
                            )
                    else:
                        # Single value
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Perform search
            search_results = await self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=True
            )
            
            # Convert results to SemanticEmbedding objects
            results = []
            for scored_point in search_results:
                # Reconstruct SemanticEmbedding from payload and vector
                payload = scored_point.payload
                
                embedding = SemanticEmbedding(
                    embedding_id=str(scored_point.id),
                    entity_id=payload["entity_id"],
                    entity_type=payload["entity_type"],
                    embedding_vector=scored_point.vector,
                    model_name=payload["model_name"],
                    model_version=payload["model_version"],
                    dimensions=payload["dimensions"],
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    quality_score=payload.get("quality_score", 1.0),
                    is_active=payload.get("is_active", True),
                    metadata={k: v for k, v in payload.items() 
                             if k not in ["entity_id", "entity_type", "model_name", 
                                         "model_version", "dimensions", "created_at", 
                                         "quality_score", "is_active"]}
                )
                
                results.append((embedding, scored_point.score))
            
            logger.debug(f"Found {len(results)} similar embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise StorageError(f"Similarity search failed: {e}")
    
    async def get_embedding(self, embedding_id: str) -> Optional[SemanticEmbedding]:
        """Retrieve specific embedding by ID."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        try:
            # Retrieve point
            points = await self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[embedding_id],
                with_payload=True,
                with_vectors=True
            )
            
            if not points:
                return None
            
            point = points[0]
            payload = point.payload
            
            # Reconstruct SemanticEmbedding
            embedding = SemanticEmbedding(
                embedding_id=str(point.id),
                entity_id=payload["entity_id"],
                entity_type=payload["entity_type"],
                embedding_vector=point.vector,
                model_name=payload["model_name"],
                model_version=payload["model_version"],
                dimensions=payload["dimensions"],
                created_at=datetime.fromisoformat(payload["created_at"]),
                quality_score=payload.get("quality_score", 1.0),
                is_active=payload.get("is_active", True),
                metadata={k: v for k, v in payload.items() 
                         if k not in ["entity_id", "entity_type", "model_name", 
                                     "model_version", "dimensions", "created_at", 
                                     "quality_score", "is_active"]}
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error retrieving embedding {embedding_id}: {e}")
            raise StorageError(f"Failed to retrieve embedding: {e}")
    
    async def delete_embedding(self, embedding_id: str) -> bool:
        """Delete embedding by ID."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        try:
            result = await self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=[embedding_id]
            )
            
            success = result.status == UpdateStatus.COMPLETED
            if success:
                logger.debug(f"Deleted embedding: {embedding_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting embedding {embedding_id}: {e}")
            raise StorageError(f"Failed to delete embedding: {e}")
    
    async def update_embedding_metadata(
        self, 
        embedding_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update embedding payload metadata."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        try:
            # Set payload (updates existing fields, adds new ones)
            result = await self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=metadata,
                points=[embedding_id]
            )
            
            success = result.status == UpdateStatus.COMPLETED
            if success:
                logger.debug(f"Updated metadata for embedding: {embedding_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating metadata for {embedding_id}: {e}")
            raise StorageError(f"Failed to update metadata: {e}")
    
    async def count_embeddings(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count embeddings with optional filters."""
        if not self._initialized or not self.client:
            raise ConnectionError("Qdrant backend not initialized")
        
        try:
            # Prepare filters
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    search_filter = Filter(must=conditions)
            
            # Count points
            result = await self.client.count(
                collection_name=self.config.collection_name,
                count_filter=search_filter
            )
            
            return result.count
            
        except Exception as e:
            logger.error(f"Error counting embeddings: {e}")
            raise StorageError(f"Failed to count embeddings: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant backend health."""
        if not self._initialized or not self.client:
            return {"status": "unhealthy", "error": "Not initialized"}
        
        try:
            # Test connection
            collections = await self.client.get_collections()
            
            # Get collection info
            collection_info = None
            if self.config.collection_name in [col.name for col in collections.collections]:
                collection_info = await self.client.get_collection(self.config.collection_name)
            
            # Count embeddings
            embedding_count = await self.count_embeddings()
            
            return {
                "status": "healthy",
                "backend": "QdrantBackend",
                "initialized": self._initialized,
                "host": f"{self.config.host}:{self.config.port}",
                "collection_name": self.config.collection_name,
                "collection_exists": collection_info is not None,
                "embedding_count": embedding_count,
                "vector_size": self.config.vector_size,
                "distance_metric": self.config.distance.name,
                "total_collections": len(collections.collections)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "QdrantBackend", 
                "error": str(e)
            }
    
    async def close(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            await self.client.close()
            self.client = None
            self._initialized = False
            logger.info("Qdrant connection closed")


class QdrantSemanticRepository:
    """Repository for semantic operations using Qdrant backend."""
    
    def __init__(self, backend: QdrantBackend):
        self.backend = backend
    
    async def store_semantic_embedding(self, embedding: SemanticEmbedding) -> str:
        """Store semantic embedding."""
        return await self.backend.store_embedding(embedding)
    
    async def find_similar_embeddings(
        self,
        query_vector: List[float],
        limit: int = 10,
        threshold: float = 0.5,
        entity_types: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        active_only: bool = True
    ) -> List[Tuple[SemanticEmbedding, float]]:
        """Find semantically similar embeddings."""
        # Prepare filters
        filters = {}
        
        if entity_types:
            filters["entity_type"] = entity_types
        
        if domains:
            filters["domain"] = domains
            
        if active_only:
            filters["is_active"] = True
        
        return await self.backend.search_similar(
            query_vector=query_vector,
            limit=limit,
            threshold=threshold,
            filters=filters if filters else None
        )
    
    async def get_embedding_by_entity(self, entity_id: str) -> Optional[SemanticEmbedding]:
        """Get embedding for specific entity."""
        # Search by entity_id in payload
        results = await self.backend.search_similar(
            query_vector=[0.0] * self.backend.config.vector_size,  # Dummy vector
            limit=1,
            threshold=0.0,  # Get any match
            filters={"entity_id": entity_id}
        )
        
        if results:
            return results[0][0]  # Return the embedding (without score)
        
        return None
    
    async def batch_store_embeddings(self, embeddings: List[SemanticEmbedding]) -> List[str]:
        """Batch store multiple embeddings."""
        return await self.backend.batch_upsert(embeddings)
    
    async def update_embedding_metadata(
        self, 
        entity_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for embedding by entity ID."""
        # First find the embedding ID by entity ID
        embedding = await self.get_embedding_by_entity(entity_id)
        if not embedding:
            raise NotFoundError(f"No embedding found for entity: {entity_id}")
        
        return await self.backend.update_embedding_metadata(
            embedding.embedding_id, metadata
        )
    
    async def get_embeddings_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        total_count = await self.backend.count_embeddings()
        active_count = await self.backend.count_embeddings({"is_active": True})
        inactive_count = total_count - active_count
        
        return {
            "total_embeddings": total_count,
            "active_embeddings": active_count,
            "inactive_embeddings": inactive_count,
            "collection_name": self.backend.config.collection_name,
            "vector_size": self.backend.config.vector_size
        }


# Factory functions
async def create_qdrant_backend(config: Dict[str, Any]) -> QdrantBackend:
    """Create and initialize Qdrant backend."""
    qdrant_config = QdrantConnectionConfig.from_dict(config)
    backend = QdrantBackend(qdrant_config)
    await backend.initialize()
    return backend


async def create_qdrant_repository(config: Dict[str, Any]) -> QdrantSemanticRepository:
    """Create Qdrant semantic repository."""
    backend = await create_qdrant_backend(config)
    return QdrantSemanticRepository(backend)
