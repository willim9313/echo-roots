"""
Hybrid storage manager combining DuckDB, Neo4j, and Qdrant.

This module provides the HybridStorageManager that coordinates between
multiple storage backends according to ADR-0001:
- DuckDB: Core ingestion, extraction results, analytics
- Neo4j: Taxonomy trees, controlled vocabulary, semantic relationships  
- Qdrant: Vector storage for Layer D semantic candidates
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

from .interfaces import (
    StorageManager, TaxonomyRepository, SemanticVectorRepository, 
    StorageError, NotFoundError
)
from .duckdb_backend import DuckDBStorageManager, DuckDBBackend
from .neo4j_backend import Neo4jBackend, Neo4jTaxonomyRepository
from ..models.taxonomy import Category, Attribute, SemanticCandidate
from ..semantic import SemanticSearchResult

# Optional Qdrant imports
try:
    from .qdrant_backend import (
        QdrantBackend, QdrantSemanticRepository, QdrantConnectionConfig,
        create_qdrant_backend
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


logger = logging.getLogger(__name__)


class HybridStorageManager(StorageManager):
    """
    Hybrid storage manager using DuckDB + Neo4j + Qdrant.
    
    Coordinates between multiple specialized backends:
    - DuckDB: Core ingestion, extraction results, analytics
    - Neo4j: Taxonomy trees, controlled vocabulary, semantic relationships  
    - Qdrant: Vector storage for Layer D semantic candidates and similarity search
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize DuckDB backend (core data)
        duckdb_config = config.get("duckdb", {})
        self.duckdb_manager = DuckDBStorageManager({"duckdb": duckdb_config})
        
        # Initialize Neo4j backend (graph data) if configured
        self.neo4j_backend = None
        self._taxonomy = None
        
        neo4j_config = config.get("neo4j", {})
        if neo4j_config.get("enabled", False):
            self.neo4j_backend = Neo4jBackend(
                uri=neo4j_config["uri"],
                user=neo4j_config["user"], 
                password=neo4j_config["password"],
                database=neo4j_config.get("database", "neo4j")
            )
            self._taxonomy = Neo4jTaxonomyRepository(self.neo4j_backend)
        else:
            logger.info("Neo4j not configured, taxonomy will use DuckDB fallback")
        
        # Initialize Qdrant backend (vector data) if configured and available
        self.qdrant_backend = None
        self._semantic_vectors = None
        
        qdrant_config = config.get("qdrant", {})
        if QDRANT_AVAILABLE and qdrant_config.get("enabled", False):
            self.qdrant_config = QdrantConnectionConfig.from_dict(qdrant_config)
            # Will be initialized in async initialize() method
        else:
            if not QDRANT_AVAILABLE:
                logger.info("Qdrant client not available, install with: pip install qdrant-client")
            else:
                logger.info("Qdrant not configured, semantic search will use DuckDB fallback")
    
    async def initialize(self) -> None:
        """Initialize all storage backends."""
        # Always initialize DuckDB
        await self.duckdb_manager.initialize()
        
        # Initialize Neo4j if configured
        if self.neo4j_backend:
            await self.neo4j_backend.initialize()
        
        # Initialize Qdrant if configured and available
        if hasattr(self, 'qdrant_config'):
            try:
                self.qdrant_backend = QdrantBackend(self.qdrant_config)
                await self.qdrant_backend.initialize()
                self._semantic_vectors = QdrantSemanticRepository(self.qdrant_backend)
                logger.info("Qdrant backend initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant backend: {e}")
                self.qdrant_backend = None
                self._semantic_vectors = None
        
        # Log initialization summary
        backends = ["DuckDB"]
        if self.neo4j_backend:
            backends.append("Neo4j")
        if self.qdrant_backend:
            backends.append("Qdrant")
        
        logger.info(f"Hybrid storage initialized with: {', '.join(backends)}")
        self._initialized = True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all storage backends."""
        health = {
            "storage_manager": "HybridStorageManager",
            "initialized": self._initialized
        }
        
        # Check DuckDB health
        duckdb_health = await self.duckdb_manager.health_check()
        health["duckdb"] = duckdb_health
        
        # Check Neo4j health if configured
        if self.neo4j_backend:
            neo4j_health = await self.neo4j_backend.health_check()
            health["neo4j"] = neo4j_health
        
        # Check Qdrant health if configured
        if self.qdrant_backend:
            qdrant_health = await self.qdrant_backend.health_check()
            health["qdrant"] = qdrant_health
        else:
            health["qdrant"] = {"status": "disabled", "reason": "Not configured or unavailable"}
        
        # Overall status
        duckdb_ok = duckdb_health.get("status") == "healthy"
        neo4j_ok = not self.neo4j_backend or health["neo4j"].get("status") == "healthy"
        qdrant_ok = not self.qdrant_backend or health["qdrant"].get("status") == "healthy"
        
        health["overall_status"] = "healthy" if (duckdb_ok and neo4j_ok and qdrant_ok) else "degraded"
        
        return health
    
    async def close(self) -> None:
        """Close all storage connections."""
        await self.duckdb_manager.close()
        
        if self.neo4j_backend:
            await self.neo4j_backend.close()
        
        if self.qdrant_backend:
            await self.qdrant_backend.close()
        
        self._initialized = False
        logger.info("Hybrid storage manager closed")
    
    # Repository access methods - delegate to appropriate backends
    
    @property
    def ingestion(self):
        """Access to ingestion repository (DuckDB)."""
        return self.duckdb_manager.ingestion
    
    @property
    def extraction(self):
        """Access to extraction repository (DuckDB)."""
        return self.duckdb_manager.extraction
    
    @property
    def taxonomy(self) -> TaxonomyRepository:
        """Access to taxonomy repository (Neo4j preferred, DuckDB fallback)."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        
        if self._taxonomy:
            return self._taxonomy
        else:
            # Fallback to DuckDB implementation
            return self.duckdb_manager.taxonomy
    
    @property
    def mappings(self):
        """Access to mappings repository (DuckDB)."""
        return self.duckdb_manager.mappings
    
    @property
    def elevation(self):
        """Access to elevation repository (DuckDB)."""
        return self.duckdb_manager.elevation
    
    @property
    def analytics(self):
        """Access to analytics repository (DuckDB)."""
        return self.duckdb_manager.analytics
    
    @property
    def semantic_vectors(self) -> Optional[SemanticVectorRepository]:
        """Access to semantic vector repository (Qdrant if available, None otherwise)."""
        return self._semantic_vectors
    
    def has_vector_backend(self) -> bool:
        """Check if vector backend (Qdrant) is available."""
        return self.qdrant_backend is not None and self._semantic_vectors is not None
    
    # Hybrid-specific semantic operations
    
    async def store_semantic_candidate(self, candidate: SemanticCandidate) -> str:
        """
        Store semantic candidate in both DuckDB (metadata) and Qdrant (vector).
        
        This coordinates storage across both backends to ensure data consistency.
        """
        # Always store in DuckDB for metadata and fallback
        # TODO: Implement DuckDB semantic candidate storage when repository is available
        
        # Store in Qdrant if available for vector search
        if self.has_vector_backend() and hasattr(candidate, 'embedding_vector'):
            try:
                from ..semantic import SemanticEmbedding
                
                # Convert candidate to embedding
                embedding = SemanticEmbedding(
                    embedding_id=f"emb_{candidate.candidate_id}",
                    entity_id=candidate.candidate_id,
                    entity_type="semantic_candidate",
                    embedding_vector=candidate.embedding_vector,
                    model_name="default",  # TODO: Get from candidate metadata
                    model_version="1.0",
                    dimensions=len(candidate.embedding_vector),
                    metadata={
                        "term": candidate.term,
                        "normalized_term": candidate.normalized_term,
                        "frequency": candidate.frequency,
                        "language": candidate.language,
                        "domain": candidate.domain,
                        "status": candidate.status,
                        "score": candidate.score
                    }
                )
                
                await self._semantic_vectors.store_semantic_embedding(embedding)
                logger.debug(f"Stored semantic candidate in Qdrant: {candidate.candidate_id}")
                
            except Exception as e:
                logger.warning(f"Failed to store candidate in Qdrant: {e}")
        
        return candidate.candidate_id
    
    async def semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        threshold: float = 0.5,
        domains: Optional[List[str]] = None
    ) -> List[SemanticSearchResult]:
        """
        Unified semantic search across available backends.
        
        Uses Qdrant for vector search if available, falls back to DuckDB text search.
        """
        if self.has_vector_backend():
            try:
                # Use Qdrant for semantic search
                return await self._qdrant_semantic_search(query, limit, threshold, domains)
            except Exception as e:
                logger.warning(f"Qdrant search failed, falling back to DuckDB: {e}")
        
        # Fallback to DuckDB text search
        return await self._duckdb_semantic_search(query, limit, domains)
    
    async def _qdrant_semantic_search(
        self,
        query: str,
        limit: int,
        threshold: float,
        domains: Optional[List[str]] = None
    ) -> List[SemanticSearchResult]:
        """Perform semantic search using Qdrant vector similarity."""
        # Generate query embedding (would need embedding provider)
        # For now, return empty list as placeholder
        logger.warning("Qdrant semantic search not fully implemented - needs embedding provider")
        return []
    
    async def _duckdb_semantic_search(
        self,
        query: str,
        limit: int,
        domains: Optional[List[str]] = None
    ) -> List[SemanticSearchResult]:
        """Fallback text search using DuckDB."""
        # Simple text matching in DuckDB
        # TODO: Implement when DuckDB semantic candidate repository is available
        logger.warning("DuckDB semantic search not implemented - needs semantic repository")
        return []
    
    async def sync_semantic_candidates_to_qdrant(self, domain: str = None) -> Dict[str, Any]:
        """
        Sync semantic candidates from DuckDB to Qdrant.
        
        This is useful for migrating existing data or ensuring consistency.
        """
        if not self.has_vector_backend():
            raise ValueError("Qdrant backend not configured")
        
        logger.info(f"Starting semantic candidates sync to Qdrant for domain: {domain or 'all'}")
        
        stats = {
            "candidates_synced": 0,
            "embeddings_created": 0,
            "errors": []
        }
        
        try:
            # TODO: Implement actual sync logic when DuckDB semantic repo is complete
            logger.warning("Semantic candidates sync not yet implemented")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to sync candidates to Qdrant: {e}")
            stats["errors"].append(str(e))
            return stats
    
    # Hybrid-specific operations
    
    async def sync_taxonomy_to_neo4j(self, domain: str = None) -> Dict[str, Any]:
        """
        Sync taxonomy data from DuckDB to Neo4j.
        
        This is useful for migrating existing data or ensuring consistency
        between the two backends.
        """
        if not self.neo4j_backend or not self._taxonomy:
            raise ValueError("Neo4j backend not configured")
        
        logger.info(f"Starting taxonomy sync to Neo4j for domain: {domain or 'all'}")
        
        stats = {
            "categories_synced": 0,
            "attributes_synced": 0,
            "errors": []
        }
        
        try:
            # Get categories from DuckDB
            # Note: This assumes we have a DuckDB taxonomy implementation
            # For now, we'll create a placeholder sync
            
            # TODO: Implement actual sync logic when DuckDB taxonomy repo is complete
            logger.warning("Taxonomy sync not yet implemented - requires DuckDB taxonomy repository")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to sync taxonomy to Neo4j: {e}")
            stats["errors"].append(str(e))
            return stats
    
    async def get_taxonomy_storage_info(self) -> Dict[str, Any]:
        """Get information about taxonomy storage configuration."""
        return {
            "taxonomy_backend": "Neo4j" if self._taxonomy else "DuckDB",
            "neo4j_configured": self.neo4j_backend is not None,
            "neo4j_connected": self.neo4j_backend._initialized if self.neo4j_backend else False,
            "supports_graph_queries": self._taxonomy is not None,
            "supports_hierarchy_navigation": True,
            "supports_semantic_relationships": self._taxonomy is not None
        }
    
    async def get_semantic_storage_info(self) -> Dict[str, Any]:
        """Get information about semantic storage configuration."""
        return {
            "semantic_backend": "Qdrant" if self.has_vector_backend() else "DuckDB",
            "qdrant_configured": hasattr(self, 'qdrant_config'),
            "qdrant_connected": self.qdrant_backend._initialized if self.qdrant_backend else False,
            "qdrant_available": QDRANT_AVAILABLE,
            "supports_vector_search": self.has_vector_backend(),
            "supports_similarity_queries": self.has_vector_backend(),
            "supports_semantic_clustering": self.has_vector_backend(),
            "fallback_to_text_search": True
        }


# Factory functions for creating storage managers

async def create_hybrid_storage(config: Dict[str, Any] = None) -> HybridStorageManager:
    """
    Create and initialize a hybrid storage manager.
    
    Args:
        config: Configuration dict with 'duckdb' and optional 'neo4j' sections
    
    Returns:
        Initialized hybrid storage manager
    """
    if config is None:
        config = {
            "duckdb": {"database_path": None},  # In-memory
            "neo4j": {"enabled": False}
        }
    
    storage = HybridStorageManager(config)
    await storage.initialize()
    return storage


async def create_duckdb_only_storage(database_path: str = None) -> DuckDBStorageManager:
    """
    Create a DuckDB-only storage manager for testing or simple deployments.
    
    Args:
        database_path: Path to DuckDB file (None for in-memory)
    
    Returns:
        Initialized DuckDB storage manager
    """
    config = {"duckdb": {"database_path": database_path}}
    storage = DuckDBStorageManager(config)
    await storage.initialize()
    return storage
