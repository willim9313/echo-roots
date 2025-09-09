"""
Hybrid storage manager combining DuckDB and Neo4j.

This module provides the HybridStorageManager that coordinates between
DuckDB (for ingestion and analytics) and Neo4j (for taxonomy and relationships)
according to ADR-0001.
"""

import logging
from typing import Dict, Any, Optional

from .interfaces import StorageManager, TaxonomyRepository, StorageError
from .duckdb_backend import DuckDBStorageManager, DuckDBBackend
from .neo4j_backend import Neo4jBackend, Neo4jTaxonomyRepository
from ..models.taxonomy import Category, Attribute, SemanticCandidate


logger = logging.getLogger(__name__)


class HybridStorageManager(StorageManager):
    """
    Hybrid storage manager using DuckDB + Neo4j.
    
    Coordinates between:
    - DuckDB: Core ingestion, extraction results, analytics
    - Neo4j: Taxonomy trees, controlled vocabulary, semantic relationships
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
    
    async def initialize(self) -> None:
        """Initialize all storage backends."""
        # Always initialize DuckDB
        await self.duckdb_manager.initialize()
        
        # Initialize Neo4j if configured
        if self.neo4j_backend:
            await self.neo4j_backend.initialize()
            logger.info("Hybrid storage initialized with DuckDB + Neo4j")
        else:
            logger.info("Hybrid storage initialized with DuckDB only")
        
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
        else:
            health["neo4j"] = {"status": "disabled", "reason": "Not configured"}
        
        # Overall status
        duckdb_ok = duckdb_health.get("status") == "healthy"
        neo4j_ok = not self.neo4j_backend or health["neo4j"].get("status") == "healthy"
        
        health["overall_status"] = "healthy" if (duckdb_ok and neo4j_ok) else "degraded"
        
        return health
    
    async def close(self) -> None:
        """Close all storage connections."""
        await self.duckdb_manager.close()
        
        if self.neo4j_backend:
            await self.neo4j_backend.close()
        
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
