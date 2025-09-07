"""
Storage layer for echo-roots taxonomy system.

This package provides a flexible storage abstraction with pluggable
backends for different storage technologies. The primary backend is
DuckDB for analytics and ingestion workloads.

Usage:
    from echo_roots.storage import create_storage, query_ingestion
    
    # Create storage manager
    storage = await create_storage()
    
    # Store and query data
    item_id = await storage.ingestion.store_item(item)
    items = await query_ingestion().filter_by_domain("ecommerce").execute(storage)
"""

from .interfaces import (
    StorageManager, StorageBackend, TransactionContext,
    IngestionRepository, ExtractionRepository, TaxonomyRepository,
    MappingRepository, ElevationRepository, AnalyticsRepository,
    StorageError, ConnectionError, IntegrityError, NotFoundError, ConflictError
)

from .duckdb_backend import (
    DuckDBBackend, DuckDBStorageManager, DuckDBTransaction
)

from .migrations import (
    Migration, MigrationManager, initialize_database, get_migration_manager
)

from .repository import (
    RepositoryCoordinator, QueryBuilder, DataValidator, StorageFactory,
    create_storage, query_ingestion, query_extraction, validate_data
)

__all__ = [
    # Core interfaces
    "StorageManager",
    "StorageBackend", 
    "TransactionContext",
    
    # Repository interfaces
    "IngestionRepository",
    "ExtractionRepository", 
    "TaxonomyRepository",
    "MappingRepository",
    "ElevationRepository",
    "AnalyticsRepository",
    
    # Exceptions
    "StorageError",
    "ConnectionError",
    "IntegrityError", 
    "NotFoundError",
    "ConflictError",
    
    # DuckDB implementation
    "DuckDBBackend",
    "DuckDBStorageManager",
    "DuckDBTransaction",
    
    # Migration system
    "Migration",
    "MigrationManager",
    "initialize_database",
    "get_migration_manager",
    
    # High-level utilities
    "RepositoryCoordinator",
    "QueryBuilder",
    "DataValidator",
    "StorageFactory",
    
    # Convenience functions
    "create_storage",
    "query_ingestion",
    "query_extraction", 
    "validate_data",
]
