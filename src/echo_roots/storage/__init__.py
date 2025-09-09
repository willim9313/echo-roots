"""
Storage interfaces and implementations for echo-roots.

This module provides the storage layer abstractions and concrete
implementations for different backend technologies.
"""

from .interfaces import (
    StorageManager,
    StorageBackend,
    IngestionRepository,
    ExtractionRepository,
    TaxonomyRepository,
    MappingRepository,
    ElevationRepository,
    AnalyticsRepository,
    TransactionContext,
    StorageError,
    ConnectionError,
    IntegrityError,
    NotFoundError,
    ConflictError,
)

from .duckdb_backend import DuckDBStorageManager
from .repository import create_storage, RepositoryCoordinator, QueryBuilder, DataValidator

# Neo4j support (optional)
try:
    from .neo4j_backend import Neo4jBackend, Neo4jTaxonomyRepository
    from .hybrid_manager import HybridStorageManager, create_hybrid_storage
    
    NEO4J_AVAILABLE = True
    
    __all_neo4j__ = [
        "Neo4jBackend",
        "Neo4jTaxonomyRepository", 
        "HybridStorageManager",
        "create_hybrid_storage",
    ]
except ImportError:
    NEO4J_AVAILABLE = False
    __all_neo4j__ = []

# Qdrant support (optional)
try:
    from .qdrant_backend import (
        QdrantBackend, QdrantSemanticRepository, QdrantConnectionConfig,
        create_qdrant_backend, create_qdrant_repository
    )
    
    QDRANT_AVAILABLE = True
    
    __all_qdrant__ = [
        "QdrantBackend",
        "QdrantSemanticRepository",
        "QdrantConnectionConfig", 
        "create_qdrant_backend",
        "create_qdrant_repository",
    ]
except ImportError:
    QDRANT_AVAILABLE = False
    __all_qdrant__ = []


__all__ = [
    # Interfaces
    "StorageManager",
    "StorageBackend",
    "IngestionRepository", 
    "ExtractionRepository",
    "TaxonomyRepository",
    "MappingRepository",
    "ElevationRepository",
    "AnalyticsRepository",
    "TransactionContext",
    
    # Exceptions
    "StorageError",
    "ConnectionError", 
    "IntegrityError",
    "NotFoundError",
    "ConflictError",
    
    # Implementations
    "DuckDBStorageManager",
    
    # Utilities
    "create_storage",
    "RepositoryCoordinator",
    "QueryBuilder", 
    "DataValidator",
    
    # Capabilities
    "NEO4J_AVAILABLE",
    "QDRANT_AVAILABLE",
] + __all_neo4j__ + __all_qdrant__

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
