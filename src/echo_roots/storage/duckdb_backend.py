"""
DuckDB storage backend implementation.

This module provides the core DuckDB-based storage implementation
for the echo-roots taxonomy system. DuckDB serves as the primary
backend for ingestion, normalization, and analytics workloads.

Features:
- High-performance analytical queries
- JSON support for flexible schemas
- In-memory and persistent storage modes
- SQL-based operations with Python integration
- Schema versioning and migrations
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncIterator, Union
from uuid import UUID, uuid4

import duckdb
from pydantic import ValidationError

from .interfaces import (
    StorageManager, StorageBackend, TransactionContext,
    IngestionRepository, ExtractionRepository, TaxonomyRepository,
    MappingRepository, ElevationRepository, AnalyticsRepository,
    StorageError, ConnectionError, IntegrityError, NotFoundError, ConflictError
)
from ..models.core import IngestionItem, ExtractionResult, ElevationProposal, Mapping
from ..models.taxonomy import Category, Attribute, SemanticCandidate
from ..models.domain import DomainPack


logger = logging.getLogger(__name__)


class DuckDBTransaction(TransactionContext):
    """Transaction context for DuckDB operations."""
    
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.connection = connection
        self._in_transaction = False
    
    async def __aenter__(self):
        await asyncio.get_event_loop().run_in_executor(
            None, self.connection.execute, "BEGIN TRANSACTION"
        )
        self._in_transaction = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._in_transaction:
            if exc_type is None:
                await self.commit()
            else:
                await self.rollback()
    
    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._in_transaction:
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, "COMMIT"
            )
            self._in_transaction = False
    
    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._in_transaction:
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, "ROLLBACK"
            )
            self._in_transaction = False


class DuckDBBackend:
    """Core DuckDB storage backend."""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path
        self.connection: Optional[duckdb.DuckDBPyConnection] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize DuckDB connection and create schema."""
        try:
            # Create connection (in-memory if no path provided)
            if self.database_path:
                path = Path(self.database_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.connection = duckdb.connect(str(path))
            else:
                self.connection = duckdb.connect()
            
            # Enable JSON extension
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, "INSTALL json; LOAD json;"
            )
            
            # Initialize schema
            await self._create_schema()
            self._initialized = True
            
            logger.info(f"DuckDB backend initialized: {self.database_path or ':memory:'}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to initialize DuckDB: {e}")
    
    async def _create_schema(self) -> None:
        """Create the database schema."""
        schema_sql = """
        -- Ingestion items table  
        CREATE TABLE IF NOT EXISTS ingestion_items (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            description TEXT,
            raw_category VARCHAR,
            raw_attributes JSON,
            source VARCHAR NOT NULL,
            language VARCHAR DEFAULT 'auto',
            metadata JSON,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status VARCHAR DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Extraction results table
        CREATE TABLE IF NOT EXISTS extraction_results (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            item_id VARCHAR NOT NULL,
            attributes JSON NOT NULL,
            terms JSON NOT NULL,
            extraction_metadata JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (item_id) REFERENCES ingestion_items(id)
        );
        
        -- Categories table
        CREATE TABLE IF NOT EXISTS categories (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            name VARCHAR NOT NULL,
            description TEXT,
            domain VARCHAR NOT NULL,
            parent_id VARCHAR,
            level INTEGER DEFAULT 0,
            category_metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES categories(id),
            UNIQUE(name, domain, parent_id)
        );
        
        -- Attributes table
        CREATE TABLE IF NOT EXISTS attributes (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            name VARCHAR NOT NULL,
            data_type VARCHAR NOT NULL,
            description TEXT,
            category_id VARCHAR NOT NULL,
            is_required BOOLEAN DEFAULT FALSE,
            attribute_metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories(id),
            UNIQUE(name, category_id)
        );
        
        -- Semantic candidates table
        CREATE TABLE IF NOT EXISTS semantic_candidates (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            term VARCHAR NOT NULL,
            normalized_term VARCHAR NOT NULL,
            frequency INTEGER DEFAULT 1,
            contexts JSON,
            cluster_id VARCHAR,
            score DOUBLE DEFAULT 0.0,
            language VARCHAR DEFAULT 'auto',
            domain VARCHAR NOT NULL,
            category_id VARCHAR,
            status VARCHAR DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            candidate_metadata JSON,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );
        
        -- Mappings table
        CREATE TABLE IF NOT EXISTS mappings (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            name VARCHAR NOT NULL,
            domain VARCHAR NOT NULL,
            source_schema JSON NOT NULL,
            target_schema JSON NOT NULL,
            field_mappings JSON NOT NULL,
            transformation_rules JSON,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, domain)
        );
        
        -- Elevation proposals table
        CREATE TABLE IF NOT EXISTS elevation_proposals (
            id VARCHAR PRIMARY KEY DEFAULT uuid(),
            proposal_type VARCHAR NOT NULL,
            domain VARCHAR NOT NULL,
            source_item_id VARCHAR,
            proposed_changes JSON NOT NULL,
            justification TEXT,
            status VARCHAR DEFAULT 'pending',
            reviewer_id VARCHAR,
            reviewer_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_item_id) REFERENCES ingestion_items(id)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_ingestion_source ON ingestion_items(source);
        CREATE INDEX IF NOT EXISTS idx_ingestion_status ON ingestion_items(status);
        CREATE INDEX IF NOT EXISTS idx_ingestion_created ON ingestion_items(created_at);
        CREATE INDEX IF NOT EXISTS idx_ingestion_language ON ingestion_items(language);
        
        CREATE INDEX IF NOT EXISTS idx_extraction_item ON extraction_results(item_id);
        CREATE INDEX IF NOT EXISTS idx_extraction_created ON extraction_results(created_at);
        
        CREATE INDEX IF NOT EXISTS idx_categories_domain ON categories(domain);
        CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);
        CREATE INDEX IF NOT EXISTS idx_categories_level ON categories(level);
        
        CREATE INDEX IF NOT EXISTS idx_attributes_category ON attributes(category_id);
        
        CREATE INDEX IF NOT EXISTS idx_semantic_domain ON semantic_candidates(domain);
        CREATE INDEX IF NOT EXISTS idx_semantic_category ON semantic_candidates(category_id);
        CREATE INDEX IF NOT EXISTS idx_semantic_term ON semantic_candidates(normalized_term);
        CREATE INDEX IF NOT EXISTS idx_semantic_cluster ON semantic_candidates(cluster_id);
        
        CREATE INDEX IF NOT EXISTS idx_mappings_domain ON mappings(domain);
        CREATE INDEX IF NOT EXISTS idx_mappings_active ON mappings(is_active);
        
        CREATE INDEX IF NOT EXISTS idx_proposals_status ON elevation_proposals(status);
        CREATE INDEX IF NOT EXISTS idx_proposals_domain ON elevation_proposals(domain);
        CREATE INDEX IF NOT EXISTS idx_proposals_created ON elevation_proposals(created_at);
        """
        
        await asyncio.get_event_loop().run_in_executor(
            None, self.connection.execute, schema_sql
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health and return metrics."""
        if not self._initialized or not self.connection:
            return {"status": "unhealthy", "error": "Not initialized"}
        
        try:
            # Test basic connectivity
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, "SELECT 1 as test"
            )
            
            # Get table counts
            counts_sql = """
            SELECT 
                'ingestion_items' as table_name, COUNT(*) as count FROM ingestion_items
            UNION ALL
            SELECT 
                'extraction_results' as table_name, COUNT(*) as count FROM extraction_results
            UNION ALL
            SELECT 
                'categories' as table_name, COUNT(*) as count FROM categories
            UNION ALL
            SELECT 
                'attributes' as table_name, COUNT(*) as count FROM attributes
            UNION ALL
            SELECT 
                'semantic_candidates' as table_name, COUNT(*) as count FROM semantic_candidates
            UNION ALL
            SELECT 
                'mappings' as table_name, COUNT(*) as count FROM mappings
            UNION ALL
            SELECT 
                'elevation_proposals' as table_name, COUNT(*) as count FROM elevation_proposals
            """
            
            counts_result = await asyncio.get_event_loop().run_in_executor(
                None, self.connection.execute, counts_sql
            )
            
            table_counts = {row[0]: row[1] for row in counts_result.fetchall()}
            
            return {
                "status": "healthy",
                "database_path": self.database_path or ":memory:",
                "table_counts": table_counts,
                "connection_test": "passed"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            await asyncio.get_event_loop().run_in_executor(
                None, self.connection.close
            )
            self.connection = None
            self._initialized = False
            logger.info("DuckDB connection closed")
    
    def transaction(self) -> DuckDBTransaction:
        """Create a new transaction context."""
        if not self.connection:
            raise ConnectionError("Database not initialized")
        return DuckDBTransaction(self.connection)


class DuckDBIngestionRepository:
    """DuckDB implementation of IngestionRepository."""
    
    def __init__(self, backend: DuckDBBackend):
        self.backend = backend
    
    async def store_item(self, item: IngestionItem) -> str:
        """Store an ingestion item."""
        if not self.backend.connection:
            raise ConnectionError("Database not initialized")
        
        try:
            sql = """
            INSERT INTO ingestion_items 
            (id, title, description, raw_category, raw_attributes, source, language, metadata, ingested_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = [
                item.item_id,
                item.title,
                item.description,
                item.raw_category,
                json.dumps(item.raw_attributes) if item.raw_attributes else None,
                item.source,
                item.language,
                json.dumps(item.metadata) if item.metadata else None,
                item.ingested_at,
                'pending'  # Default status
            ]
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.backend.connection.execute, sql, params
            )
            
            return item.item_id
            
        except Exception as e:
            logger.error(f"Failed to store ingestion item: {e}")
            raise StorageError(f"Failed to store item: {e}")
    
    async def get_item(self, item_id: str) -> Optional[IngestionItem]:
        """Retrieve an ingestion item by ID."""
        if not self.backend.connection:
            raise ConnectionError("Database not initialized")
        
        try:
            sql = """
            SELECT id, title, description, raw_category, raw_attributes, source, 
                   language, metadata, ingested_at, status, created_at, updated_at
            FROM ingestion_items 
            WHERE id = ?
            """
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.backend.connection.execute, sql, [item_id]
            )
            
            row = result.fetchone()
            if not row:
                return None
            
            return IngestionItem(
                item_id=row[0],
                title=row[1],
                description=row[2],
                raw_category=row[3],
                raw_attributes=json.loads(row[4]) if row[4] else {},
                source=row[5],
                language=row[6],
                metadata=json.loads(row[7]) if row[7] else {},
                ingested_at=row[8]
            )
            
        except Exception as e:
            logger.error(f"Failed to get ingestion item {item_id}: {e}")
            raise StorageError(f"Failed to get item: {e}")
    
    async def list_items(
        self, 
        source: Optional[str] = None,
        language: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[IngestionItem]:
        """List ingestion items with optional filtering."""
        if not self.backend.connection:
            raise ConnectionError("Database not initialized")
        
        try:
            conditions = []
            params = []
            
            if source:
                conditions.append("source = ?")
                params.append(source)
            
            if language:
                conditions.append("language = ?")
                params.append(language)
            
            if status:
                conditions.append("status = ?")
                params.append(status)
            
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            sql = f"""
            SELECT id, title, description, raw_category, raw_attributes, source, 
                   language, metadata, ingested_at, status, created_at, updated_at
            FROM ingestion_items
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """
            
            params.extend([limit, offset])
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.backend.connection.execute, sql, params
            )
            
            items = []
            for row in result.fetchall():
                items.append(IngestionItem(
                    item_id=row[0],
                    title=row[1],
                    description=row[2],
                    raw_category=row[3],
                    raw_attributes=json.loads(row[4]) if row[4] else {},
                    source=row[5],
                    language=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                    ingested_at=row[8]
                ))
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to list ingestion items: {e}")
            raise StorageError(f"Failed to list items: {e}")
    
    async def update_status(self, item_id: str, status: str) -> bool:
        """Update the status of an ingestion item."""
        if not self.backend.connection:
            raise ConnectionError("Database not initialized")
        
        try:
            sql = """
            UPDATE ingestion_items 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.backend.connection.execute, sql, [status, item_id]
            )
            
            # Check if any rows were affected
            # DuckDB doesn't have rowcount, so let's check if item exists
            check_sql = "SELECT COUNT(*) FROM ingestion_items WHERE id = ?"
            check_result = await asyncio.get_event_loop().run_in_executor(
                None, self.backend.connection.execute, check_sql, [item_id]
            )
            
            return check_result.fetchone()[0] > 0
            
        except Exception as e:
            logger.error(f"Failed to update item status {item_id}: {e}")
            raise StorageError(f"Failed to update status: {e}")
    
    async def delete_item(self, item_id: str) -> bool:
        """Delete an ingestion item."""
        if not self.backend.connection:
            raise ConnectionError("Database not initialized")
        
        try:
            # Delete related extraction results first (if any)
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.backend.connection.execute,
                "DELETE FROM extraction_results WHERE item_id = ?",
                [item_id]
            )
            
            # Delete the item
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.backend.connection.execute,
                "DELETE FROM ingestion_items WHERE id = ?",
                [item_id]
            )
            
            # Check if item was actually deleted by trying to find it
            check_sql = "SELECT COUNT(*) FROM ingestion_items WHERE id = ?"
            check_result = await asyncio.get_event_loop().run_in_executor(
                None, self.backend.connection.execute, check_sql, [item_id]
            )
            
            return check_result.fetchone()[0] == 0
            
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {e}")
            raise StorageError(f"Failed to delete item: {e}")


class DuckDBStorageManager(StorageManager):
    """Main storage manager using DuckDB backend."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize DuckDB backend
        database_path = config.get("duckdb", {}).get("database_path")
        self.backend = DuckDBBackend(database_path)
        
        # Initialize repositories
        self._ingestion = DuckDBIngestionRepository(self.backend)
        # TODO: Initialize other repositories as we implement them
        self._extraction = None
        self._taxonomy = None
        self._mappings = None
        self._elevation = None
        self._analytics = None
    
    async def initialize(self) -> None:
        """Initialize the storage manager."""
        await self.backend.initialize()
        self._initialized = True
        logger.info("DuckDB storage manager initialized")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of storage backends."""
        backend_health = await self.backend.health_check()
        
        return {
            "storage_manager": "DuckDBStorageManager",
            "initialized": self._initialized,
            "duckdb": backend_health
        }
    
    async def close(self) -> None:
        """Close storage connections."""
        await self.backend.close()
        self._initialized = False
        logger.info("DuckDB storage manager closed")
    
    @property
    def ingestion(self) -> IngestionRepository:
        """Access to ingestion repository."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        return self._ingestion
    
    @property
    def extraction(self) -> ExtractionRepository:
        """Access to extraction repository."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        if not self._extraction:
            raise NotImplementedError("Extraction repository not yet implemented")
        return self._extraction
    
    @property
    def taxonomy(self) -> TaxonomyRepository:
        """Access to taxonomy repository."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        if not self._taxonomy:
            raise NotImplementedError("Taxonomy repository not yet implemented")
        return self._taxonomy
    
    @property
    def mappings(self) -> MappingRepository:
        """Access to mappings repository."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        if not self._mappings:
            raise NotImplementedError("Mappings repository not yet implemented")
        return self._mappings
    
    @property
    def elevation(self) -> ElevationRepository:
        """Access to elevation repository."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        if not self._elevation:
            raise NotImplementedError("Elevation repository not yet implemented")
        return self._elevation
    
    @property
    def analytics(self) -> AnalyticsRepository:
        """Access to analytics repository."""
        if not self._initialized:
            raise StorageError("Storage manager not initialized")
        if not self._analytics:
            raise NotImplementedError("Analytics repository not yet implemented")
        return self._analytics
