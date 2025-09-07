"""
High-level repository patterns for echo-roots storage.

This module provides convenience classes and utilities that build
on top of the storage interfaces to provide common patterns and
workflows for working with the taxonomy data.

Features:
- Repository composition and coordination
- Transaction management across repositories
- Bulk operations and batch processing
- Query builders and advanced filtering
- Data validation and integrity checks
"""

import logging
from typing import List, Optional, Dict, Any, AsyncIterator, Union
from datetime import datetime
from uuid import uuid4

from .interfaces import (
    StorageManager, TransactionContext,
    IngestionRepository, ExtractionRepository, TaxonomyRepository,
    MappingRepository, ElevationRepository, AnalyticsRepository,
    StorageError, NotFoundError, ConflictError
)
from .duckdb_backend import DuckDBStorageManager
from .migrations import initialize_database
from ..models.core import IngestionItem, ExtractionResult, ElevationProposal, Mapping
from ..models.taxonomy import Category, Attribute, SemanticCandidate
from ..models.domain import DomainPack


logger = logging.getLogger(__name__)


class RepositoryCoordinator:
    """
    Coordinates operations across multiple repositories.
    
    Provides high-level workflows that involve multiple storage
    components and ensures data consistency across repositories.
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
    
    async def ingest_and_extract(
        self,
        item: IngestionItem,
        auto_extract: bool = True
    ) -> tuple[str, Optional[str]]:
        """
        Store an ingestion item and optionally trigger extraction.
        
        Returns:
            Tuple of (item_id, extraction_result_id)
        """
        try:
            # Store the ingestion item
            item_id = await self.storage.ingestion.store_item(item)
            
            extraction_id = None
            if auto_extract:
                # TODO: Trigger extraction pipeline
                # This would integrate with the T3 LLM extraction pipeline
                logger.info(f"Auto-extraction triggered for item {item_id}")
            
            return item_id, extraction_id
            
        except Exception as e:
            logger.error(f"Failed to ingest and extract item: {e}")
            raise StorageError(f"Ingestion workflow failed: {e}")
    
    async def process_domain_pack(self, domain_pack: DomainPack) -> Dict[str, Any]:
        """
        Process a domain pack and update the taxonomy.
        
        This involves:
        1. Validating the domain pack structure
        2. Storing/updating categories and attributes
        3. Creating semantic terms
        4. Setting up field mappings
        
        Returns:
            Summary of changes made
        """
        changes = {
            "categories_created": 0,
            "categories_updated": 0,
            "attributes_created": 0,
            "semantic_terms_created": 0,
            "mappings_created": 0
        }
        
        try:
            # TODO: Implement when taxonomy repository is complete
            logger.info(f"Processing domain pack: {domain_pack.name}")
            
            # For now, just return empty changes
            return changes
            
        except Exception as e:
            logger.error(f"Failed to process domain pack {domain_pack.name}: {e}")
            raise StorageError(f"Domain pack processing failed: {e}")
    
    async def bulk_ingest(
        self,
        items: List[IngestionItem],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Bulk ingest multiple items efficiently.
        
        Processes items in batches to optimize performance and
        provide progress tracking for large datasets.
        
        Returns:
            Summary of ingestion results
        """
        results = {
            "total_items": len(items),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}, items {i+1}-{min(i+batch_size, len(items))}")
                
                for item in batch:
                    try:
                        await self.storage.ingestion.store_item(item)
                        results["successful"] += 1
                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append({
                            "item_id": getattr(item, 'id', 'unknown'),
                            "error": str(e)
                        })
                        logger.error(f"Failed to store item: {e}")
            
            logger.info(f"Bulk ingest complete: {results['successful']} successful, {results['failed']} failed")
            return results
            
        except Exception as e:
            logger.error(f"Bulk ingest failed: {e}")
            raise StorageError(f"Bulk ingest failed: {e}")


class QueryBuilder:
    """
    Builder pattern for constructing complex queries.
    
    Provides a fluent interface for building filtered queries
    across different repositories.
    """
    
    def __init__(self, repository_type: str):
        self.repository_type = repository_type
        self.filters = {}
        self.sorting = []
        self.pagination = {"limit": 100, "offset": 0}
    
    def filter_by_source(self, source: str) -> 'QueryBuilder':
        """Filter results by source."""
        self.filters["source"] = source
        return self
    
    def filter_by_status(self, status: str) -> 'QueryBuilder':
        """Filter results by status."""
        self.filters["status"] = status
        return self
    
    def filter_by_date_range(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'QueryBuilder':
        """Filter results by date range."""
        if start_date:
            self.filters["start_date"] = start_date
        if end_date:
            self.filters["end_date"] = end_date
        return self
    
    def filter_by_confidence(
        self,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None
    ) -> 'QueryBuilder':
        """Filter results by confidence range."""
        if min_confidence is not None:
            self.filters["confidence_min"] = min_confidence
        if max_confidence is not None:
            self.filters["confidence_max"] = max_confidence
        return self
    
    def sort_by(self, field: str, ascending: bool = True) -> 'QueryBuilder':
        """Add sorting criteria."""
        self.sorting.append({
            "field": field,
            "ascending": ascending
        })
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Set result limit."""
        self.pagination["limit"] = limit
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """Set result offset."""
        self.pagination["offset"] = offset
        return self
    
    def page(self, page: int, page_size: int = 100) -> 'QueryBuilder':
        """Set pagination by page number."""
        self.pagination["limit"] = page_size
        self.pagination["offset"] = (page - 1) * page_size
        return self
    
    async def execute(self, storage: StorageManager) -> List[Any]:
        """Execute the query against the appropriate repository."""
        if self.repository_type == "ingestion":
            return await storage.ingestion.list_items(
                source=self.filters.get("source"),
                language=self.filters.get("language"),
                status=self.filters.get("status"),
                limit=self.pagination["limit"],
                offset=self.pagination["offset"]
            )
        elif self.repository_type == "extraction":
            return await storage.extraction.list_results(
                source=self.filters.get("source"),
                limit=self.pagination["limit"],
                offset=self.pagination["offset"]
            )
        else:
            raise ValueError(f"Unsupported repository type: {self.repository_type}")


class DataValidator:
    """
    Provides data validation and integrity checking.
    
    Ensures data consistency and validates relationships
    between different entities in the storage system.
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage = storage_manager
    
    async def validate_ingestion_item(self, item: IngestionItem) -> List[str]:
        """
        Validate an ingestion item for common issues.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not item.domain:
            errors.append("Domain is required")
        
        if not item.source_type:
            errors.append("Source type is required")
        
        if not item.source_identifier:
            errors.append("Source identifier is required")
        
        if not item.raw_content or not item.raw_content.strip():
            errors.append("Raw content cannot be empty")
        
        # Check for duplicates
        try:
            existing_items = await self.storage.ingestion.list_items(
                domain=item.domain,
                limit=1000  # TODO: Use more efficient existence check
            )
            
            for existing in existing_items:
                if (existing.source_type == item.source_type and
                    existing.source_identifier == item.source_identifier):
                    errors.append(f"Duplicate item: {item.source_type}:{item.source_identifier}")
                    break
        
        except Exception as e:
            logger.warning(f"Could not check for duplicates: {e}")
        
        return errors
    
    async def validate_extraction_result(
        self, 
        result: ExtractionResult
    ) -> List[str]:
        """
        Validate an extraction result.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if not result.item_id:
            errors.append("Item ID is required")
        
        if not result.domain:
            errors.append("Domain is required")
        
        if not result.extracted_categories:
            errors.append("At least one category must be extracted")
        
        # Validate confidence score
        if result.confidence_score is not None:
            if not (0.0 <= result.confidence_score <= 1.0):
                errors.append("Confidence score must be between 0.0 and 1.0")
        
        # Check if source item exists
        try:
            source_item = await self.storage.ingestion.get_item(result.item_id)
            if not source_item:
                errors.append(f"Source item {result.item_id} not found")
            elif source_item.domain != result.domain:
                errors.append("Domain mismatch between extraction result and source item")
        
        except Exception as e:
            logger.warning(f"Could not validate source item: {e}")
        
        return errors


class StorageFactory:
    """
    Factory for creating and configuring storage managers.
    
    Provides convenient methods for setting up storage with
    different configurations and backends.
    """
    
    @staticmethod
    async def create_duckdb_storage(
        database_path: Optional[str] = None,
        initialize_schema: bool = True
    ) -> DuckDBStorageManager:
        """
        Create a DuckDB storage manager.
        
        Args:
            database_path: Path to DuckDB file (None for in-memory)
            initialize_schema: Whether to run migrations on startup
        
        Returns:
            Configured and initialized storage manager
        """
        config = {
            "duckdb": {
                "database_path": database_path
            }
        }
        
        storage = DuckDBStorageManager(config)
        await storage.initialize()
        
        if initialize_schema:
            await initialize_database(storage.backend)
        
        logger.info(f"Created DuckDB storage: {database_path or ':memory:'}")
        return storage
    
    @staticmethod
    async def create_test_storage() -> DuckDBStorageManager:
        """
        Create an in-memory storage manager for testing.
        
        Returns:
            Configured test storage manager
        """
        return await StorageFactory.create_duckdb_storage(
            database_path=None,
            initialize_schema=True
        )


# Convenience functions for common operations

async def create_storage(config: Optional[Dict[str, Any]] = None) -> StorageManager:
    """
    Create and initialize a storage manager with default configuration.
    
    Args:
        config: Optional configuration dict. If None, uses in-memory DuckDB.
    
    Returns:
        Initialized storage manager
    """
    if config is None:
        return await StorageFactory.create_test_storage()
    
    # For now, only DuckDB is implemented
    return await StorageFactory.create_duckdb_storage(
        database_path=config.get("duckdb", {}).get("database_path"),
        initialize_schema=config.get("initialize_schema", True)
    )


def query_ingestion() -> QueryBuilder:
    """Create a query builder for ingestion items."""
    return QueryBuilder("ingestion")


def query_extraction() -> QueryBuilder:
    """Create a query builder for extraction results."""
    return QueryBuilder("extraction")


async def validate_data(
    storage: StorageManager,
    item: Union[IngestionItem, ExtractionResult]
) -> List[str]:
    """
    Validate a data item against storage constraints.
    
    Args:
        storage: Storage manager for validation queries
        item: Item to validate
    
    Returns:
        List of validation errors
    """
    validator = DataValidator(storage)
    
    if isinstance(item, IngestionItem):
        return await validator.validate_ingestion_item(item)
    elif isinstance(item, ExtractionResult):
        return await validator.validate_extraction_result(item)
    else:
        return [f"Unsupported item type: {type(item)}"]
