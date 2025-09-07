"""
Tests for the storage interfaces and DuckDB backend.

These tests verify the storage layer functionality including:
- Storage manager initialization and health checks
- Repository operations (CRUD)
- Transaction management
- Migration system
- Data validation
- Error handling
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from echo_roots.storage import (
    create_storage, StorageFactory, DuckDBStorageManager,
    StorageError, NotFoundError, ConnectionError,
    query_ingestion, validate_data, RepositoryCoordinator,
    initialize_database, get_migration_manager
)
from echo_roots.models.core import IngestionItem
from echo_roots.models.taxonomy import Category


class TestStorageFactory:
    """Test storage factory functionality."""
    
    @pytest.mark.asyncio
    async def test_create_test_storage(self):
        """Test creating in-memory test storage."""
        storage = await StorageFactory.create_test_storage()
        
        assert isinstance(storage, DuckDBStorageManager)
        assert storage._initialized
        
        # Test health check
        health = await storage.health_check()
        assert health["status"] == "healthy"
        assert health["duckdb"]["status"] == "healthy"
        
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_create_file_storage(self):
        """Test creating file-based storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            
            storage = await StorageFactory.create_duckdb_storage(
                database_path=str(db_path)
            )
            
            assert isinstance(storage, DuckDBStorageManager)
            assert storage._initialized
            assert db_path.exists()
            
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_create_storage_convenience(self):
        """Test convenience create_storage function."""
        storage = await create_storage()
        
        assert isinstance(storage, DuckDBStorageManager)
        assert storage._initialized
        
        await storage.close()


class TestDuckDBBackend:
    """Test DuckDB backend implementation."""
    
    @pytest.fixture
    async def storage(self):
        """Create test storage."""
        storage = await StorageFactory.create_test_storage()
        yield storage
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_backend_initialization(self, storage):
        """Test backend initialization."""
        assert storage.backend._initialized
        assert storage.backend.connection is not None
    
    @pytest.mark.asyncio
    async def test_health_check(self, storage):
        """Test health check functionality."""
        health = await storage.health_check()
        
        assert health["status"] == "healthy"
        assert health["storage_manager"] == "DuckDBStorageManager"
        assert health["initialized"] is True
        
        duckdb_health = health["duckdb"]
        assert duckdb_health["status"] == "healthy"
        assert "table_counts" in duckdb_health
        assert "connection_test" in duckdb_health
    
    @pytest.mark.asyncio
    async def test_transaction_context(self, storage):
        """Test transaction management."""
        async with storage.backend.transaction() as tx:
            # Transaction should be active
            assert tx._in_transaction
        
        # Transaction should be committed
        assert not tx._in_transaction
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, storage):
        """Test transaction rollback on error."""
        try:
            async with storage.backend.transaction():
                # Insert a test item
                await storage.ingestion.store_item(IngestionItem(
                    domain="test",
                    source_type="test",
                    source_identifier="test1",
                    raw_content="test content"
                ))
                
                # Force an error to trigger rollback
                raise ValueError("Test error")
        
        except ValueError:
            pass
        
        # Verify item was not stored due to rollback
        items = await storage.ingestion.list_items(domain="test")
        assert len(items) == 0


class TestIngestionRepository:
    """Test ingestion repository operations."""
    
    @pytest.fixture
    async def storage(self):
        """Create test storage."""
        storage = await StorageFactory.create_test_storage()
        yield storage
        await storage.close()
    
    @pytest.fixture
    def sample_item(self):
        """Create a sample ingestion item."""
        return IngestionItem(
            domain="ecommerce",
            source_type="product_listing",
            source_identifier="prod_123",
            raw_content="Sample product data",
            metadata={"category": "electronics", "brand": "TestBrand"}
        )
    
    @pytest.mark.asyncio
    async def test_store_and_get_item(self, storage, sample_item):
        """Test storing and retrieving an item."""
        # Store item
        item_id = await storage.ingestion.store_item(sample_item)
        assert item_id is not None
        
        # Retrieve item
        retrieved_item = await storage.ingestion.get_item(item_id)
        assert retrieved_item is not None
        assert retrieved_item.domain == sample_item.domain
        assert retrieved_item.source_type == sample_item.source_type
        assert retrieved_item.source_identifier == sample_item.source_identifier
        assert retrieved_item.raw_content == sample_item.raw_content
        assert retrieved_item.metadata == sample_item.metadata
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_item(self, storage):
        """Test retrieving a non-existent item."""
        result = await storage.ingestion.get_item("nonexistent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_items_empty(self, storage):
        """Test listing items when none exist."""
        items = await storage.ingestion.list_items()
        assert items == []
    
    @pytest.mark.asyncio
    async def test_list_items_with_data(self, storage, sample_item):
        """Test listing items with data."""
        # Store multiple items
        item1_id = await storage.ingestion.store_item(sample_item)
        
        item2 = IngestionItem(
            domain="fashion",
            source_type="product_listing",
            source_identifier="prod_456",
            raw_content="Fashion product data"
        )
        item2_id = await storage.ingestion.store_item(item2)
        
        # List all items
        all_items = await storage.ingestion.list_items()
        assert len(all_items) == 2
        
        # List by domain
        ecommerce_items = await storage.ingestion.list_items(domain="ecommerce")
        assert len(ecommerce_items) == 1
        assert ecommerce_items[0].domain == "ecommerce"
        
        fashion_items = await storage.ingestion.list_items(domain="fashion")
        assert len(fashion_items) == 1
        assert fashion_items[0].domain == "fashion"
    
    @pytest.mark.asyncio
    async def test_update_status(self, storage, sample_item):
        """Test updating item status."""
        # Store item
        item_id = await storage.ingestion.store_item(sample_item)
        
        # Update status
        success = await storage.ingestion.update_status(item_id, "processing")
        assert success
        
        # Verify status update
        retrieved_item = await storage.ingestion.get_item(item_id)
        assert retrieved_item.status == "processing"
        
        # Test updating non-existent item
        success = await storage.ingestion.update_status("nonexistent", "completed")
        assert not success
    
    @pytest.mark.asyncio
    async def test_delete_item(self, storage, sample_item):
        """Test deleting an item."""
        # Store item
        item_id = await storage.ingestion.store_item(sample_item)
        
        # Verify item exists
        assert await storage.ingestion.get_item(item_id) is not None
        
        # Delete item
        success = await storage.ingestion.delete_item(item_id)
        assert success
        
        # Verify item is gone
        assert await storage.ingestion.get_item(item_id) is None
        
        # Test deleting non-existent item
        success = await storage.ingestion.delete_item("nonexistent")
        assert not success
    
    @pytest.mark.asyncio
    async def test_list_with_pagination(self, storage):
        """Test pagination in list operations."""
        # Store multiple items
        for i in range(25):
            item = IngestionItem(
                domain="test",
                source_type="test",
                source_identifier=f"test_{i}",
                raw_content=f"Content {i}"
            )
            await storage.ingestion.store_item(item)
        
        # Test default pagination
        items_page1 = await storage.ingestion.list_items(domain="test", limit=10)
        assert len(items_page1) == 10
        
        # Test offset
        items_page2 = await storage.ingestion.list_items(domain="test", limit=10, offset=10)
        assert len(items_page2) == 10
        
        # Verify different items
        page1_ids = {item.id for item in items_page1}
        page2_ids = {item.id for item in items_page2}
        assert page1_ids.isdisjoint(page2_ids)
        
        # Test remaining items
        items_page3 = await storage.ingestion.list_items(domain="test", limit=10, offset=20)
        assert len(items_page3) == 5


class TestQueryBuilder:
    """Test query builder functionality."""
    
    @pytest.fixture
    async def storage(self):
        """Create test storage with sample data."""
        storage = await StorageFactory.create_test_storage()
        
        # Add sample data
        for i in range(10):
            domain = "ecommerce" if i % 2 == 0 else "fashion"
            status = "pending" if i % 3 == 0 else "completed"
            
            item = IngestionItem(
                domain=domain,
                source_type="product",
                source_identifier=f"item_{i}",
                raw_content=f"Content {i}",
                status=status
            )
            await storage.ingestion.store_item(item)
        
        yield storage
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_query_builder_basic(self, storage):
        """Test basic query builder functionality."""
        # Query all items
        all_items = await query_ingestion().execute(storage)
        assert len(all_items) == 10
        
        # Query by domain
        ecommerce_items = await query_ingestion().filter_by_domain("ecommerce").execute(storage)
        assert len(ecommerce_items) == 5
        assert all(item.domain == "ecommerce" for item in ecommerce_items)
        
        # Query by status
        pending_items = await query_ingestion().filter_by_status("pending").execute(storage)
        assert len(pending_items) > 0
        assert all(item.status == "pending" for item in pending_items)
    
    @pytest.mark.asyncio
    async def test_query_builder_chaining(self, storage):
        """Test chaining multiple filters."""
        # Chain domain and status filters
        filtered_items = await (
            query_ingestion()
            .filter_by_domain("ecommerce")
            .filter_by_status("completed")
            .execute(storage)
        )
        
        assert all(
            item.domain == "ecommerce" and item.status == "completed"
            for item in filtered_items
        )
    
    @pytest.mark.asyncio
    async def test_query_builder_pagination(self, storage):
        """Test query builder pagination."""
        # Test limit
        limited_items = await query_ingestion().limit(3).execute(storage)
        assert len(limited_items) == 3
        
        # Test offset
        offset_items = await query_ingestion().offset(5).limit(3).execute(storage)
        assert len(offset_items) == 3
        
        # Test page method
        page_items = await query_ingestion().page(2, 3).execute(storage)
        assert len(page_items) == 3


class TestDataValidation:
    """Test data validation functionality."""
    
    @pytest.fixture
    async def storage(self):
        """Create test storage."""
        storage = await StorageFactory.create_test_storage()
        yield storage
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_validate_valid_item(self, storage):
        """Test validation of a valid item."""
        valid_item = IngestionItem(
            domain="ecommerce",
            source_type="product",
            source_identifier="prod_123",
            raw_content="Sample product content"
        )
        
        errors = await validate_data(storage, valid_item)
        assert errors == []
    
    @pytest.mark.asyncio
    async def test_validate_invalid_item(self, storage):
        """Test validation of an invalid item."""
        invalid_item = IngestionItem(
            domain="",  # Empty domain
            source_type="",  # Empty source type
            source_identifier="prod_123",
            raw_content=""  # Empty content
        )
        
        errors = await validate_data(storage, invalid_item)
        assert len(errors) > 0
        assert any("Domain is required" in error for error in errors)
        assert any("Source type is required" in error for error in errors)
        assert any("Raw content cannot be empty" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_validate_duplicate_item(self, storage):
        """Test validation detects duplicates."""
        # Store first item
        item1 = IngestionItem(
            domain="ecommerce",
            source_type="product",
            source_identifier="prod_123",
            raw_content="Original content"
        )
        await storage.ingestion.store_item(item1)
        
        # Try to validate duplicate
        duplicate_item = IngestionItem(
            domain="ecommerce",
            source_type="product",
            source_identifier="prod_123",  # Same identifier
            raw_content="Duplicate content"
        )
        
        errors = await validate_data(storage, duplicate_item)
        assert len(errors) > 0
        assert any("Duplicate item" in error for error in errors)


class TestRepositoryCoordinator:
    """Test repository coordinator functionality."""
    
    @pytest.fixture
    async def storage(self):
        """Create test storage."""
        storage = await StorageFactory.create_test_storage()
        yield storage
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_ingest_and_extract(self, storage):
        """Test coordinated ingestion and extraction."""
        coordinator = RepositoryCoordinator(storage)
        
        item = IngestionItem(
            domain="ecommerce",
            source_type="product",
            source_identifier="prod_123",
            raw_content="Sample product"
        )
        
        # Test ingestion without auto-extract
        item_id, extraction_id = await coordinator.ingest_and_extract(
            item, auto_extract=False
        )
        
        assert item_id is not None
        assert extraction_id is None
        
        # Verify item was stored
        stored_item = await storage.ingestion.get_item(item_id)
        assert stored_item is not None
    
    @pytest.mark.asyncio
    async def test_bulk_ingest(self, storage):
        """Test bulk ingestion functionality."""
        coordinator = RepositoryCoordinator(storage)
        
        # Create test items
        items = []
        for i in range(15):
            items.append(IngestionItem(
                domain="test",
                source_type="bulk",
                source_identifier=f"bulk_{i}",
                raw_content=f"Bulk content {i}"
            ))
        
        # Bulk ingest with small batch size
        results = await coordinator.bulk_ingest(items, batch_size=5)
        
        assert results["total_items"] == 15
        assert results["successful"] == 15
        assert results["failed"] == 0
        assert len(results["errors"]) == 0
        
        # Verify items were stored
        stored_items = await storage.ingestion.list_items(domain="test")
        assert len(stored_items) == 15


class TestMigrationSystem:
    """Test database migration system."""
    
    @pytest.mark.asyncio
    async def test_migration_manager(self):
        """Test migration manager functionality."""
        # Create storage without initializing schema
        config = {"duckdb": {"database_path": None}}
        storage = DuckDBStorageManager(config)
        await storage.initialize()
        
        try:
            # Get migration manager
            migration_manager = await get_migration_manager(storage.backend)
            
            # Check initial state
            current_version = await migration_manager.get_current_version()
            assert current_version is None
            
            # Apply migrations
            await migration_manager.migrate_to_latest()
            
            # Check final state
            current_version = await migration_manager.get_current_version()
            assert current_version is not None
            
            # Get migration status
            status = await migration_manager.get_migration_status()
            assert status["current_version"] is not None
            assert status["applied_count"] > 0
            assert status["pending_count"] == 0
            
        finally:
            await storage.close()
    
    @pytest.mark.asyncio
    async def test_initialize_database(self):
        """Test database initialization."""
        config = {"duckdb": {"database_path": None}}
        storage = DuckDBStorageManager(config)
        await storage.initialize()
        
        try:
            # Initialize database
            await initialize_database(storage.backend)
            
            # Verify schema exists
            health = await storage.health_check()
            assert health["status"] == "healthy"
            
            # Verify tables exist by checking counts
            table_counts = health["duckdb"]["table_counts"]
            expected_tables = [
                "ingestion_items", "extraction_results", "categories",
                "attributes", "semantic_terms", "mappings",
                "elevation_proposals", "schema_migrations"
            ]
            
            for table in expected_tables:
                assert table in table_counts
                assert isinstance(table_counts[table], int)
        
        finally:
            await storage.close()


if __name__ == "__main__":
    pytest.main([__file__])
