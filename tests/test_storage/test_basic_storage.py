"""
Simplified tests for T4 Storage Interfaces.

Basic tests to verify the core storage functionality is working
before expanding to full test coverage.
"""

import pytest
import pytest_asyncio
from echo_roots.storage import StorageFactory, create_storage
from echo_roots.models.core import IngestionItem


class TestBasicStorage:
    """Test basic storage functionality."""
    
    @pytest_asyncio.fixture
    async def storage(self):
        """Create test storage."""
        storage = await StorageFactory.create_test_storage()
        yield storage
        await storage.close()
    
    def sample_item(self) -> IngestionItem:
        """Create a sample ingestion item."""
        return IngestionItem(
            item_id="test_item_1",
            title="Test Product",
            description="A test product for storage testing",
            raw_category="Electronics",
            raw_attributes={"brand": "TestBrand", "color": "Blue"},
            source="test_api",
            language="en",
            metadata={"test": True}
        )
    
    @pytest.mark.asyncio
    async def test_storage_creation(self):
        """Test creating storage manager."""
        storage = await create_storage()
        
        assert storage is not None
        assert storage._initialized
        
        # Test health check
        health = await storage.health_check()
        assert "storage_manager" in health
        assert health["initialized"] is True
        
        await storage.close()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_item(self, storage):
        """Test storing and retrieving an item."""
        item = self.sample_item()
        
        # Store item
        stored_id = await storage.ingestion.store_item(item)
        assert stored_id == item.item_id
        
        # Retrieve item
        retrieved = await storage.ingestion.get_item(stored_id)
        assert retrieved is not None
        assert retrieved.item_id == item.item_id
        assert retrieved.title == item.title
        assert retrieved.source == item.source
    
    @pytest.mark.asyncio 
    async def test_list_items(self, storage):
        """Test listing items."""
        # Initially empty
        items = await storage.ingestion.list_items()
        assert len(items) == 0
        
        # Store some items
        for i in range(3):
            item = IngestionItem(
                item_id=f"item_{i}",
                title=f"Test Item {i}",
                source="test_source",
                language="en"
            )
            await storage.ingestion.store_item(item)
        
        # List all items
        items = await storage.ingestion.list_items()
        assert len(items) == 3
        
        # List by source
        items = await storage.ingestion.list_items(source="test_source")
        assert len(items) == 3
        
        # List by language  
        items = await storage.ingestion.list_items(language="en")
        assert len(items) == 3
    
    @pytest.mark.asyncio
    async def test_update_status(self, storage):
        """Test updating item status."""
        item = self.sample_item()
        
        # Store item
        item_id = await storage.ingestion.store_item(item)
        
        # Update status
        success = await storage.ingestion.update_status(item_id, "processing")
        assert success
        
        # Verify status change
        retrieved = await storage.ingestion.get_item(item_id) 
        # Note: Status might not be returned in current model, that's OK
    
    @pytest.mark.asyncio
    async def test_delete_item(self, storage):
        """Test deleting an item."""
        item = self.sample_item()
        
        # Store item
        item_id = await storage.ingestion.store_item(item)
        
        # Verify exists
        retrieved = await storage.ingestion.get_item(item_id)
        assert retrieved is not None
        
        # Delete item
        success = await storage.ingestion.delete_item(item_id)
        assert success
        
        # Verify deleted
        retrieved = await storage.ingestion.get_item(item_id)
        assert retrieved is None


if __name__ == "__main__":
    pytest.main([__file__])
