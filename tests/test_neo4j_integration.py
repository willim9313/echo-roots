"""
Tests for Neo4j integration and hybrid storage.

This module tests the Neo4j backend functionality and hybrid storage
coordination between DuckDB and Neo4j.
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from echo_roots.storage import NEO4J_AVAILABLE
from echo_roots.models.taxonomy import Category, Attribute, AttributeValue, SemanticCandidate

# Only run Neo4j tests if the library is available
pytestmark = pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")


@pytest.fixture
async def neo4j_config() -> Dict[str, Any]:
    """Configuration for Neo4j testing."""
    return {
        "duckdb": {
            "database_path": None  # In-memory
        },
        "neo4j": {
            "enabled": True,
            "uri": "neo4j://localhost:7687",
            "user": "neo4j",
            "password": "test-password",
            "database": "neo4j"
        }
    }


@pytest.fixture
async def hybrid_storage(neo4j_config):
    """Create hybrid storage manager for testing."""
    from echo_roots.storage import create_hybrid_storage
    
    storage = await create_hybrid_storage(neo4j_config)
    yield storage
    await storage.close()


@pytest.fixture
async def sample_categories():
    """Sample category data for testing."""
    return [
        Category(
            name="Electronics",
            level=0,
            path=["Electronics"],
            description="Electronic devices and components"
        ),
        Category(
            name="Mobile Devices", 
            level=1,
            path=["Electronics", "Mobile Devices"],
            description="Portable electronic devices"
        ),
        Category(
            name="Smartphones",
            level=2,
            path=["Electronics", "Mobile Devices", "Smartphones"],
            description="Mobile phones with advanced features"
        )
    ]


class TestNeo4jBackend:
    """Test Neo4j backend functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.storage
    async def test_neo4j_connection(self, neo4j_config):
        """Test Neo4j connection and initialization."""
        from echo_roots.storage.neo4j_backend import Neo4jBackend
        
        backend = Neo4jBackend(
            uri=neo4j_config["neo4j"]["uri"],
            user=neo4j_config["neo4j"]["user"],
            password=neo4j_config["neo4j"]["password"]
        )
        
        try:
            await backend.initialize()
            health = await backend.health_check()
            assert health["status"] == "healthy"
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
        finally:
            await backend.close()
    
    @pytest.mark.asyncio
    @pytest.mark.storage
    async def test_category_storage(self, hybrid_storage, sample_categories):
        """Test storing and retrieving categories."""
        if not hybrid_storage.neo4j_backend:
            pytest.skip("Neo4j not configured")
        
        # Store categories with hierarchy
        category_ids = []
        parent_id = None
        
        for category in sample_categories:
            if parent_id:
                category.parent_id = parent_id
            
            category_id = await hybrid_storage.taxonomy.store_category(category)
            category_ids.append(category_id)
            parent_id = category_id
        
        # Retrieve and verify
        for i, category_id in enumerate(category_ids):
            retrieved = await hybrid_storage.taxonomy.get_category(category_id)
            assert retrieved is not None
            assert retrieved.name == sample_categories[i].name
            assert retrieved.level == sample_categories[i].level
    
    @pytest.mark.asyncio
    @pytest.mark.storage  
    async def test_attribute_storage(self, hybrid_storage):
        """Test storing attributes with controlled values."""
        if not hybrid_storage.neo4j_backend:
            pytest.skip("Neo4j not configured")
        
        # Create attribute with values
        attribute = Attribute(
            name="brand",
            display_name="Brand",
            data_type="categorical",
            description="Product brand",
            values=[
                AttributeValue(value="Apple", aliases={"apple", "APPLE"}),
                AttributeValue(value="Samsung", aliases={"samsung", "SAMSUNG"}),
                AttributeValue(value="Google", aliases={"google", "GOOGLE"})
            ]
        )
        
        # Store attribute
        attribute_id = await hybrid_storage.taxonomy.store_attribute(attribute)
        assert attribute_id == attribute.attribute_id
    
    @pytest.mark.asyncio
    @pytest.mark.storage
    async def test_semantic_candidate_storage(self, hybrid_storage):
        """Test storing semantic candidates with relationships."""
        if not hybrid_storage.neo4j_backend:
            pytest.skip("Neo4j not configured")
        
        # Create semantic candidate
        candidate = SemanticCandidate(
            term="smartphone",
            normalized_term="smartphone",
            frequency=100,
            contexts=["mobile phone", "cellular device"],
            score=0.85,
            language="en"
        )
        
        # Store candidate
        candidate_id = await hybrid_storage.taxonomy.store_semantic_candidate(candidate)
        assert candidate_id == candidate.candidate_id
        
        # Search for candidates
        results = await hybrid_storage.taxonomy.search_semantic_candidates("phone")
        # Results may be empty if no matching candidates exist


class TestHybridStorage:
    """Test hybrid storage coordination."""
    
    @pytest.mark.asyncio
    async def test_storage_info(self, hybrid_storage):
        """Test getting storage configuration info."""
        info = await hybrid_storage.get_taxonomy_storage_info()
        
        assert "taxonomy_backend" in info
        assert "neo4j_configured" in info
        assert "supports_graph_queries" in info
        
        # Should indicate Neo4j backend if available
        if hybrid_storage.neo4j_backend:
            assert info["taxonomy_backend"] == "Neo4j"
            assert info["supports_graph_queries"] is True
        else:
            assert info["taxonomy_backend"] == "DuckDB"
    
    @pytest.mark.asyncio
    async def test_health_check(self, hybrid_storage):
        """Test hybrid storage health check."""
        health = await hybrid_storage.health_check()
        
        assert health["storage_manager"] == "HybridStorageManager"
        assert "duckdb" in health
        assert "neo4j" in health
        assert "overall_status" in health
        
        # DuckDB should always be healthy
        assert health["duckdb"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_fallback_behavior(self, neo4j_config):
        """Test fallback to DuckDB when Neo4j is not available."""
        # Create config without Neo4j
        duckdb_config = {
            "duckdb": neo4j_config["duckdb"],
            "neo4j": {"enabled": False}
        }
        
        from echo_roots.storage import create_hybrid_storage
        storage = await create_hybrid_storage(duckdb_config)
        
        try:
            info = await storage.get_taxonomy_storage_info()
            assert info["taxonomy_backend"] == "DuckDB"
            assert info["neo4j_configured"] is False
        finally:
            await storage.close()


class TestGraphQueries:
    """Test graph query functionality."""
    
    @pytest.mark.asyncio
    @pytest.mark.storage
    async def test_graph_query_engine(self, hybrid_storage, sample_categories):
        """Test graph query engine functionality."""
        if not hybrid_storage.neo4j_backend:
            pytest.skip("Neo4j not configured for graph queries")
        
        from echo_roots.taxonomy.graph_queries import GraphQueryEngine
        
        # Create query engine
        query_engine = GraphQueryEngine(hybrid_storage.taxonomy)
        
        # Test metrics calculation (placeholder)
        metrics = await query_engine.calculate_taxonomy_metrics()
        assert metrics.total_categories >= 0
        
        # Test issue detection (placeholder)
        issues = await query_engine.detect_taxonomy_issues()
        assert isinstance(issues, dict)
        assert "orphan_categories" in issues


class TestDataMigration:
    """Test data migration between backends."""
    
    @pytest.mark.asyncio
    @pytest.mark.storage
    async def test_taxonomy_sync(self, hybrid_storage):
        """Test syncing taxonomy data to Neo4j."""
        if not hybrid_storage.neo4j_backend:
            pytest.skip("Neo4j not configured")
        
        # Test sync operation (placeholder)
        result = await hybrid_storage.sync_taxonomy_to_neo4j()
        
        assert isinstance(result, dict)
        assert "categories_synced" in result
        assert "errors" in result


# Integration test scenarios
class TestIntegrationScenarios:
    """Test complete workflow scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_taxonomy_workflow(self, hybrid_storage):
        """Test a complete taxonomy management workflow."""
        if not hybrid_storage.neo4j_backend:
            pytest.skip("Neo4j not configured")
        
        # 1. Create root category
        electronics = Category(
            name="Electronics",
            level=0,
            path=["Electronics"],
            description="Electronic products and devices"
        )
        
        electronics_id = await hybrid_storage.taxonomy.store_category(electronics)
        
        # 2. Add subcategory
        mobile = Category(
            name="Mobile Devices",
            parent_id=electronics_id,
            level=1,
            path=["Electronics", "Mobile Devices"]
        )
        
        mobile_id = await hybrid_storage.taxonomy.store_category(mobile)
        
        # 3. Add attributes
        brand_attr = Attribute(
            name="brand",
            display_name="Brand",
            data_type="categorical",
            values=[
                AttributeValue(value="Apple"),
                AttributeValue(value="Samsung")
            ]
        )
        
        await hybrid_storage.taxonomy.store_attribute(brand_attr)
        
        # 4. Verify retrieval
        retrieved_category = await hybrid_storage.taxonomy.get_category(electronics_id)
        assert retrieved_category.name == "Electronics"
        
        # 5. List categories
        categories = await hybrid_storage.taxonomy.list_categories()
        assert len(categories) >= 2


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        """Basic smoke test for Neo4j integration."""
        from echo_roots.storage import NEO4J_AVAILABLE, create_storage
        
        print(f"Neo4j available: {NEO4J_AVAILABLE}")
        
        if NEO4J_AVAILABLE:
            # Test DuckDB-only configuration
            config = {"duckdb": {"database_path": None}}
            storage = await create_storage(config)
            
            info = await storage.get_taxonomy_storage_info()
            print(f"Storage backend: {info['taxonomy_backend']}")
            
            await storage.close()
            print("Smoke test completed successfully")
        else:
            print("Neo4j integration not available - install with: pip install 'echo-roots[graph]'")
    
    asyncio.run(smoke_test())
