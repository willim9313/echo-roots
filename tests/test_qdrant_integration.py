"""
Comprehensive integration tests for Qdrant vector storage.

Tests the complete integration between Echo Roots semantic processing
and Qdrant vector database including embedding generation, storage,
retrieval, and search operations.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, UTC
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from echo_roots.models.taxonomy import SemanticCandidate
from echo_roots.semantic import SemanticEmbedding
from echo_roots.storage.qdrant_backend import QdrantBackend, QdrantSemanticRepository
from echo_roots.semantic.embedding_providers import (
    SentenceTransformersProvider, EmbeddingProviderFactory
)
from echo_roots.semantic.pipeline import SemanticProcessingPipeline
from echo_roots.semantic.search import SemanticSearchEngine, SearchConfiguration


@pytest.fixture
def qdrant_config():
    """Qdrant configuration for testing."""
    return {
        "host": "localhost",
        "port": 6333,
        "prefer_grpc": False,
        "timeout": 10.0,
        "collections": {
            "test_collection": {
                "vector_size": 384,
                "distance": "Cosine"
            }
        }
    }


@pytest.fixture
def sample_candidates():
    """Sample semantic candidates for testing."""
    return [
        SemanticCandidate(
            candidate_id="cand_1",
            term="smartphone",
            normalized_term="smartphone",
            frequency=100,
            contexts=["mobile device for communication"],
            score=0.95,
            language="en",
            domain="electronics"
        ),
        SemanticCandidate(
            candidate_id="cand_2",
            term="mobile phone",
            normalized_term="mobile phone",
            frequency=80,
            contexts=["portable telephone"],
            score=0.90,
            language="en",
            domain="electronics"
        ),
        SemanticCandidate(
            candidate_id="cand_3",
            term="laptop computer",
            normalized_term="laptop computer",
            frequency=75,
            contexts=["portable personal computer"],
            score=0.88,
            language="en",
            domain="electronics"
        )
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        SemanticEmbedding(
            embedding_id="emb_1",
            entity_id="cand_1",
            entity_type="semantic_candidate",
            embedding_vector=[0.1] * 384,
            model_name="test_model",
            model_version="1.0",
            dimensions=384,
            metadata={"term": "smartphone", "domain": "electronics"}
        ),
        SemanticEmbedding(
            embedding_id="emb_2",
            entity_id="cand_2",
            entity_type="semantic_candidate",
            embedding_vector=[0.2] * 384,
            model_name="test_model",
            model_version="1.0",
            dimensions=384,
            metadata={"term": "mobile phone", "domain": "electronics"}
        )
    ]


class TestQdrantBackend:
    """Test Qdrant backend functionality."""
    
    @pytest.mark.asyncio
    async def test_qdrant_backend_initialization(self, qdrant_config):
        """Test Qdrant backend initialization."""
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            mock_client.return_value.get_collections.return_value = MagicMock()
            
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            assert backend.client is not None
            assert backend.config == qdrant_config
            mock_client.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collection_creation(self, qdrant_config):
        """Test collection creation and management."""
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.create_collection.return_value = True
            
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            # Test collection creation
            result = await backend.create_collection("test_collection", vector_size=384)
            assert result is True
            
            mock_instance.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_storage_and_retrieval(self, qdrant_config, sample_embeddings):
        """Test embedding storage and retrieval."""
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            
            # Mock successful storage
            mock_instance.upsert.return_value = MagicMock(status="completed")
            
            # Mock retrieval
            mock_point = MagicMock()
            mock_point.id = "emb_1"
            mock_point.payload = sample_embeddings[0].metadata
            mock_point.vector = sample_embeddings[0].embedding_vector
            mock_instance.retrieve.return_value = [mock_point]
            
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            # Test storage
            repository = backend.semantic_repository
            result = await repository.store_embedding(sample_embeddings[0])
            assert result == "emb_1"
            
            # Test retrieval
            retrieved = await repository.get_embedding("emb_1")
            assert retrieved is not None
            assert retrieved.embedding_id == "emb_1"
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, qdrant_config, sample_embeddings):
        """Test vector similarity search."""
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            
            # Mock search results
            mock_result = MagicMock()
            mock_result.id = "emb_1"
            mock_result.score = 0.95
            mock_result.payload = sample_embeddings[0].metadata
            mock_result.vector = sample_embeddings[0].embedding_vector
            
            mock_instance.search.return_value = [mock_result]
            
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            repository = backend.semantic_repository
            
            # Test similarity search
            query_vector = [0.1] * 384
            results = await repository.find_similar_embeddings(
                query_vector=query_vector,
                limit=10,
                threshold=0.8
            )
            
            assert len(results) == 1
            embedding, similarity = results[0]
            assert embedding.embedding_id == "emb_1"
            assert similarity == 0.95
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, qdrant_config, sample_embeddings):
        """Test batch storage and operations."""
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.upsert.return_value = MagicMock(status="completed")
            
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            repository = backend.semantic_repository
            
            # Test batch storage
            embedding_ids = await repository.batch_store_embeddings(sample_embeddings)
            
            assert len(embedding_ids) == len(sample_embeddings)
            assert all(isinstance(eid, str) for eid in embedding_ids)
    
    @pytest.mark.asyncio
    async def test_health_check(self, qdrant_config):
        """Test health check functionality."""
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.get_collections.return_value = MagicMock()
            
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            # Test health check
            is_healthy = await backend.health_check()
            assert is_healthy is True
            
            # Test health check with exception
            mock_instance.get_collections.side_effect = Exception("Connection failed")
            is_healthy = await backend.health_check()
            assert is_healthy is False


class TestEmbeddingProviders:
    """Test embedding provider implementations."""
    
    @pytest.mark.asyncio
    async def test_sentence_transformers_provider(self):
        """Test SentenceTransformers embedding provider."""
        config = {
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu"
        }
        
        with patch('sentence_transformers.SentenceTransformer') as mock_model:
            # Mock model
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            mock_instance.encode.return_value = [[0.1] * 384, [0.2] * 384]
            mock_instance.similarity.return_value = [[0.95]]
            
            provider = SentenceTransformersProvider(config)
            await provider.initialize()
            
            # Test single embedding
            embedding = await provider.generate_embedding("test text")
            assert len(embedding) == 384
            
            # Test batch embeddings
            texts = ["text1", "text2"]
            embeddings = await provider.generate_batch_embeddings(texts)
            assert len(embeddings) == 2
            assert all(len(emb) == 384 for emb in embeddings)
            
            # Test similarity calculation
            similarity = await provider.calculate_similarity([0.1] * 384, [0.2] * 384)
            assert isinstance(similarity, float)
    
    @pytest.mark.asyncio
    async def test_embedding_provider_factory(self):
        """Test embedding provider factory."""
        # Test SentenceTransformers provider creation
        config = {"model_name": "all-MiniLM-L6-v2"}
        provider = EmbeddingProviderFactory.create_provider("sentence_transformers", config)
        assert isinstance(provider, SentenceTransformersProvider)
        
        # Test default provider creation
        default_provider = EmbeddingProviderFactory.create_default_provider()
        assert default_provider is not None
        
        # Test invalid provider type
        with pytest.raises(ValueError):
            EmbeddingProviderFactory.create_provider("invalid_type", {})


class TestSemanticPipeline:
    """Test semantic processing pipeline."""
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider for testing."""
        provider = AsyncMock()
        provider.generate_embedding.return_value = [0.1] * 384
        provider.generate_batch_embeddings.return_value = [[0.1] * 384, [0.2] * 384]
        provider.model_name = "test_model"
        return provider
    
    @pytest.fixture
    def mock_vector_repository(self):
        """Mock vector repository for testing."""
        repository = AsyncMock()
        repository.batch_store_embeddings.return_value = ["emb_1", "emb_2"]
        return repository
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, mock_embedding_provider, mock_vector_repository):
        """Test pipeline initialization."""
        pipeline = SemanticProcessingPipeline(
            embedding_provider=mock_embedding_provider,
            vector_repository=mock_vector_repository
        )
        
        await pipeline.initialize()
        mock_embedding_provider.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_candidate_processing(self, mock_embedding_provider, mock_vector_repository, sample_candidates):
        """Test semantic candidate processing."""
        pipeline = SemanticProcessingPipeline(
            embedding_provider=mock_embedding_provider,
            vector_repository=mock_vector_repository,
            batch_size=2
        )
        
        await pipeline.initialize()
        
        # Process candidates
        embedding_ids = await pipeline.process_semantic_candidates(sample_candidates)
        
        # Verify results
        assert len(embedding_ids) > 0
        mock_embedding_provider.generate_batch_embeddings.assert_called()
        mock_vector_repository.batch_store_embeddings.assert_called()
    
    @pytest.mark.asyncio
    async def test_text_embedding_generation(self, mock_embedding_provider, mock_vector_repository):
        """Test direct text embedding generation."""
        pipeline = SemanticProcessingPipeline(
            embedding_provider=mock_embedding_provider,
            vector_repository=mock_vector_repository
        )
        
        await pipeline.initialize()
        
        texts = ["smartphone", "mobile phone"]
        embeddings = await pipeline.generate_embeddings_for_texts(texts)
        
        assert len(embeddings) == 2
        assert all(emb.embedding_vector is not None for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_similarity_matrix_calculation(self, mock_embedding_provider, sample_candidates):
        """Test similarity matrix calculation."""
        mock_embedding_provider.calculate_similarity.return_value = 0.85
        
        pipeline = SemanticProcessingPipeline(
            embedding_provider=mock_embedding_provider
        )
        
        await pipeline.initialize()
        
        similarity_matrix = await pipeline.calculate_similarity_matrix(sample_candidates[:2])
        
        assert len(similarity_matrix) == 2
        assert len(similarity_matrix[0]) == 2
        assert similarity_matrix[0][0] == 1.0  # Self-similarity
    
    @pytest.mark.asyncio
    async def test_pipeline_statistics(self, mock_embedding_provider, mock_vector_repository, sample_candidates):
        """Test pipeline statistics tracking."""
        pipeline = SemanticProcessingPipeline(
            embedding_provider=mock_embedding_provider,
            vector_repository=mock_vector_repository
        )
        
        await pipeline.initialize()
        await pipeline.process_semantic_candidates(sample_candidates)
        
        stats = pipeline.get_statistics()
        
        assert stats["candidates_processed"] == len(sample_candidates)
        assert stats["embeddings_generated"] > 0
        assert "duration_seconds" in stats or stats["start_time"] is not None


class TestSemanticSearchEngine:
    """Test semantic search engine integration."""
    
    @pytest.fixture
    def mock_repository(self):
        """Mock semantic repository for testing."""
        repository = AsyncMock()
        return repository
    
    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider for testing."""
        provider = AsyncMock()
        provider.generate_embedding.return_value = [0.1] * 384
        return provider
    
    @pytest.fixture
    def mock_vector_repository(self):
        """Mock vector repository for testing."""
        repository = AsyncMock()
        
        # Mock similarity search results
        from echo_roots.semantic import SemanticEmbedding
        mock_embedding = SemanticEmbedding(
            embedding_id="emb_1",
            entity_id="entity_1",
            entity_type="semantic_candidate",
            embedding_vector=[0.1] * 384,
            model_name="test",
            model_version="1.0",
            dimensions=384,
            metadata={"source_text": "smartphone"}
        )
        repository.find_similar_embeddings.return_value = [(mock_embedding, 0.95)]
        
        return repository
    
    @pytest.mark.asyncio
    async def test_search_engine_initialization(self, mock_repository, mock_embedding_provider, mock_vector_repository):
        """Test search engine initialization."""
        engine = SemanticSearchEngine(
            repository=mock_repository,
            embedding_provider=mock_embedding_provider,
            vector_repository=mock_vector_repository
        )
        
        assert engine.repository is mock_repository
        assert engine.embedding_provider is mock_embedding_provider
        assert engine.vector_repository is mock_vector_repository
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search(self, mock_repository, mock_embedding_provider, mock_vector_repository):
        """Test vector similarity search."""
        from echo_roots.semantic import SemanticQuery
        
        engine = SemanticSearchEngine(
            repository=mock_repository,
            embedding_provider=mock_embedding_provider,
            vector_repository=mock_vector_repository
        )
        
        query = SemanticQuery(
            query_text="smartphone",
            target_entity_types=["semantic_candidate"]
        )
        
        config = SearchConfiguration()
        results, metrics = await engine.search(query, config)
        
        assert len(results) > 0
        assert results[0].entity_id == "entity_1"
        assert metrics.strategy_used == config.strategy
        
        # Verify Qdrant was used preferentially
        mock_vector_repository.find_similar_embeddings.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_fallback_to_repository(self, mock_repository, mock_embedding_provider):
        """Test search fallback when Qdrant is not available."""
        from echo_roots.semantic import SemanticQuery
        
        # Mock repository similarity search
        from echo_roots.semantic import SemanticEmbedding
        mock_embedding = SemanticEmbedding(
            embedding_id="emb_1",
            entity_id="entity_1",
            entity_type="semantic_candidate",
            embedding_vector=[0.1] * 384,
            model_name="test",
            model_version="1.0",
            dimensions=384,
            metadata={"source_text": "smartphone"}
        )
        mock_repository.find_similar_embeddings.return_value = [(mock_embedding, 0.90)]
        
        engine = SemanticSearchEngine(
            repository=mock_repository,
            embedding_provider=mock_embedding_provider,
            vector_repository=None  # No Qdrant
        )
        
        query = SemanticQuery(
            query_text="smartphone",
            target_entity_types=["semantic_candidate"]
        )
        
        results, metrics = await engine.search(query)
        
        assert len(results) > 0
        mock_repository.find_similar_embeddings.assert_called_once()


class TestIntegrationComplete:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_integration(self, sample_candidates):
        """Test complete pipeline from candidates to search."""
        # This would be an integration test that requires actual Qdrant instance
        # For now, we'll test with mocks but structure for real integration
        
        with patch('qdrant_client.AsyncQdrantClient') as mock_client:
            # Setup mocks for complete flow
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.get_collections.return_value = MagicMock(collections=[])
            mock_instance.create_collection.return_value = True
            mock_instance.upsert.return_value = MagicMock(status="completed")
            
            # Mock search results
            mock_result = MagicMock()
            mock_result.id = "emb_1"
            mock_result.score = 0.95
            mock_result.payload = {"source_text": "smartphone", "domain": "electronics"}
            mock_result.vector = [0.1] * 384
            mock_instance.search.return_value = [mock_result]
            
            # Mock embedding generation
            with patch('sentence_transformers.SentenceTransformer') as mock_st:
                mock_st_instance = MagicMock()
                mock_st.return_value = mock_st_instance
                mock_st_instance.encode.return_value = [[0.1] * 384]
                
                # Initialize components
                qdrant_config = {
                    "host": "localhost",
                    "port": 6333,
                    "collections": {
                        "semantic_candidates": {"vector_size": 384, "distance": "Cosine"}
                    }
                }
                
                # Test complete flow
                backend = QdrantBackend(qdrant_config)
                await backend.initialize()
                
                # Create embedding provider
                embedding_config = {"model_name": "all-MiniLM-L6-v2", "device": "cpu"}
                provider = SentenceTransformersProvider(embedding_config)
                await provider.initialize()
                
                # Create pipeline
                pipeline = SemanticProcessingPipeline(
                    embedding_provider=provider,
                    vector_repository=backend.semantic_repository
                )
                await pipeline.initialize()
                
                # Process candidates
                embedding_ids = await pipeline.process_semantic_candidates(sample_candidates)
                assert len(embedding_ids) > 0
                
                # Create search engine and test search
                engine = SemanticSearchEngine(
                    repository=AsyncMock(),  # Mock repository
                    embedding_provider=provider,
                    vector_repository=backend.semantic_repository
                )
                
                from echo_roots.semantic import SemanticQuery
                query = SemanticQuery(query_text="mobile device")
                
                results, metrics = await engine.search(query)
                assert len(results) > 0
                assert metrics.query_time_ms >= 0


@pytest.mark.integration
class TestQdrantIntegrationLive:
    """Live integration tests requiring actual Qdrant instance."""
    
    @pytest.mark.skipif(
        not pytest.qdrant_available,
        reason="Qdrant instance not available for integration tests"
    )
    @pytest.mark.asyncio
    async def test_live_qdrant_integration(self, sample_candidates):
        """Test with actual Qdrant instance."""
        # This test would run only when Qdrant is available
        # pytest --qdrant-url=http://localhost:6333
        
        qdrant_config = {
            "host": "localhost",
            "port": 6333,
            "collections": {
                "test_semantic_candidates": {
                    "vector_size": 384,
                    "distance": "Cosine"
                }
            }
        }
        
        try:
            # Test actual Qdrant operations
            backend = QdrantBackend(qdrant_config)
            await backend.initialize()
            
            # Create test collection
            await backend.create_collection("test_semantic_candidates", vector_size=384)
            
            # Test health check
            assert await backend.health_check()
            
            # Create embedding provider
            embedding_config = {"model_name": "all-MiniLM-L6-v2", "device": "cpu"}
            provider = SentenceTransformersProvider(embedding_config)
            await provider.initialize()
            
            # Process candidates
            pipeline = SemanticProcessingPipeline(
                embedding_provider=provider,
                vector_repository=backend.semantic_repository
            )
            await pipeline.initialize()
            
            embedding_ids = await pipeline.process_semantic_candidates(sample_candidates)
            assert len(embedding_ids) == len(sample_candidates)
            
            # Test search
            engine = SemanticSearchEngine(
                repository=AsyncMock(),
                embedding_provider=provider,
                vector_repository=backend.semantic_repository
            )
            
            from echo_roots.semantic import SemanticQuery
            query = SemanticQuery(query_text="phone")
            results, metrics = await engine.search(query)
            
            assert len(results) > 0
            assert any("phone" in r.entity_text.lower() for r in results)
            
        finally:
            # Cleanup test collection
            try:
                await backend.delete_collection("test_semantic_candidates")
            except:
                pass  # Ignore cleanup errors


def pytest_configure(config):
    """Configure pytest with custom markers and options."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    
    # Check if Qdrant is available for integration tests
    qdrant_url = config.getoption("--qdrant-url", default=None)
    pytest.qdrant_available = qdrant_url is not None


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--qdrant-url",
        action="store",
        default=None,
        help="Qdrant URL for integration tests"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
