"""
Tests for T5 Ingestion Pipeline

Tests the complete end-to-end data flow from raw input through
domain adaptation, LLM extraction, and storage.
"""

import pytest
import pytest_asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock, patch
import asyncio

from echo_roots.pipelines.ingestion import (
    IngestionConfig,
    IngestionStats,
    IngestionPipeline,
    BatchProcessor,
    StreamProcessor,
    PipelineCoordinator
)
from echo_roots.models.core import IngestionItem, ExtractionResult, ProcessingStatus
from echo_roots.models.domain import DomainPack
from echo_roots.pipelines.openai_client import MockLLMClient
from echo_roots.storage.duckdb_backend import DuckDBBackend


class TestIngestionConfig:
    """Test IngestionConfig validation and setup."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IngestionConfig()
        
        assert config.domain_pack_path == "domains/"
        assert config.domain_name == "ecommerce"
        assert config.batch_size == 100
        assert config.max_concurrent == 10
        assert config.enable_validation is True
        assert config.skip_on_error is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = IngestionConfig(batch_size=50, max_concurrent=5)
        assert config.batch_size == 50
        assert config.max_concurrent == 5
        
        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            IngestionConfig(batch_size=0)
        
        # Invalid max_concurrent
        with pytest.raises(ValueError, match="max_concurrent must be positive"):
            IngestionConfig(max_concurrent=-1)
        
        # Invalid quality score
        with pytest.raises(ValueError, match="min_quality_score must be between 0 and 1"):
            IngestionConfig(min_quality_score=1.5)


class TestIngestionStats:
    """Test IngestionStats tracking and calculations."""
    
    def test_stats_initialization(self):
        """Test stats default values."""
        stats = IngestionStats()
        
        assert stats.total_items == 0
        assert stats.processed_items == 0
        assert stats.successful_items == 0
        assert stats.failed_items == 0
        assert stats.success_rate == 0.0
        assert stats.average_quality == 0.0
        assert stats.processing_time == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = IngestionStats()
        
        # No items processed
        assert stats.success_rate == 0.0
        
        # Some items processed
        stats.processed_items = 10
        stats.successful_items = 8
        assert stats.success_rate == 80.0
    
    def test_average_quality_calculation(self):
        """Test average quality calculation."""
        stats = IngestionStats()
        
        # No quality scores
        assert stats.average_quality == 0.0
        
        # With quality scores
        stats.quality_scores = [0.8, 0.9, 0.7]
        assert abs(stats.average_quality - 0.8) < 0.001  # Allow floating point precision
    
    def test_error_tracking(self):
        """Test error tracking functionality."""
        stats = IngestionStats()
        
        # Add an error
        error = ValueError("Test error")
        stats.add_error("item_123", error, "test_context")
        
        assert len(stats.errors) == 1
        error_record = stats.errors[0]
        assert error_record["item_id"] == "item_123"
        assert error_record["error"] == "Test error"
        assert error_record["error_type"] == "ValueError"
        assert error_record["context"] == "test_context"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = IngestionStats()
        stats.total_items = 10
        stats.successful_items = 8
        stats.processed_items = 10
        stats.quality_scores = [0.8, 0.9]
        
        result = stats.to_dict()
        
        assert result["total_items"] == 10
        assert result["successful_items"] == 8
        assert result["success_rate"] == 80.0
        assert abs(result["average_quality"] - 0.85) < 0.001  # Allow floating point precision


class TestIngestionPipeline:
    """Test main IngestionPipeline functionality."""
    
    @pytest_asyncio.fixture
    async def sample_domain_pack(self):
        """Create a sample domain pack for testing."""
        return DomainPack(
            domain="test_domain",
            taxonomy_version="1.0.0",
            input_mapping={
                "title": "name",
                "description": "desc"
            },
            output_schema={
                "core_item": {
                    "title": {"type": "string"},
                    "description": {"type": "string"}
                },
                "attributes": ["brand", "price", "category"]
            },
            attribute_hints={
                "brand": {"type": "string", "required": False},
                "price": {"type": "number", "required": False},
                "category": {"type": "string", "required": False}
            }
        )
    
    @pytest_asyncio.fixture
    async def mock_storage_backend(self):
        """Create a mock storage backend."""
        backend = Mock()
        backend.initialize = AsyncMock()
        backend.cleanup = AsyncMock()
        return backend
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return IngestionConfig(
            domain_name="test_domain",
            batch_size=2,
            max_concurrent=2,
            enable_storage=False,  # Disable storage for basic tests
            skip_on_error=True
        )
    
    @pytest_asyncio.fixture
    async def pipeline(self, pipeline_config, mock_storage_backend):
        """Create configured pipeline for testing."""
        llm_client = MockLLMClient()
        pipeline = IngestionPipeline(
            pipeline_config,
            llm_client=llm_client,
            storage_backend=mock_storage_backend
        )
        
        # Mock the domain loading
        with patch.object(pipeline, '_domain_loader') as mock_loader:
            mock_domain_pack = DomainPack(
                domain="test_domain",
                taxonomy_version="1.0.0",
                input_mapping={"title": "name", "description": "desc"},
                output_schema={
                    "core_item": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "attributes": ["brand", "price"]
                },
                attribute_hints={
                    "brand": {"type": "string", "required": False},
                    "price": {"type": "number", "required": False}
                }
            )
            mock_loader.load_domain_pack.return_value = mock_domain_pack
            
            await pipeline.initialize()
        
        return pipeline
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline._domain_pack is not None
        assert pipeline._domain_adapter is not None
        assert pipeline._extraction_pipeline is not None
        assert pipeline.stats.total_items == 0
    
    @pytest.mark.asyncio
    async def test_process_single_item(self, pipeline):
        """Test processing a single raw data item."""
        raw_data = {
            "name": "Test Product",
            "desc": "Test description",
            "price": "$29.99"
        }
        
        result = await pipeline.process_item(raw_data, item_id="test_123")
        
        assert result is not None
        assert isinstance(result, ExtractionResult)
        assert pipeline.stats.processed_items == 1
        assert pipeline.stats.successful_items == 1
        assert pipeline.stats.failed_items == 0
    
    @pytest.mark.asyncio
    async def test_process_batch(self, pipeline):
        """Test processing multiple items in a batch."""
        raw_data_items = [
            {"name": "Product 1", "desc": "Description 1"},
            {"name": "Product 2", "desc": "Description 2"},
            {"name": "Product 3", "desc": "Description 3"}
        ]
        
        results = await pipeline.process_batch(raw_data_items)
        
        assert len(results) == 3
        assert all(result is not None for result in results)
        assert pipeline.stats.total_items == 3
        assert pipeline.stats.successful_items == 3
        assert pipeline.stats.success_rate == 100.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in pipeline."""
        # Create configuration
        pipeline_config = IngestionConfig(
            domain_name="test_domain",
            batch_size=2,
            max_concurrent=2,
            enable_storage=False,
            skip_on_error=True
        )
        
        # Create pipeline with failing LLM client
        failing_client = Mock()
        failing_client.extract_structured_data = AsyncMock(
            side_effect=Exception("LLM error")
        )
        
        pipeline = IngestionPipeline(pipeline_config, llm_client=failing_client)
        
        # Mock domain loading
        with patch.object(pipeline, '_domain_loader') as mock_loader:
            mock_domain_pack = DomainPack(
                domain="test_domain",
                taxonomy_version="1.0.0",
                input_mapping={"title": "name"},
                output_schema={
                    "core_item": {"title": {"type": "string"}},
                    "attributes": ["brand"]
                },
                attribute_hints={"brand": {"type": "string", "required": False}}
            )
            mock_loader.load_domain_pack.return_value = mock_domain_pack
            await pipeline.initialize()
        
        raw_data = {"name": "Test Product"}
        
        # Should handle error gracefully with skip_on_error=True
        result = await pipeline.process_item(raw_data, item_id="test_fail")
        
        assert result is None
        assert pipeline.stats.failed_items == 1
        assert len(pipeline.stats.errors) == 1
    
    @pytest.mark.asyncio
    async def test_quality_control(self, pipeline):
        """Test quality control functionality."""
        # Configure strict quality requirements
        pipeline.config.min_quality_score = 0.9
        pipeline.config.enable_validation = True
        
        raw_data = {"name": "Low Quality Product", "desc": "Poor description"}
        
        # Mock low quality result
        with patch.object(pipeline._extraction_pipeline, 'process_item') as mock_process:
            mock_result = ExtractionResult(
                item_id="test_quality",
                attributes=[],
                terms=[],
                metadata={"quality_score": 0.5}  # Below threshold
            )
            mock_process.return_value = mock_result
            
            result = await pipeline.process_item(raw_data, item_id="test_quality")
            
            # Should be skipped due to low quality
            assert result is None
            assert pipeline.stats.skipped_items == 1


class TestBatchProcessor:
    """Test BatchProcessor functionality."""
    
    @pytest_asyncio.fixture
    async def batch_processor(self, pipeline):
        """Create batch processor for testing."""
        return BatchProcessor(pipeline)
    
    @pytest.mark.asyncio
    async def test_process_file(self, batch_processor):
        """Test processing a JSON file."""
        # Create temporary test file
        test_data = [
            {"name": "Product 1", "desc": "Description 1"},
            {"name": "Product 2", "desc": "Description 2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            stats = await batch_processor.process_file(temp_path)
            
            assert stats.total_items == 2
            assert stats.successful_items == 2
        finally:
            Path(temp_path).unlink()  # Clean up
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, batch_processor):
        """Test processing non-existent file."""
        with pytest.raises(FileNotFoundError):
            await batch_processor.process_file("nonexistent.json")
    
    @pytest.mark.asyncio
    async def test_process_invalid_json(self, batch_processor):
        """Test processing invalid JSON structure."""
        # Create file with object instead of array
        test_data = {"name": "Single Product"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="JSON file must contain an array"):
                await batch_processor.process_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestStreamProcessor:
    """Test StreamProcessor functionality."""
    
    @pytest_asyncio.fixture
    async def stream_processor(self, pipeline):
        """Create stream processor for testing."""
        return StreamProcessor(pipeline)
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, stream_processor):
        """Test basic stream processing."""
        # Create async generator for test data
        async def test_stream():
            test_data = [
                {"name": "Product 1", "desc": "Stream item 1"},
                {"name": "Product 2", "desc": "Stream item 2"}
            ]
            for item in test_data:
                yield item
        
        results = []
        
        def callback(result):
            results.append(result)
        
        # Process stream
        await stream_processor.start_stream(test_stream(), callback)
        
        assert len(results) == 2
        assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_stream_stop(self, stream_processor):
        """Test stopping stream processing."""
        async def infinite_stream():
            counter = 0
            while True:
                yield {"name": f"Product {counter}", "desc": f"Item {counter}"}
                counter += 1
                # Add small delay to allow stop
                await asyncio.sleep(0.01)
        
        # Start processing and stop after short delay
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stream_processor.stop_stream()
        
        # Run both coroutines
        await asyncio.gather(
            stream_processor.start_stream(infinite_stream()),
            stop_after_delay()
        )
        
        # Should complete without error
        assert not stream_processor._running


class TestPipelineCoordinator:
    """Test PipelineCoordinator functionality."""
    
    @pytest_asyncio.fixture
    async def coordinator(self):
        """Create pipeline coordinator for testing."""
        return PipelineCoordinator()
    
    @pytest.mark.asyncio
    async def test_register_and_get_pipeline(self, coordinator):
        """Test registering and retrieving pipelines."""
        config = IngestionConfig(
            domain_name="test_domain",
            enable_storage=False
        )
        
        # Register pipeline
        coordinator.register_pipeline("test_pipeline", config)
        
        assert "test_pipeline" in coordinator.configs
        
        # Mock domain loading for pipeline creation
        with patch('echo_roots.pipelines.ingestion.DomainPackLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_domain_pack.return_value = DomainPack(
                name="test_domain",
                version="1.0.0",
                description="Test",
                schema_mapping={},
                attribute_hints=[],
                output_schema={}
            )
            mock_loader_class.return_value = mock_loader
            
            # Get pipeline (should create and initialize)
            pipeline = await coordinator.get_pipeline("test_pipeline")
            
            assert isinstance(pipeline, IngestionPipeline)
            assert "test_pipeline" in coordinator.pipelines
    
    @pytest.mark.asyncio
    async def test_unregistered_pipeline(self, coordinator):
        """Test accessing unregistered pipeline."""
        with pytest.raises(ValueError, match="No configuration registered"):
            await coordinator.get_pipeline("nonexistent")
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self, coordinator):
        """Test cleanup of all pipelines."""
        # Add mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.cleanup = AsyncMock()
        coordinator.pipelines["test"] = mock_pipeline
        
        await coordinator.cleanup_all()
        
        assert len(coordinator.pipelines) == 0
        mock_pipeline.cleanup.assert_called_once()


class TestIntegration:
    """Integration tests for complete ingestion workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self):
        """Test complete end-to-end ingestion workflow."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Configure pipeline with real storage
            config = IngestionConfig(
                domain_name="test_domain",
                storage_path=db_path,
                enable_storage=True,
                batch_size=2
            )
            
            # Create pipeline with mock LLM
            llm_client = MockLLMClient()
            storage_backend = DuckDBBackend(db_path)
            
            pipeline = IngestionPipeline(
                config,
                llm_client=llm_client,
                storage_backend=storage_backend
            )
            
            # Mock domain loading
            with patch.object(pipeline, '_domain_loader') as mock_loader:
                mock_domain_pack = DomainPack(
                    domain="test_domain",
                    taxonomy_version="1.0.0",
                    input_mapping={"title": "name", "description": "desc"},
                    output_schema={
                        "core_item": {
                            "title": {"type": "string"},
                            "description": {"type": "string"}
                        },
                        "attributes": ["brand", "price", "category"]
                    },
                    attribute_hints={
                        "brand": {"type": "string", "required": False},
                        "price": {"type": "number", "required": False},
                        "category": {"type": "string", "required": False}
                    }
                )
                mock_loader.load_domain_pack.return_value = mock_domain_pack
                
                await pipeline.initialize()
                
                # Process test data
                raw_data_items = [
                    {
                        "name": "iPhone 15 Pro",
                        "desc": "Latest smartphone with advanced camera",
                        "price": "$999"
                    },
                    {
                        "name": "MacBook Air",
                        "desc": "Thin and light laptop with M2 chip",
                        "price": "$1199"
                    }
                ]
                
                results = await pipeline.process_batch(raw_data_items)
                
                # Verify results
                assert len(results) == 2
                assert all(result is not None for result in results)
                assert pipeline.stats.success_rate == 100.0
                
                # Verify storage (items should be stored)
                stored_items = await pipeline._storage.storage.ingestion.list_items()
                assert len(stored_items) == 2
                
                await pipeline.cleanup()
        
        finally:
            # Clean up temporary file
            Path(db_path).unlink(missing_ok=True)
