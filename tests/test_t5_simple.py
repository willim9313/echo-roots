"""
Simplified T5 Ingestion Pipeline Tests

Focus on core functionality without complex domain pack validation.
"""

import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from echo_roots.pipelines.ingestion import (
    IngestionConfig,
    IngestionStats,
    IngestionPipeline,
    BatchProcessor,
    StreamProcessor,
    PipelineCoordinator
)
from echo_roots.models.core import (
    IngestionItem, 
    ExtractionResult, 
    ProcessingStatus,
    ExtractionMetadata
)
from datetime import datetime


class TestBasicIngestionPipeline:
    """Test basic ingestion pipeline functionality without complex domain validation."""
    
    @pytest.mark.asyncio
    async def test_simple_pipeline_creation(self):
        """Test basic pipeline creation."""
        config = IngestionConfig(
            domain_name="test_domain",
            enable_storage=False
        )
        
        # Create pipeline with mock components
        pipeline = IngestionPipeline(config)
        
        # Should be created successfully
        assert pipeline.config == config
        assert pipeline.stats is not None
        assert pipeline.stats.total_items == 0
    
    @pytest.mark.asyncio 
    async def test_stats_tracking(self):
        """Test stats tracking functionality."""
        stats = IngestionStats()
        
        # Test initialization
        assert stats.total_items == 0
        assert stats.success_rate == 0.0
        
        # Test calculations
        stats.total_items = 10
        stats.processed_items = 10
        stats.successful_items = 8
        assert stats.success_rate == 80.0
        
        # Test quality scores
        stats.quality_scores = [0.8, 0.9, 0.7]
        assert abs(stats.average_quality - 0.8) < 0.001
    
    @pytest.mark.asyncio
    async def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = IngestionConfig(batch_size=50)
        assert config.batch_size == 50
        
        # Invalid config
        with pytest.raises(ValueError):
            IngestionConfig(batch_size=0)
    
    @pytest.mark.asyncio
    async def test_mock_processing_flow(self):
        """Test processing flow with mocked components."""
        config = IngestionConfig(
            domain_name="test_domain",
            enable_storage=False,
            batch_size=2
        )
        
        pipeline = IngestionPipeline(config)
        
        # Mock all dependencies
        mock_domain_pack = Mock()
        mock_domain_pack.name = "test_domain"
        
        mock_adapter = Mock()
        mock_item = IngestionItem(
            item_id="test_123",
            title="Test Product",
            description="Test description",
            raw_category="electronics",
            raw_attributes={},
            source="test",
            language="en",
            metadata={}
            # ingested_at will be auto-generated
        )
        mock_adapter.adapt_input.return_value = mock_item
        
        mock_extraction_pipeline = Mock()
        mock_result = ExtractionResult(
            item_id="test_123",
            attributes=[],
            terms=[],
            metadata=ExtractionMetadata(
                model="test-model",
                run_id="test-run-123",
                extracted_at=datetime.now(),
                processing_time_ms=100
            )
        )
        mock_extraction_pipeline.process_item = AsyncMock(return_value=mock_result)
        
        # Set mocked components
        pipeline._domain_pack = mock_domain_pack
        pipeline._domain_adapter = mock_adapter
        pipeline._extraction_pipeline = mock_extraction_pipeline
        
        # Test processing
        raw_data = {"name": "Test Product", "desc": "Test description"}
        result = await pipeline.process_item(raw_data, item_id="test_123")
        
        # Verify results
        assert result is not None
        assert isinstance(result, ExtractionResult)
        assert result.item_id == "test_123"
        assert pipeline.stats.successful_items == 1
        assert pipeline.stats.processed_items == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in pipeline."""
        config = IngestionConfig(
            domain_name="test_domain",
            enable_storage=False,
            skip_on_error=True
        )
        
        pipeline = IngestionPipeline(config)
        
        # Mock failing extraction
        mock_domain_pack = Mock()
        mock_adapter = Mock()
        mock_adapter.adapt_input.side_effect = Exception("Adapter failed")
        
        # Set components so initialization check passes
        pipeline._domain_pack = mock_domain_pack
        pipeline._domain_adapter = mock_adapter
        pipeline._extraction_pipeline = Mock()  # Add this so init check passes
        
        # Should handle error gracefully
        result = await pipeline.process_item({"test": "data"})
        
        assert result is None
        assert pipeline.stats.failed_items == 1
        assert len(pipeline.stats.errors) == 1
        assert pipeline.stats.errors[0]["error"] == "Adapter failed"
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing functionality."""
        config = IngestionConfig(
            domain_name="test_domain",
            enable_storage=False,
            batch_size=2
        )
        
        pipeline = IngestionPipeline(config)
        
        # Mock successful processing
        async def mock_process_item(raw_data, item_id=None):
            return ExtractionResult(
                item_id=item_id or "test",
                attributes=[],
                terms=[],
                metadata=ExtractionMetadata(
                    model="test-model",
                    run_id="test-run",
                    extracted_at=datetime.now()
                )
            )
        
        pipeline.process_item = mock_process_item
        
        # Test batch
        raw_data_items = [
            {"name": "Product 1"},
            {"name": "Product 2"},
            {"name": "Product 3"}
        ]
        
        results = await pipeline.process_batch(raw_data_items)
        
        assert len(results) == 3
        assert all(result is not None for result in results)
        assert pipeline.stats.total_items == 3


class TestBatchProcessorSimple:
    """Test BatchProcessor with simple file operations."""
    
    @pytest.mark.asyncio
    async def test_process_file_mock(self):
        """Test file processing with mocked pipeline."""
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process_batch = AsyncMock()
        mock_pipeline.stats = IngestionStats()
        mock_pipeline.stats.total_items = 2
        mock_pipeline.stats.successful_items = 2
        
        processor = BatchProcessor(mock_pipeline)
        
        # Create test file
        test_data = [
            {"name": "Product 1", "desc": "Description 1"},
            {"name": "Product 2", "desc": "Description 2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            stats = await processor.process_file(temp_path)
            
            # Verify file was processed
            mock_pipeline.process_batch.assert_called_once_with(test_data)
            assert stats.total_items == 2
            assert stats.successful_items == 2
        finally:
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self):
        """Test processing non-existent file."""
        mock_pipeline = Mock()
        processor = BatchProcessor(mock_pipeline)
        
        with pytest.raises(FileNotFoundError):
            await processor.process_file("nonexistent.json")


class TestStreamProcessorSimple:
    """Test StreamProcessor with simple async streams."""
    
    @pytest.mark.asyncio
    async def test_stream_processing_mock(self):
        """Test basic stream processing."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process_item = AsyncMock(return_value=Mock())
        mock_pipeline.config = IngestionConfig(skip_on_error=True)
        
        processor = StreamProcessor(mock_pipeline)
        
        # Simple async generator
        async def test_stream():
            data = [
                {"name": "Product 1"},
                {"name": "Product 2"}
            ]
            for item in data:
                yield item
        
        results = []
        def callback(result):
            results.append(result)
        
        # Process stream
        await processor.start_stream(test_stream(), callback)
        
        # Verify processing
        assert mock_pipeline.process_item.call_count == 2
        assert len(results) == 2


class TestPipelineCoordinatorSimple:
    """Test PipelineCoordinator basic functionality."""
    
    @pytest.mark.asyncio
    async def test_register_pipeline(self):
        """Test pipeline registration."""
        coordinator = PipelineCoordinator()
        
        config = IngestionConfig(domain_name="test_domain")
        coordinator.register_pipeline("test_pipeline", config)
        
        assert "test_pipeline" in coordinator.configs
        assert coordinator.configs["test_pipeline"] == config
    
    @pytest.mark.asyncio
    async def test_unregistered_pipeline_error(self):
        """Test error for unregistered pipeline."""
        coordinator = PipelineCoordinator()
        
        with pytest.raises(ValueError, match="No configuration registered"):
            await coordinator.get_pipeline("nonexistent")
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        coordinator = PipelineCoordinator()
        
        # Add mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.cleanup = AsyncMock()
        coordinator.pipelines["test"] = mock_pipeline
        
        # Cleanup
        await coordinator.cleanup_all()
        
        # Verify cleanup called and pipelines cleared
        mock_pipeline.cleanup.assert_called_once()
        assert len(coordinator.pipelines) == 0


@pytest.mark.asyncio
async def test_end_to_end_mock():
    """Test end-to-end workflow with mocked components."""
    # Configuration
    config = IngestionConfig(
        domain_name="test_domain",
        batch_size=2,
        enable_storage=False
    )
    
    # Create pipeline
    pipeline = IngestionPipeline(config)
    
    # Mock all dependencies for successful processing
    mock_domain_pack = Mock()
    mock_domain_pack.name = "test_domain"
    
    mock_adapter = Mock()
    def mock_adapt(raw_data, item_id=None):
        return IngestionItem(
            item_id=item_id or "generated_id",
            title=raw_data.get("name", "Unknown"),
            description=raw_data.get("desc", ""),
            raw_category="test",
            raw_attributes=raw_data,
            source="test",
            language="en",
            metadata={}
            # ingested_at will be auto-generated
        )
    mock_adapter.adapt_input.side_effect = mock_adapt
    
    mock_extraction_pipeline = Mock()
    def mock_process(item, validate=False):
        return ExtractionResult(
            item_id=item.item_id,
            attributes=[],
            terms=[],
            metadata=ExtractionMetadata(
                model="test-model",
                run_id="test-run",
                extracted_at=datetime.now()
            )
        )
    mock_extraction_pipeline.process_item = AsyncMock(side_effect=mock_process)
    
    # Set up pipeline
    pipeline._domain_pack = mock_domain_pack
    pipeline._domain_adapter = mock_adapter
    pipeline._extraction_pipeline = mock_extraction_pipeline
    
    # Test data
    raw_data_items = [
        {"name": "iPhone 15 Pro", "desc": "Latest smartphone", "price": "$999"},
        {"name": "MacBook Air", "desc": "Laptop with M2 chip", "price": "$1199"}
    ]
    
    # Process batch
    results = await pipeline.process_batch(raw_data_items)
    
    # Verify results
    assert len(results) == 2
    assert all(result is not None for result in results)
    assert all(isinstance(result, ExtractionResult) for result in results)
    
    # Verify stats
    assert pipeline.stats.total_items == 2
    assert pipeline.stats.successful_items == 2
    assert pipeline.stats.success_rate == 100.0
    assert len(pipeline.stats.quality_scores) == 2
    assert all(score == 0.8 for score in pipeline.stats.quality_scores)
