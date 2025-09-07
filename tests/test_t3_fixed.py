"""Fixed T3 tests with correct constructor usage.

This test file uses the actual T3 implementation with proper model integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from echo_roots.models.core import (
    IngestionItem, ExtractionResult, AttributeExtraction, SemanticTerm, ExtractionMetadata
)
from echo_roots.pipelines.extraction import ExtractorConfig
from echo_roots.pipelines.openai_client import MockLLMClient


@pytest.fixture
def sample_ingestion_item():
    """Create a sample ingestion item with correct field names."""
    return IngestionItem(
        item_id="test_item_1",
        title="iPhone 15 Pro",
        description="Latest smartphone with advanced camera system",
        source="test_catalog",
        metadata={"url": "https://example.com/iphone15pro"}
    )


class TestExtractorConfig:
    """Test ExtractorConfig functionality."""
    
    def test_config_creation(self):
        """Test creating ExtractorConfig with valid parameters."""
        config = ExtractorConfig(
            model_name="gpt-4",
            temperature=0.3,
            max_tokens=2000,
            timeout_seconds=60,
            retry_attempts=3,
            batch_size=5
        )
        
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.timeout_seconds == 60
        assert config.retry_attempts == 3
        assert config.batch_size == 5
    
    def test_config_defaults(self):
        """Test ExtractorConfig default values."""
        config = ExtractorConfig()
        
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.1
        assert config.max_tokens == 2000
        assert config.timeout_seconds == 30
        assert config.retry_attempts == 3
        assert config.batch_size == 10


class TestMockLLMClient:
    """Test MockLLMClient functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_client_extraction(self):
        """Test MockLLMClient provides realistic responses."""
        client = MockLLMClient()
        
        prompt = """Extract attributes from: "iPhone 15 Pro - $999.99"
        
        Required attributes:
        - title (text, required)
        - price (numeric, optional)"""
        
        response = await client.extract_structured_data(
            prompt=prompt,
            model="gpt-4",
            temperature=0.1
        )
        
        assert "attributes" in response
        assert "terms" in response
        assert len(response["attributes"]) >= 1
        assert len(response["terms"]) >= 1
        
        # Check response format
        for attr in response["attributes"]:
            assert "name" in attr
            assert "value" in attr
            assert "evidence" in attr
            assert "confidence" in attr
            assert 0.0 <= attr["confidence"] <= 1.0
        
        for term in response["terms"]:
            assert "term" in term
            assert "context" in term
            assert "confidence" in term
            assert 0.0 <= term["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_mock_client_call_count(self):
        """Test MockLLMClient tracks call count."""
        client = MockLLMClient()
        
        assert client.call_count == 0
        
        await client.extract_structured_data("test prompt", "gpt-4")
        assert client.call_count == 1
        
        await client.extract_structured_data("test prompt 2", "gpt-4")
        assert client.call_count == 2
    
    @pytest.mark.asyncio
    async def test_mock_client_prompt_analysis(self):
        """Test MockLLMClient analyzes prompts for realistic responses."""
        client = MockLLMClient()
        
        # Test iPhone prompt
        response = await client.extract_structured_data(
            "Extract title from: iPhone 15 Pro description",
            "gpt-4"
        )
        
        # Should extract some relevant attributes
        found_title = any(attr["name"] == "title" for attr in response["attributes"])
        assert found_title
        
        # Should extract iPhone-related terms
        found_smartphone = any("smartphone" in term["term"] for term in response["terms"])
        found_apple = any("apple" in term["term"] for term in response["terms"])
        assert found_smartphone or found_apple


class TestBasicIngestionItem:
    """Test basic IngestionItem creation and validation."""
    
    def test_create_basic_item(self):
        """Test creating a basic IngestionItem."""
        item = IngestionItem(
            item_id="test_123",
            title="Test Product",
            source="test_source"
        )
        
        assert item.item_id == "test_123"
        assert item.title == "Test Product"
        assert item.source == "test_source"
        assert item.description is None
        assert item.language == "auto"
        assert isinstance(item.metadata, dict)
    
    def test_create_full_item(self):
        """Test creating a full IngestionItem with all fields."""
        item = IngestionItem(
            item_id="test_123",
            title="Test Product",
            description="A test product description",
            raw_category="electronics",
            raw_attributes={"color": "blue", "size": "large"},
            source="api_catalog",
            language="en",
            metadata={"source_url": "https://example.com"}
        )
        
        assert item.item_id == "test_123"
        assert item.title == "Test Product"
        assert item.description == "A test product description"
        assert item.raw_category == "electronics"
        assert item.raw_attributes["color"] == "blue"
        assert item.source == "api_catalog"
        assert item.language == "en"
        assert item.metadata["source_url"] == "https://example.com"
    
    def test_item_id_validation(self):
        """Test item_id validation."""
        # Test whitespace trimming
        item = IngestionItem(
            item_id="  test_123  ",
            title="Test Product",
            source="test_source"
        )
        assert item.item_id == "test_123"
        
        # Test empty item_id validation (Pydantic validation error)
        with pytest.raises(Exception):  # Could be ValueError or ValidationError
            IngestionItem(
                item_id="   ",
                title="Test Product",
                source="test_source"
            )


class TestBasicModels:
    """Test basic model creation."""
    
    def test_attribute_extraction_creation(self):
        """Test creating AttributeExtraction."""
        attr = AttributeExtraction(
            name="title",
            value="iPhone 15 Pro",
            evidence="Found in product title",
            confidence=0.95
        )
        
        assert attr.name == "title"
        assert attr.value == "iPhone 15 Pro"
        assert attr.evidence == "Found in product title"
        assert attr.confidence == 0.95
    
    def test_semantic_term_creation(self):
        """Test creating SemanticTerm."""
        term = SemanticTerm(
            term="smartphone",
            context="mobile phone device",
            confidence=0.90
        )
        
        assert term.term == "smartphone"
        assert term.context == "mobile phone device"
        assert term.confidence == 0.90
    
    def test_extraction_result_creation(self):
        """Test creating ExtractionResult."""
        attributes = [
            AttributeExtraction(
                name="title",
                value="iPhone 15 Pro", 
                evidence="Product title"
            ),
            AttributeExtraction(
                name="category",
                value="smartphone",
                evidence="Device type",
                confidence=0.90
            )
        ]
        
        terms = [
            SemanticTerm(term="smartphone", context="mobile device", confidence=0.85),
            SemanticTerm(term="iPhone", context="Apple product", confidence=0.95)
        ]
        
        from datetime import datetime
        metadata = ExtractionMetadata(
            model="gpt-4",
            run_id="test_run_123",
            extracted_at=datetime.now(),
            processing_time_ms=1500
        )
        
        result = ExtractionResult(
            item_id="test_123",
            attributes=attributes,
            terms=terms,
            metadata=metadata
        )
        
        assert result.item_id == "test_123"
        assert len(result.attributes) == 2
        assert len(result.terms) == 2
        assert result.metadata.model == "gpt-4"
        assert result.metadata.processing_time_ms == 1500
        assert result.attributes[0].name == "title"
        assert result.terms[0].term == "smartphone"


class TestAsyncOperations:
    """Test async operation basics."""
    
    @pytest.mark.asyncio
    async def test_basic_async_function(self):
        """Test basic async functionality."""
        async def simple_async():
            await asyncio.sleep(0.001)  # Very short sleep
            return "success"
        
        result = await simple_async()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_mock_async_client(self):
        """Test MockLLMClient async operations."""
        client = MockLLMClient()
        
        # Test multiple concurrent calls
        tasks = [
            client.extract_structured_data(f"prompt {i}", "gpt-4")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert client.call_count == 3
        
        # Each result should have the expected structure
        for result in results:
            assert "attributes" in result
            assert "terms" in result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
