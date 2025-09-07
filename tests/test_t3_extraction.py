"""Tests for the LLM extraction pipeline (T3).

This module tests the LLM-based extraction pipeline that processes
IngestionItems through domain-specific prompts to produce ExtractionResults.

Key test areas:
- ExtractorConfig validation and usage
- PromptBuilder template generation and field mapping
- LLMExtractor orchestration and error handling
- ExtractionPipeline end-to-end workflows
- OpenAI client integration and mocking
- Validation and post-processing
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from echo_roots.models.core import IngestionItem, ExtractionResult, AttributeExtraction, SemanticTerm
from echo_roots.models.domain import DomainPack
from echo_roots.pipelines.extraction import (
    ExtractorConfig, PromptBuilder, LLMExtractor, ExtractionPipeline, ExtractionError
)
from echo_roots.pipelines.openai_client import OpenAIClient, MockLLMClient
from echo_roots.pipelines.validation import (
    ExtractionValidator, ResultNormalizer, QualityAnalyzer, PostProcessor,
    ValidationIssue, QualityMetrics
)


@pytest.fixture
def sample_domain_pack():
    """Create a sample domain pack for testing."""
    return DomainPack(
        domain_name="test_domain",
        version="1.0.0",
        description="Test domain for extraction pipeline",
        output_schema={
            "attributes": [
                {"key": "title", "type": "text", "required": True},
                {"key": "price", "type": "numeric", "required": False},
                {"key": "category", "type": "categorical", "allow_values": ["electronics", "books", "clothing"]},
                {"key": "availability", "type": "boolean", "required": False}
            ],
            "terms": {
                "min_count": 2,
                "categories": ["product", "feature", "brand"]
            }
        },
        attribute_hints={
            "title": {
                "extraction_hints": ["Look for product names", "Check headings"],
                "examples": ["iPhone 15 Pro", "MacBook Air"],
                "required": True
            },
            "price": {
                "extraction_hints": ["Look for currency symbols", "Check for numeric values"],
                "pattern": r"\\$?\\d+\\.\\d{2}",
                "examples": ["$999.99", "1299.00"]
            }
        },
        prompt_templates={
            "attribute_extraction": """Extract structured attributes from this {domain_name} item:

{text}

Required attributes:
{attribute_definitions}

Respond with JSON containing 'attributes' array with objects having 'name', 'value', 'evidence', 'confidence' fields.""",
            "term_extraction": """Extract semantic terms from this {domain_name} item:

{text}

Extract terms in these categories: {term_categories}

Respond with JSON containing 'terms' array with objects having 'term', 'context', 'confidence' fields."""
        }
    )


@pytest.fixture
def sample_ingestion_item():
    """Create a sample ingestion item for testing."""
    return IngestionItem(
        id="test_item_1",
        text="iPhone 15 Pro - $999.99. Latest smartphone with advanced camera system and titanium design. Available in stores now.",
        metadata={
            "source": "product_catalog",
            "timestamp": "2024-01-15T10:30:00Z",
            "url": "https://example.com/iphone15pro"
        }
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return {
        "attributes": [
            {
                "name": "title",
                "value": "iPhone 15 Pro",
                "evidence": "iPhone 15 Pro - $999.99",
                "confidence": 0.95
            },
            {
                "name": "price",
                "value": "$999.99",
                "evidence": "iPhone 15 Pro - $999.99",
                "confidence": 0.90
            },
            {
                "name": "category",
                "value": "electronics",
                "evidence": "smartphone with advanced camera system",
                "confidence": 0.85
            },
            {
                "name": "availability",
                "value": "true",
                "evidence": "Available in stores now",
                "confidence": 0.80
            }
        ],
        "terms": [
            {
                "term": "smartphone",
                "context": "Latest smartphone with advanced camera system",
                "confidence": 0.90
            },
            {
                "term": "camera system",
                "context": "advanced camera system and titanium design",
                "confidence": 0.85
            },
            {
                "term": "titanium design",
                "context": "titanium design. Available in stores",
                "confidence": 0.80
            }
        ]
    }


class TestExtractorConfig:
    """Test ExtractorConfig functionality."""
    
    def test_config_creation(self):
        """Test creating ExtractorConfig with valid parameters."""
        config = ExtractorConfig(
            llm_model="gpt-4",
            temperature=0.3,
            max_tokens=2000,
            timeout=60.0,
            max_retries=3,
            batch_size=5
        )
        
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.3
        assert config.max_tokens == 2000
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.batch_size == 5
    
    def test_config_defaults(self):
        """Test ExtractorConfig default values."""
        config = ExtractorConfig()
        
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.temperature == 0.1
        assert config.max_tokens == 1500
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.batch_size == 10
    
    def test_config_validation(self):
        """Test ExtractorConfig parameter validation."""
        # Test invalid temperature
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
            ExtractorConfig(temperature=3.0)
        
        # Test invalid max_tokens
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ExtractorConfig(max_tokens=0)
        
        # Test invalid timeout
        with pytest.raises(ValueError, match="timeout must be positive"):
            ExtractorConfig(timeout=-1.0)
        
        # Test invalid max_retries
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ExtractorConfig(max_retries=-1)
        
        # Test invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            ExtractorConfig(batch_size=0)


class TestPromptBuilder:
    """Test PromptBuilder functionality."""
    
    def test_prompt_builder_creation(self, sample_domain_pack):
        """Test creating PromptBuilder with domain pack."""
        builder = PromptBuilder(sample_domain_pack)
        assert builder.domain_pack == sample_domain_pack
    
    def test_build_attribute_prompt(self, sample_domain_pack, sample_ingestion_item):
        """Test building attribute extraction prompt."""
        builder = PromptBuilder(sample_domain_pack)
        prompt = builder.build_attribute_prompt(sample_ingestion_item)
        
        assert "Extract structured attributes" in prompt
        assert "test_domain" in prompt
        assert sample_ingestion_item.text in prompt
        assert "title" in prompt
        assert "price" in prompt
        assert "category" in prompt
        assert "availability" in prompt
    
    def test_build_term_prompt(self, sample_domain_pack, sample_ingestion_item):
        """Test building term extraction prompt."""
        builder = PromptBuilder(sample_domain_pack)
        prompt = builder.build_term_prompt(sample_ingestion_item)
        
        assert "Extract semantic terms" in prompt
        assert "test_domain" in prompt
        assert sample_ingestion_item.text in prompt
        assert "product" in prompt
        assert "feature" in prompt
        assert "brand" in prompt
    
    def test_format_attribute_definitions(self, sample_domain_pack):
        """Test formatting attribute definitions."""
        builder = PromptBuilder(sample_domain_pack)
        definitions = builder._format_attribute_definitions()
        
        assert "title (text, required)" in definitions
        assert "price (numeric, optional)" in definitions
        assert "category (categorical: electronics, books, clothing)" in definitions
        assert "availability (boolean, optional)" in definitions
    
    def test_missing_template_handling(self, sample_domain_pack):
        """Test handling of missing prompt templates."""
        # Remove templates to test error handling
        sample_domain_pack.prompt_templates = {}
        
        builder = PromptBuilder(sample_domain_pack)
        
        with pytest.raises(ExtractionError, match="Missing prompt template"):
            builder.build_attribute_prompt(sample_ingestion_item)


class TestLLMExtractor:
    """Test LLMExtractor functionality."""
    
    @pytest.fixture
    def mock_llm_client(self, mock_llm_response):
        """Create a mock LLM client."""
        client = Mock()
        client.extract_structured_data = AsyncMock(return_value=mock_llm_response)
        return client
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExtractorConfig(
            llm_model="gpt-3.5-turbo",
            temperature=0.1,
            max_retries=2
        )
    
    def test_extractor_creation(self, mock_llm_client, config):
        """Test creating LLMExtractor."""
        extractor = LLMExtractor(mock_llm_client, config)
        assert extractor.llm_client == mock_llm_client
        assert extractor.config == config
    
    @pytest.mark.asyncio
    async def test_extract_attributes(self, mock_llm_client, config, sample_ingestion_item, mock_llm_response):
        """Test attribute extraction."""
        extractor = LLMExtractor(mock_llm_client, config)
        prompt = "Test prompt for attributes"
        
        attributes = await extractor.extract_attributes(sample_ingestion_item, prompt)
        
        assert len(attributes) == 4
        assert attributes[0].name == "title"
        assert attributes[0].value == "iPhone 15 Pro"
        assert attributes[0].confidence == 0.95
        
        mock_llm_client.extract_structured_data.assert_called_once_with(
            prompt=prompt,
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout
        )
    
    @pytest.mark.asyncio
    async def test_extract_terms(self, mock_llm_client, config, sample_ingestion_item, mock_llm_response):
        """Test term extraction."""
        extractor = LLMExtractor(mock_llm_client, config)
        prompt = "Test prompt for terms"
        
        terms = await extractor.extract_terms(sample_ingestion_item, prompt)
        
        assert len(terms) == 3
        assert terms[0].term == "smartphone"
        assert terms[0].confidence == 0.90
        
        mock_llm_client.extract_structured_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_with_retries(self, config):
        """Test error handling and retry logic."""
        # Create a client that fails twice then succeeds
        client = Mock()
        client.extract_structured_data = AsyncMock(
            side_effect=[
                Exception("Network error"),
                Exception("Rate limit"),
                {"attributes": [], "terms": []}
            ]
        )
        
        extractor = LLMExtractor(client, config)
        prompt = "Test prompt"
        item = IngestionItem(id="test", text="test text")
        
        # Should succeed on third try
        attributes = await extractor.extract_attributes(item, prompt)
        assert attributes == []
        assert client.extract_structured_data.call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, config):
        """Test behavior when max retries are exceeded."""
        client = Mock()
        client.extract_structured_data = AsyncMock(side_effect=Exception("Persistent error"))
        
        extractor = LLMExtractor(client, config)
        prompt = "Test prompt"
        item = IngestionItem(id="test", text="test text")
        
        with pytest.raises(ExtractionError, match="Failed to extract after 2 retries"):
            await extractor.extract_attributes(item, prompt)
    
    @pytest.mark.asyncio
    async def test_invalid_response_format(self, config):
        """Test handling of invalid LLM response format."""
        client = Mock()
        client.extract_structured_data = AsyncMock(return_value={"invalid": "format"})
        
        extractor = LLMExtractor(client, config)
        prompt = "Test prompt"
        item = IngestionItem(id="test", text="test text")
        
        with pytest.raises(ExtractionError, match="Invalid response format"):
            await extractor.extract_attributes(item, prompt)


class TestExtractionPipeline:
    """Test ExtractionPipeline functionality."""
    
    @pytest.fixture
    def mock_llm_client(self, mock_llm_response):
        """Create a mock LLM client."""
        client = Mock()
        client.extract_structured_data = AsyncMock(return_value=mock_llm_response)
        return client
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExtractorConfig(max_retries=1)
    
    @pytest.fixture
    def pipeline(self, sample_domain_pack, mock_llm_client, config):
        """Create extraction pipeline for testing."""
        return ExtractionPipeline(sample_domain_pack, mock_llm_client, config)
    
    def test_pipeline_creation(self, sample_domain_pack, mock_llm_client, config):
        """Test creating ExtractionPipeline."""
        pipeline = ExtractionPipeline(sample_domain_pack, mock_llm_client, config)
        assert pipeline.domain_pack == sample_domain_pack
        assert pipeline.llm_client == mock_llm_client
        assert pipeline.config == config
    
    @pytest.mark.asyncio
    async def test_process_single_item(self, pipeline, sample_ingestion_item):
        """Test processing a single ingestion item."""
        result = await pipeline.process_item(sample_ingestion_item)
        
        assert isinstance(result, ExtractionResult)
        assert result.item_id == sample_ingestion_item.id
        assert len(result.attributes) == 4
        assert len(result.terms) == 3
        assert result.metadata["domain"] == "test_domain"
    
    @pytest.mark.asyncio
    async def test_process_batch(self, pipeline, sample_ingestion_item):
        """Test processing a batch of items."""
        items = [sample_ingestion_item, sample_ingestion_item]  # Duplicate for testing
        results = await pipeline.process_batch(items)
        
        assert len(results) == 2
        assert all(isinstance(result, ExtractionResult) for result in results)
    
    @pytest.mark.asyncio
    async def test_process_with_validation(self, pipeline, sample_ingestion_item):
        """Test processing with validation enabled."""
        result = await pipeline.process_item(sample_ingestion_item, validate=True)
        
        # Result should be processed and validated
        assert isinstance(result, ExtractionResult)
        assert result.metadata.get("validation_passed") is not None
    
    @pytest.mark.asyncio
    async def test_process_empty_text(self, pipeline):
        """Test processing item with empty text."""
        empty_item = IngestionItem(id="empty", text="", metadata={})
        
        with pytest.raises(ExtractionError, match="Item text is empty"):
            await pipeline.process_item(empty_item)
    
    @pytest.mark.asyncio
    async def test_error_propagation(self, sample_domain_pack, config):
        """Test that errors from LLM client are properly propagated."""
        failing_client = Mock()
        failing_client.extract_structured_data = AsyncMock(side_effect=Exception("LLM error"))
        
        pipeline = ExtractionPipeline(sample_domain_pack, failing_client, config)
        item = IngestionItem(id="test", text="test text")
        
        with pytest.raises(ExtractionError):
            await pipeline.process_item(item)


class TestValidation:
    """Test validation and post-processing functionality."""
    
    def test_validator_creation(self, sample_domain_pack):
        """Test creating ExtractionValidator."""
        validator = ExtractionValidator(sample_domain_pack)
        assert validator.domain_pack == sample_domain_pack
        assert len(validator.expected_attributes) > 0
    
    def test_valid_result_validation(self, sample_domain_pack):
        """Test validation of a valid extraction result."""
        validator = ExtractionValidator(sample_domain_pack)
        
        result = ExtractionResult(
            item_id="test",
            attributes=[
                AttributeExtraction("title", "iPhone 15 Pro", "iPhone 15 Pro - $999.99", 0.95),
                AttributeExtraction("price", "$999.99", "$999.99", 0.90)
            ],
            terms=[
                SemanticTerm("smartphone", "Latest smartphone", 0.85),
                SemanticTerm("camera", "advanced camera system", 0.80)
            ],
            metadata={}
        )
        
        is_valid, issues = validator.validate(result)
        assert is_valid
        # May have warnings but no errors
        assert all(issue.severity != 'error' for issue in issues)
    
    def test_invalid_result_validation(self, sample_domain_pack):
        """Test validation of an invalid extraction result."""
        validator = ExtractionValidator(sample_domain_pack)
        
        # Missing required attribute, invalid confidence
        result = ExtractionResult(
            item_id="test",
            attributes=[
                AttributeExtraction("price", "", "empty evidence", 1.5)  # Empty value, invalid confidence
            ],
            terms=[],
            metadata={}
        )
        
        is_valid, issues = validator.validate(result)
        assert not is_valid
        
        # Should have errors for missing required field and invalid confidence
        errors = [issue for issue in issues if issue.severity == 'error']
        assert len(errors) >= 2
    
    def test_normalizer(self, sample_domain_pack):
        """Test result normalization."""
        normalizer = ResultNormalizer(sample_domain_pack)
        
        result = ExtractionResult(
            item_id="test",
            attributes=[
                AttributeExtraction("TITLE ", "  iPhone 15 Pro  ", "Evidence text", 0.95),
                AttributeExtraction("price-value", "$999.99", "Price evidence", 0.90)
            ],
            terms=[
                SemanticTerm("  smartphone  ", "Context text", 0.85)
            ],
            metadata={}
        )
        
        normalized = normalizer.normalize(result)
        
        # Check normalization
        assert normalized.attributes[0].name == "title"
        assert normalized.attributes[0].value == "iPhone 15 Pro"
        assert normalized.attributes[1].name == "price_value"
        assert normalized.terms[0].term == "smartphone"
    
    def test_quality_analyzer(self, sample_domain_pack):
        """Test quality analysis."""
        analyzer = QualityAnalyzer(sample_domain_pack)
        
        result = ExtractionResult(
            item_id="test",
            attributes=[
                AttributeExtraction("title", "iPhone 15 Pro", "Good evidence", 0.95),
                AttributeExtraction("price", "$999.99", "Price evidence", 0.90)
            ],
            terms=[
                SemanticTerm("smartphone", "Context", 0.85)
            ],
            metadata={}
        )
        
        # Simulate validation issues
        issues = [
            ValidationIssue("warning", "quality", "Minor issue"),
            ValidationIssue("info", "schema", "Info message")
        ]
        
        metrics = analyzer.analyze(result, issues)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.avg_attribute_confidence == 0.925
        assert metrics.avg_term_confidence == 0.85
        assert 0.0 <= metrics.total_quality_score <= 1.0
    
    def test_post_processor(self, sample_domain_pack):
        """Test complete post-processing workflow."""
        processor = PostProcessor(sample_domain_pack)
        
        result = ExtractionResult(
            item_id="test",
            attributes=[
                AttributeExtraction("title", "iPhone 15 Pro", "Evidence", 0.95)
            ],
            terms=[
                SemanticTerm("smartphone", "Context", 0.85)
            ],
            metadata={}
        )
        
        processed_result, issues, metrics = processor.process(result)
        
        assert isinstance(processed_result, ExtractionResult)
        assert isinstance(issues, list)
        assert isinstance(metrics, QualityMetrics)


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
            model="gpt-3.5-turbo",
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


class TestIntegration:
    """Integration tests for the complete T3 pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_extraction(self, sample_domain_pack, sample_ingestion_item):
        """Test complete end-to-end extraction pipeline."""
        # Use MockLLMClient for integration test
        client = MockLLMClient()
        config = ExtractorConfig(batch_size=1)
        
        pipeline = ExtractionPipeline(sample_domain_pack, client, config)
        
        # Process item
        result = await pipeline.process_item(sample_ingestion_item, validate=True)
        
        # Verify result structure
        assert isinstance(result, ExtractionResult)
        assert result.item_id == sample_ingestion_item.id
        assert len(result.attributes) > 0
        assert len(result.terms) > 0
        
        # Verify metadata
        assert result.metadata["domain"] == "test_domain"
        assert "processing_time" in result.metadata
        assert "validation_passed" in result.metadata
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, sample_domain_pack):
        """Test batch processing with validation."""
        client = MockLLMClient()
        config = ExtractorConfig(batch_size=2)
        
        pipeline = ExtractionPipeline(sample_domain_pack, client, config)
        
        # Create multiple items
        items = [
            IngestionItem(id=f"item_{i}", text=f"Product {i} description", metadata={})
            for i in range(3)
        ]
        
        # Process batch
        results = await pipeline.process_batch(items, validate=True)
        
        assert len(results) == 3
        assert all(isinstance(result, ExtractionResult) for result in results)
        assert all(result.metadata.get("validation_passed") is not None for result in results)
    
    def test_domain_pack_integration(self, sample_domain_pack):
        """Test integration with domain pack configuration."""
        # Test that all components can use the domain pack
        builder = PromptBuilder(sample_domain_pack)
        validator = ExtractionValidator(sample_domain_pack)
        normalizer = ResultNormalizer(sample_domain_pack)
        
        # Verify they all reference the same domain
        assert builder.domain_pack.domain_name == "test_domain"
        assert validator.domain_pack.domain_name == "test_domain"
        assert normalizer.domain_pack.domain_name == "test_domain"
        
        # Test expected attributes extraction
        expected_attrs = validator.expected_attributes
        assert "title" in expected_attrs
        assert "price" in expected_attrs
        assert expected_attrs["title"]["required"] == True


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_very_long_text(self, sample_domain_pack):
        """Test processing very long text."""
        client = MockLLMClient()
        config = ExtractorConfig()
        pipeline = ExtractionPipeline(sample_domain_pack, client, config)
        
        # Create item with very long text
        long_text = "Long product description. " * 1000  # Very long text
        item = IngestionItem(id="long", text=long_text, metadata={})
        
        # Should handle gracefully
        result = await pipeline.process_item(item)
        assert isinstance(result, ExtractionResult)
    
    def test_malformed_domain_pack(self):
        """Test handling of malformed domain pack."""
        # Create domain pack with missing required fields
        malformed_pack = DomainPack(
            domain_name="malformed",
            version="1.0.0",
            description="Malformed pack",
            output_schema={},  # Empty schema
            attribute_hints={},
            prompt_templates={}  # Missing templates
        )
        
        # Should handle gracefully in prompt builder
        builder = PromptBuilder(malformed_pack)
        item = IngestionItem(id="test", text="test", metadata={})
        
        with pytest.raises(ExtractionError):
            builder.build_attribute_prompt(item)
    
    @pytest.mark.asyncio
    async def test_empty_batch_processing(self, sample_domain_pack):
        """Test processing empty batch."""
        client = MockLLMClient()
        pipeline = ExtractionPipeline(sample_domain_pack, client, ExtractorConfig())
        
        results = await pipeline.process_batch([])
        assert results == []
    
    def test_unicode_handling(self, sample_domain_pack):
        """Test handling of Unicode text."""
        item = IngestionItem(
            id="unicode",
            text="产品名称: iPhone 15 Pro - ¥7999.99 🚀",
            metadata={}
        )
        
        # Should handle Unicode in prompt building
        builder = PromptBuilder(sample_domain_pack)
        prompt = builder.build_attribute_prompt(item)
        assert "产品名称" in prompt
        assert "🚀" in prompt


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
