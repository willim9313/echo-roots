"""Tests for core data models."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from pydantic import ValidationError

from echo_roots.models.core import (
    IngestionItem,
    AttributeExtraction,
    SemanticTerm,
    ExtractionResult,
    ExtractionMetadata,
    ElevationProposal,
    ElevationMetrics,
    Mapping,
)


class TestIngestionItem:
    """Test cases for IngestionItem model."""
    
    def test_ingestion_item_creation_valid(self):
        """Test creating a valid ingestion item."""
        item = IngestionItem(
            item_id="test-001",
            title="Test Product",
            description="A test product description",
            source="test-api",
            language="en"
        )
        
        assert item.item_id == "test-001"
        assert item.title == "Test Product"
        assert item.source == "test-api"
        assert item.language == "en"
        assert isinstance(item.ingested_at, datetime)
        assert item.raw_attributes == {}
        assert item.metadata == {}
    
    def test_ingestion_item_with_all_fields(self):
        """Test creating an ingestion item with all optional fields."""
        metadata = {"source_url": "https://example.com", "batch_id": "batch-001"}
        raw_attrs = {"brand": "Apple", "color": "Black", "price": 999.99}
        
        item = IngestionItem(
            item_id="test-002",
            title="Apple iPhone",
            description="Latest iPhone model",
            raw_category="Electronics > Smartphones",
            raw_attributes=raw_attrs,
            source="ecommerce-api",
            language="en",
            metadata=metadata
        )
        
        assert item.raw_category == "Electronics > Smartphones"
        assert item.raw_attributes == raw_attrs
        assert item.metadata == metadata
    
    def test_ingestion_item_validation_errors(self):
        """Test validation errors for invalid data."""
        # Empty item_id
        with pytest.raises(ValidationError):
            IngestionItem(item_id="", title="Test", source="test")
        
        # Whitespace-only item_id
        with pytest.raises(ValidationError):
            IngestionItem(item_id="   ", title="Test", source="test")
        
        # Empty title
        with pytest.raises(ValidationError):
            IngestionItem(item_id="test", title="", source="test")
    
    def test_ingestion_item_string_trimming(self):
        """Test that string fields are properly trimmed."""
        item = IngestionItem(
            item_id="  test-001  ",
            title="  Test Product  ",
            source="  test-api  "
        )
        
        assert item.item_id == "test-001"
        assert item.title == "Test Product"
        assert item.source == "test-api"


class TestAttributeExtraction:
    """Test cases for AttributeExtraction model."""
    
    def test_attribute_extraction_valid(self):
        """Test creating a valid attribute extraction."""
        attr = AttributeExtraction(
            name="brand",
            value="Apple",
            evidence="Apple iPhone 15 Pro"
        )
        
        assert attr.name == "brand"
        assert attr.value == "Apple"
        assert attr.evidence == "Apple iPhone 15 Pro"
        assert attr.confidence is None
    
    def test_attribute_extraction_with_confidence(self):
        """Test attribute extraction with confidence score."""
        attr = AttributeExtraction(
            name="color",
            value="Natural Titanium",
            evidence="Natural Titanium color option",
            confidence=0.95
        )
        
        assert attr.confidence == 0.95
    
    def test_attribute_extraction_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        AttributeExtraction(name="test", value="test", evidence="test", confidence=0.0)
        AttributeExtraction(name="test", value="test", evidence="test", confidence=1.0)
        AttributeExtraction(name="test", value="test", evidence="test", confidence=0.5)
        
        # Invalid confidence scores
        with pytest.raises(ValueError):
            AttributeExtraction(name="test", value="test", evidence="test", confidence=-0.1)
        
        with pytest.raises(ValueError):
            AttributeExtraction(name="test", value="test", evidence="test", confidence=1.1)


class TestSemanticTerm:
    """Test cases for SemanticTerm model."""
    
    def test_semantic_term_valid(self):
        """Test creating a valid semantic term."""
        term = SemanticTerm(
            term="smartphone",
            context="Latest smartphone with advanced features",
            confidence=0.85
        )
        
        assert term.term == "smartphone"
        assert term.context == "Latest smartphone with advanced features"
        assert term.confidence == 0.85
        assert term.frequency is None
    
    def test_semantic_term_with_frequency(self):
        """Test semantic term with frequency count."""
        term = SemanticTerm(
            term="iPhone",
            context="Apple iPhone series",
            confidence=0.9,
            frequency=42
        )
        
        assert term.frequency == 42
    
    def test_semantic_term_frequency_validation(self):
        """Test frequency validation."""
        # Valid frequency
        SemanticTerm(term="test", context="test", confidence=0.5, frequency=1)
        
        # Invalid frequency (zero or negative)
        with pytest.raises(ValueError):
            SemanticTerm(term="test", context="test", confidence=0.5, frequency=0)
        
        with pytest.raises(ValueError):
            SemanticTerm(term="test", context="test", confidence=0.5, frequency=-1)


class TestExtractionResult:
    """Test cases for ExtractionResult model."""
    
    def test_extraction_result_valid(self):
        """Test creating a valid extraction result."""
        metadata = ExtractionMetadata(
            model="gpt-4",
            run_id="run-001",
            extracted_at=datetime.now()
        )
        
        attributes = [
            AttributeExtraction(name="brand", value="Apple", evidence="Apple iPhone"),
            AttributeExtraction(name="color", value="Black", evidence="Black color")
        ]
        
        terms = [
            SemanticTerm(term="smartphone", context="smartphone features", confidence=0.9)
        ]
        
        result = ExtractionResult(
            item_id="test-001",
            attributes=attributes,
            terms=terms,
            metadata=metadata
        )
        
        assert result.item_id == "test-001"
        assert len(result.attributes) == 2
        assert len(result.terms) == 1
        assert result.metadata.model == "gpt-4"
    
    def test_extraction_result_unique_attributes(self):
        """Test that duplicate attribute names are not allowed."""
        metadata = ExtractionMetadata(
            model="gpt-4",
            run_id="run-001",
            extracted_at=datetime.now()
        )
        
        # Duplicate attribute names
        attributes = [
            AttributeExtraction(name="brand", value="Apple", evidence="Apple iPhone"),
            AttributeExtraction(name="brand", value="Samsung", evidence="Samsung Galaxy")
        ]
        
        with pytest.raises(ValueError, match="Duplicate attribute names"):
            ExtractionResult(
                item_id="test-001",
                attributes=attributes,
                metadata=metadata
            )
    
    def test_extraction_result_empty_collections(self):
        """Test extraction result with empty attributes and terms."""
        metadata = ExtractionMetadata(
            model="gpt-4",
            run_id="run-001",
            extracted_at=datetime.now()
        )
        
        result = ExtractionResult(
            item_id="test-001",
            metadata=metadata
        )
        
        assert result.attributes == []
        assert result.terms == []


class TestElevationProposal:
    """Test cases for ElevationProposal model."""
    
    def test_elevation_proposal_valid(self):
        """Test creating a valid elevation proposal."""
        metrics = ElevationMetrics(
            frequency=150,
            coverage=0.25,
            stability_score=0.8
        )
        
        proposal = ElevationProposal(
            term="smartphone",
            proposed_attribute="device_type",
            justification="Term appears frequently with consistent meaning across product descriptions",
            metrics=metrics,
            submitted_by="user-001"
        )
        
        assert proposal.term == "smartphone"
        assert proposal.proposed_attribute == "device_type"
        assert proposal.status == "pending"
        assert proposal.reviewed_at is None
        assert len(proposal.proposal_id) > 0  # UUID generated
    
    def test_elevation_proposal_status_transitions(self):
        """Test status transitions and validation."""
        metrics = ElevationMetrics(frequency=100, coverage=0.2, stability_score=0.7)
        
        # Pending proposal should not have reviewed_at
        proposal = ElevationProposal(
            term="test",
            proposed_attribute="test_attr",
            justification="test justification for proposal",
            metrics=metrics,
            submitted_by="user-001",
            status="pending"
        )
        assert proposal.reviewed_at is None
        
        # Approved proposal must have reviewed_at
        with pytest.raises(ValueError, match="reviewed_at is required"):
            ElevationProposal(
                term="test",
                proposed_attribute="test_attr",
                justification="test justification for proposal",
                metrics=metrics,
                submitted_by="user-001",
                status="approved"
            )
        
        # Pending proposal should not have reviewed_at set
        with pytest.raises(ValueError, match="reviewed_at should not be set"):
            ElevationProposal(
                term="test",
                proposed_attribute="test_attr",
                justification="test justification for proposal",
                metrics=metrics,
                submitted_by="user-001",
                status="pending",
                reviewed_at=datetime.now()
            )


class TestMapping:
    """Test cases for Mapping model."""
    
    def test_mapping_valid(self):
        """Test creating a valid mapping."""
        mapping = Mapping(
            from_term="cellphone",
            to_term="mobile phone",
            relation_type="alias",
            created_by="admin-001"
        )
        
        assert mapping.from_term == "cellphone"
        assert mapping.to_term == "mobile phone"
        assert mapping.relation_type == "alias"
        assert mapping.valid_to is None
        assert len(mapping.mapping_id) > 0  # UUID generated
    
    def test_mapping_deprecate_relation(self):
        """Test deprecate relation type allows same from/to terms."""
        mapping = Mapping(
            from_term="old_term",
            to_term="old_term",
            relation_type="deprecate",
            created_by="admin-001"
        )
        
        assert mapping.from_term == mapping.to_term
        assert mapping.relation_type == "deprecate"
    
    def test_mapping_same_terms_validation(self):
        """Test that non-deprecate mappings cannot have same from/to terms."""
        with pytest.raises(ValueError, match="from_term and to_term cannot be the same"):
            Mapping(
                from_term="same_term",
                to_term="same_term", 
                relation_type="alias",
                created_by="admin-001"
            )
    
    def test_mapping_date_validation(self):
        """Test date validation for valid_from and valid_to."""
        now = datetime.now()
        future = now + timedelta(days=1)
        past = now - timedelta(days=1)
        
        # Valid: valid_to after valid_from
        Mapping(
            from_term="old",
            to_term="new",
            relation_type="replace",
            created_by="admin",
            valid_from=now,
            valid_to=future
        )
        
        # Invalid: valid_to before valid_from
        with pytest.raises(ValueError, match="valid_to must be after valid_from"):
            Mapping(
                from_term="old",
                to_term="new", 
                relation_type="replace",
                created_by="admin",
                valid_from=now,
                valid_to=past
            )
    
    def test_mapping_string_trimming(self):
        """Test string trimming for terms."""
        mapping = Mapping(
            from_term="  old term  ",
            to_term="  new term  ",
            relation_type="replace",
            created_by="admin"
        )
        
        assert mapping.from_term == "old term"
        assert mapping.to_term == "new term"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_large_text_fields(self):
        """Test handling of large text fields within limits."""
        # Large but valid description
        large_description = "A" * 4999  # Just under 5000 limit
        
        item = IngestionItem(
            item_id="test",
            title="Test",
            description=large_description,
            source="test"
        )
        
        assert len(item.description) == 4999
        
        # Too large description should fail
        too_large = "A" * 5001
        with pytest.raises(ValueError):
            IngestionItem(
                item_id="test",
                title="Test", 
                description=too_large,
                source="test"
            )
    
    def test_unicode_handling(self):
        """Test proper Unicode handling for multilingual content."""
        item = IngestionItem(
            item_id="unicode-test",
            title="测试产品 Test Product",
            description="这是一个测试产品 This is a test product",
            source="test",
            language="zh-CN"
        )
        
        assert "测试" in item.title
        assert "这是" in item.description
        assert item.language == "zh-CN"
    
    def test_json_serialization(self):
        """Test that models can be serialized to JSON."""
        item = IngestionItem(
            item_id="json-test",
            title="Test Product",
            source="test"
        )
        
        # Should not raise exception
        json_str = item.model_dump_json()
        assert "json-test" in json_str
        assert "Test Product" in json_str
        
        # Should be able to recreate from JSON
        data = item.model_dump()
        recreated = IngestionItem(**data)
        assert recreated.item_id == item.item_id
        assert recreated.title == item.title
