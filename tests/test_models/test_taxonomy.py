"""Tests for taxonomy framework models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from echo_roots.models.taxonomy import (
    Category,
    Attribute,
    AttributeValue,
    SemanticCandidate,
    SemanticRelation,
)


class TestCategory:
    """Test cases for Category model."""
    
    def test_category_creation_root(self):
        """Test creating a root category."""
        category = Category(
            name="Electronics",
            level=0,
            path=["Electronics"]
        )
        
        assert category.name == "Electronics"
        assert category.level == 0
        assert category.path == ["Electronics"]
        assert category.parent_id is None
        assert category.status == "active"
        assert len(category.category_id) > 0  # UUID generated
    
    def test_category_creation_child(self):
        """Test creating a child category."""
        category = Category(
            name="Smartphones",
            parent_id="parent-uuid",
            level=2,
            path=["Electronics", "Mobile", "Smartphones"]
        )
        
        assert category.name == "Smartphones"
        assert category.parent_id == "parent-uuid"
        assert category.level == 2
        assert len(category.path) == 3
    
    def test_category_multilingual_labels(self):
        """Test multilingual label support."""
        labels = {
            "en": "Electronics",
            "zh-CN": "电子产品",
            "zh-TW": "電子產品"
        }
        
        category = Category(
            name="Electronics",
            level=0,
            path=["Electronics"],
            labels=labels
        )
        
        assert category.labels["en"] == "Electronics"
        assert category.labels["zh-CN"] == "电子产品"
        assert category.labels["zh-TW"] == "電子產品"
    
    def test_category_hierarchy_validation_root(self):
        """Test hierarchy validation for root categories."""
        # Valid root category
        Category(name="Root", level=0, path=["Root"])
        
        # Root cannot have parent
        with pytest.raises(ValueError, match="Root categories.*cannot have a parent"):
            Category(
                name="Root",
                level=0,
                path=["Root"],
                parent_id="some-parent"
            )
    
    def test_category_hierarchy_validation_child(self):
        """Test hierarchy validation for child categories."""
        # Valid child category
        Category(
            name="Child",
            parent_id="parent-id",
            level=1,
            path=["Parent", "Child"]
        )
        
        # Child must have parent
        with pytest.raises(ValueError, match="Non-root categories must have a parent_id"):
            Category(name="Child", level=1, path=["Parent", "Child"])
    
    def test_category_path_validation(self):
        """Test path validation rules."""
        # Path length must match level + 1
        with pytest.raises(ValidationError):
            Category(name="Test", parent_id="parent", level=1, path=["Too", "Many", "Components"])
        
        # Last path component must match name
        with pytest.raises(ValidationError):
            Category(name="Test", level=0, path=["Different"])
        
        # Empty path components not allowed
        with pytest.raises(ValidationError):
            Category(name="Test", parent_id="parent", level=1, path=["Parent", ""])
    
    def test_category_name_validation(self):
        """Test category name validation."""
        # Empty name
        with pytest.raises(ValidationError):
            Category(name="", level=0, path=[""])
        
        # Whitespace-only name
        with pytest.raises(ValidationError):
            Category(name="   ", level=0, path=["   "])
    
    def test_category_string_trimming(self):
        """Test string trimming for category fields."""
        category = Category(
            name="  Electronics  ",
            level=0,
            path=["  Electronics  "]
        )
        
        assert category.name == "Electronics"
        assert category.path == ["Electronics"]


class TestAttributeValue:
    """Test cases for AttributeValue model."""
    
    def test_attribute_value_creation(self):
        """Test creating a valid attribute value."""
        value = AttributeValue(
            value="Red",
            labels={"en": "Red", "zh-CN": "红色"},
            aliases={"crimson", "cherry"}
        )
        
        assert value.value == "Red"
        assert value.labels["en"] == "Red"
        assert value.labels["zh-CN"] == "红色"
        assert "crimson" in value.aliases
        assert "cherry" in value.aliases
        assert value.status == "active"
    
    def test_attribute_value_validation(self):
        """Test attribute value validation."""
        # Empty value
        with pytest.raises(ValidationError):
            AttributeValue(value="")
        
        # Whitespace-only value  
        with pytest.raises(ValidationError):
            AttributeValue(value="   ")


class TestAttribute:
    """Test cases for Attribute model."""
    
    def test_attribute_creation_categorical(self):
        """Test creating a categorical attribute."""
        values = [
            AttributeValue(value="Red"),
            AttributeValue(value="Blue"),
            AttributeValue(value="Green")
        ]
        
        attribute = Attribute(
            name="color",
            display_name="Color",
            data_type="categorical",
            values=values
        )
        
        assert attribute.name == "color"
        assert attribute.display_name == "Color"
        assert attribute.data_type == "categorical"
        assert len(attribute.values) == 3
        assert not attribute.required
    
    def test_attribute_creation_text(self):
        """Test creating a text attribute."""
        attribute = Attribute(
            name="description",
            display_name="Product Description",
            data_type="text",
            required=True
        )
        
        assert attribute.data_type == "text"
        assert attribute.required is True
        assert attribute.values == []  # No values for text attributes
    
    def test_attribute_name_validation(self):
        """Test attribute name validation."""
        # Valid names
        Attribute(name="brand", display_name="Brand", data_type="text")
        Attribute(name="color_variant", display_name="Color", data_type="text")
        Attribute(name="size_us", display_name="US Size", data_type="text")
        
        # Invalid names (must start with letter, only lowercase, underscores)
        with pytest.raises(ValueError):
            Attribute(name="1invalid", display_name="Test", data_type="text")
        
        with pytest.raises(ValueError):
            Attribute(name="Invalid-Name", display_name="Test", data_type="text")
        
        with pytest.raises(ValueError):
            Attribute(name="InvalidCase", display_name="Test", data_type="text")
    
    def test_attribute_categorical_validation(self):
        """Test validation rules for categorical attributes."""
        # Categorical attributes must have values
        with pytest.raises(ValueError, match="Categorical attributes must have at least one value"):
            Attribute(
                name="color",
                display_name="Color",
                data_type="categorical",
                values=[]
            )
        
        # Non-categorical attributes cannot have values
        values = [AttributeValue(value="Test")]
        with pytest.raises(ValueError, match="Only categorical attributes can have predefined values"):
            Attribute(
                name="description",
                display_name="Description",
                data_type="text",
                values=values
            )
    
    def test_attribute_value_uniqueness(self):
        """Test that attribute values must be unique."""
        values = [
            AttributeValue(value="Red"),
            AttributeValue(value="Red"),  # Duplicate
            AttributeValue(value="Blue")
        ]
        
        with pytest.raises(ValueError, match="Attribute values must be unique"):
            Attribute(
                name="color",
                display_name="Color",
                data_type="categorical",
                values=values
            )
    
    def test_attribute_multilingual_support(self):
        """Test multilingual labels for attributes."""
        labels = {
            "en": "Brand",
            "zh-CN": "品牌",
            "zh-TW": "品牌"
        }
        
        attribute = Attribute(
            name="brand",
            display_name="Brand",
            data_type="text",
            labels=labels
        )
        
        assert attribute.labels["en"] == "Brand"
        assert attribute.labels["zh-CN"] == "品牌"


class TestSemanticRelation:
    """Test cases for SemanticRelation model."""
    
    def test_semantic_relation_creation(self):
        """Test creating a valid semantic relation."""
        relation = SemanticRelation(
            from_term="smartphone",
            to_term="mobile phone",
            relation_type="similar",
            strength=0.85,
            evidence_count=42
        )
        
        assert relation.from_term == "smartphone"
        assert relation.to_term == "mobile phone"
        assert relation.relation_type == "similar"
        assert relation.strength == 0.85
        assert relation.evidence_count == 42
        assert len(relation.relation_id) > 0  # UUID generated
    
    def test_semantic_relation_types(self):
        """Test different relation types."""
        relation_types = ["similar", "related", "variant", "broader", "narrower", "co_occurs"]
        
        for rel_type in relation_types:
            relation = SemanticRelation(
                from_term="term1",
                to_term="term2",
                relation_type=rel_type,
                strength=0.5,
                evidence_count=10
            )
            assert relation.relation_type == rel_type
    
    def test_semantic_relation_validation(self):
        """Test semantic relation validation."""
        # Terms cannot be the same
        with pytest.raises(ValueError, match="from_term and to_term cannot be the same"):
            SemanticRelation(
                from_term="same",
                to_term="same",
                relation_type="similar",
                strength=0.5,
                evidence_count=1
            )
        
        # Case-insensitive check
        with pytest.raises(ValueError, match="from_term and to_term cannot be the same"):
            SemanticRelation(
                from_term="Term",
                to_term="term",
                relation_type="similar",
                strength=0.5,
                evidence_count=1
            )


class TestSemanticCandidate:
    """Test cases for SemanticCandidate model."""
    
    def test_semantic_candidate_creation(self):
        """Test creating a valid semantic candidate."""
        contexts = [
            "smartphone with advanced features",
            "latest smartphone model",
            "smartphone camera quality"
        ]
        
        candidate = SemanticCandidate(
            term="smartphone",
            normalized_term="smartphone",
            frequency=150,
            contexts=contexts,
            score=0.85
        )
        
        assert candidate.term == "smartphone"
        assert candidate.normalized_term == "smartphone"
        assert candidate.frequency == 150
        assert len(candidate.contexts) == 3
        assert candidate.score == 0.85
        assert candidate.status == "active"
        assert candidate.language == "auto"
    
    def test_semantic_candidate_with_relations(self):
        """Test semantic candidate with relations."""
        relations = [
            SemanticRelation(
                from_term="smartphone",
                to_term="mobile phone",
                relation_type="similar",
                strength=0.9,
                evidence_count=50
            )
        ]
        
        candidate = SemanticCandidate(
            term="smartphone",
            normalized_term="smartphone",
            frequency=100,
            score=0.8,
            relations=relations
        )
        
        assert len(candidate.relations) == 1
        assert candidate.relations[0].to_term == "mobile phone"
    
    def test_semantic_candidate_contexts_filtering(self):
        """Test that empty contexts are filtered out."""
        contexts = ["valid context", "", "   ", "another valid context"]
        
        candidate = SemanticCandidate(
            term="test",
            normalized_term="test",
            frequency=10,
            score=0.5,
            contexts=contexts
        )
        
        # Empty and whitespace-only contexts should be filtered
        assert len(candidate.contexts) == 2
        assert "valid context" in candidate.contexts
        assert "another valid context" in candidate.contexts
        assert "" not in candidate.contexts
    
    def test_semantic_candidate_validation(self):
        """Test semantic candidate validation."""
        # Empty term
        with pytest.raises(ValidationError):
            SemanticCandidate(
                term="",
                normalized_term="test",
                frequency=1,
                score=0.5
            )
        
        # Empty normalized_term
        with pytest.raises(ValidationError):
            SemanticCandidate(
                term="test",
                normalized_term="",
                frequency=1,
                score=0.5
            )
    
    def test_semantic_candidate_status_values(self):
        """Test valid status values."""
        valid_statuses = ["active", "clustered", "merged", "elevated", "deprecated"]
        
        for status in valid_statuses:
            candidate = SemanticCandidate(
                term="test",
                normalized_term="test",
                frequency=1,
                score=0.5,
                status=status
            )
            assert candidate.status == status


class TestEdgeCases:
    """Test edge cases and error conditions for taxonomy models."""
    
    def test_deep_category_hierarchy(self):
        """Test maximum category hierarchy depth."""
        # Should work at max depth (level 9, path length 10)
        deep_path = [f"Level{i}" for i in range(10)]  # 0-9 = 10 levels
        
        Category(
            name="Level9",
            parent_id="parent",
            level=9,
            path=deep_path
        )
        
        # Should fail beyond max depth
        with pytest.raises(ValidationError):
            Category(
                name="TooDeep",
                parent_id="parent",
                level=10,
                path=deep_path + ["TooDeep"]
            )
    
    def test_unicode_in_taxonomy(self):
        """Test Unicode support in taxonomy models."""
        # Chinese category
        category = Category(
            name="电子产品",
            level=0,
            path=["电子产品"],
            labels={"zh-CN": "电子产品", "en": "Electronics"}
        )
        
        assert category.name == "电子产品"
        assert "电子产品" in category.path
        
        # Unicode attribute value
        value = AttributeValue(
            value="红色",
            labels={"zh-CN": "红色", "en": "Red"}
        )
        
        assert value.value == "红色"
        assert value.labels["zh-CN"] == "红色"
    
    def test_model_serialization(self):
        """Test JSON serialization of taxonomy models."""
        category = Category(
            name="Test",
            level=0,
            path=["Test"]
        )
        
        # Should serialize without error
        json_data = category.model_dump_json()
        assert "Test" in json_data
        
        # Should deserialize correctly
        data = category.model_dump()
        recreated = Category(**data)
        assert recreated.name == category.name
        assert recreated.level == category.level
