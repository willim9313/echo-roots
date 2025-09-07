"""Tests for domain pack models."""

import pytest
from echo_roots.models.domain import (
    DomainPack,
    AttributeConfig,
    ValidationRule,
    MetricConfig,
    RuntimeConfig,
)


class TestAttributeConfig:
    """Test cases for AttributeConfig model."""
    
    def test_attribute_config_creation(self):
        """Test creating a valid attribute config."""
        config = AttributeConfig(
            key="brand",
            type="categorical",
            examples=["Apple", "Samsung", "Google"],
            notes="Product brand identifier",
            required=True
        )
        
        assert config.key == "brand"
        assert config.type == "categorical"
        assert len(config.examples) == 3
        assert config.notes == "Product brand identifier"
        assert config.required is True
        assert config.normalize_to_lower is False
    
    def test_attribute_config_with_restrictions(self):
        """Test attribute config with value restrictions."""
        config = AttributeConfig(
            key="color",
            type="categorical",
            allow_values=["red", "blue", "green"],
            normalize_to_lower=True,
            max_length=20
        )
        
        assert config.allow_values == ["red", "blue", "green"]
        assert config.normalize_to_lower is True
        assert config.max_length == 20
    
    def test_attribute_config_key_validation(self):
        """Test attribute key validation."""
        # Valid keys
        AttributeConfig(key="brand", type="text")
        AttributeConfig(key="color_variant", type="text")
        AttributeConfig(key="size_us", type="text")
        
        # Invalid keys
        with pytest.raises(ValueError):
            AttributeConfig(key="1invalid", type="text")
        
        with pytest.raises(ValueError):
            AttributeConfig(key="Invalid-Key", type="text")
        
        with pytest.raises(ValueError):
            AttributeConfig(key="", type="text")
    
    def test_attribute_config_type_validation(self):
        """Test attribute type validation."""
        valid_types = ["categorical", "text", "numeric", "boolean", "date"]
        
        for attr_type in valid_types:
            AttributeConfig(key="test", type=attr_type)
        
        # Invalid type
        with pytest.raises(ValueError):
            AttributeConfig(key="test", type="invalid_type")


class TestValidationRule:
    """Test cases for ValidationRule model."""
    
    def test_validation_rule_creation(self):
        """Test creating a valid validation rule."""
        rule = ValidationRule(
            field="price",
            rule_type="range",
            parameters={"min": 0, "max": 100000},
            error_message="Price must be between 0 and 100000"
        )
        
        assert rule.field == "price"
        assert rule.rule_type == "range"
        assert rule.parameters["min"] == 0
        assert rule.parameters["max"] == 100000
        assert rule.error_message == "Price must be between 0 and 100000"
    
    def test_validation_rule_types(self):
        """Test different validation rule types."""
        rule_types = ["required", "length", "pattern", "range", "custom"]
        
        for rule_type in rule_types:
            rule = ValidationRule(
                field="test_field",
                rule_type=rule_type,
                parameters={"test": "value"}
            )
            assert rule.rule_type == rule_type
    
    def test_validation_rule_field_validation(self):
        """Test validation rule field validation."""
        # Empty field name
        with pytest.raises(ValueError):
            ValidationRule(
                field="",
                rule_type="required",
                parameters={}
            )


class TestMetricConfig:
    """Test cases for MetricConfig model."""
    
    def test_metric_config_creation(self):
        """Test creating a valid metric config."""
        config = MetricConfig(
            name="attribute_coverage",
            params={"required_attributes": ["brand", "price"]},
            threshold=0.95
        )
        
        assert config.name == "attribute_coverage"
        assert "required_attributes" in config.params
        assert config.threshold == 0.95
    
    def test_metric_config_without_threshold(self):
        """Test metric config without threshold."""
        config = MetricConfig(
            name="extraction_speed",
            params={"unit": "items_per_second"}
        )
        
        assert config.threshold is None
        assert config.params["unit"] == "items_per_second"


class TestRuntimeConfig:
    """Test cases for RuntimeConfig model."""
    
    def test_runtime_config_defaults(self):
        """Test runtime config with default values."""
        config = RuntimeConfig()
        
        assert config.language_default == "auto"
        assert config.dedupe_by == []
        assert config.skip_if_missing == []
        assert config.batch_size == 100
        assert config.timeout_seconds == 300
    
    def test_runtime_config_custom_values(self):
        """Test runtime config with custom values."""
        config = RuntimeConfig(
            language_default="zh-CN",
            dedupe_by=["title", "url"],
            skip_if_missing=["title"],
            batch_size=50,
            timeout_seconds=600
        )
        
        assert config.language_default == "zh-CN"
        assert config.dedupe_by == ["title", "url"]
        assert config.skip_if_missing == ["title"]
        assert config.batch_size == 50
        assert config.timeout_seconds == 600
    
    def test_runtime_config_language_validation(self):
        """Test language code validation."""
        # Valid language codes
        valid_langs = ["en", "zh-CN", "zh-TW", "auto"]
        for lang in valid_langs:
            RuntimeConfig(language_default=lang)
        
        # Note: Current regex pattern is permissive for compatibility


class TestDomainPack:
    """Test cases for DomainPack model."""
    
    def test_domain_pack_creation_minimal(self):
        """Test creating a minimal domain pack."""
        input_mapping = {
            "title": ["product_name", "name"],
            "description": ["product_desc", "desc"],
            "language": ["lang", "language"]
        }
        
        output_schema = {
            "core_item": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "language": {"type": "string"}
            },
            "attributes": []
        }
        
        domain_pack = DomainPack(
            domain="ecommerce",
            taxonomy_version="2024.01",
            input_mapping=input_mapping,
            output_schema=output_schema
        )
        
        assert domain_pack.domain == "ecommerce"
        assert domain_pack.taxonomy_version == "2024.01"
        assert "title" in domain_pack.input_mapping
        assert "core_item" in domain_pack.output_schema
    
    def test_domain_pack_creation_complete(self):
        """Test creating a complete domain pack."""
        input_mapping = {
            "title": ["product_name"],
            "description": ["product_desc"],
            "language": ["lang"]
        }
        
        output_schema = {
            "core_item": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "language": {"type": "string"}
            },
            "attributes": [
                {"key": "brand", "type": "categorical"},
                {"key": "price", "type": "numeric"}
            ]
        }
        
        domain_pack = DomainPack(
            domain="ecommerce",
            taxonomy_version="2024.01",
            input_mapping=input_mapping,
            output_schema=output_schema,
            attribute_hints={
                "brand": {"examples": ["Apple", "Samsung"]},
                "price": {"required": True}
            },
            prompts={
                "extract_brand": "Extract brand from: {text}",
                "extract_price": "Extract price from: {text}"
            },
            evaluation={"metrics": ["accuracy", "coverage"]}
        )
        
        assert len(domain_pack.attribute_hints) == 2
        assert "extract_brand" in domain_pack.prompts
        assert domain_pack.evaluation is not None
    
    def test_domain_pack_domain_validation(self):
        """Test domain name validation."""
        base_config = {
            "taxonomy_version": "2024.01",
            "input_mapping": {
                "title": ["name"],
                "description": ["desc"],
                "language": ["lang"]
            },
            "output_schema": {
                "core_item": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"}
                },
                "attributes": []
            }
        }
        
        # Valid domains
        valid_domains = ["ecommerce", "zh-news", "social_media"]
        for domain in valid_domains:
            DomainPack(domain=domain, **base_config)
        
        # Invalid domains
        with pytest.raises(ValueError):
            DomainPack(domain="E-commerce", **base_config)
        
        with pytest.raises(ValueError):
            DomainPack(domain="", **base_config)
    
    def test_domain_pack_taxonomy_version_validation(self):
        """Test taxonomy version validation."""
        base_config = {
            "domain": "test",
            "input_mapping": {
                "title": ["name"],
                "description": ["desc"],
                "language": ["lang"]
            },
            "output_schema": {
                "core_item": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"}
                },
                "attributes": []
            }
        }
        
        # Valid versions
        valid_versions = ["2024.01", "2024.12", "2023.06-beta"]
        for version in valid_versions:
            DomainPack(taxonomy_version=version, **base_config)
        
        # Invalid versions
        with pytest.raises(ValueError):
            DomainPack(taxonomy_version="v2024.01", **base_config)
        
        with pytest.raises(ValueError):
            DomainPack(taxonomy_version="2024.1", **base_config)
    
    def test_domain_pack_input_mapping_validation(self):
        """Test input mapping validation."""
        base_config = {
            "domain": "test",
            "taxonomy_version": "2024.01",
            "output_schema": {
                "core_item": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"}
                },
                "attributes": []
            }
        }
        
        # Missing required fields
        with pytest.raises(ValueError, match="input_mapping must include core field: title"):
            DomainPack(
                input_mapping={
                    "description": ["desc"],
                    "language": ["lang"]
                },
                **base_config
            )
        
        # Empty field list
        with pytest.raises(ValueError, match="must be a non-empty list"):
            DomainPack(
                input_mapping={
                    "title": [],
                    "description": ["desc"],
                    "language": ["lang"]
                },
                **base_config
            )
    
    def test_domain_pack_output_schema_validation(self):
        """Test output schema validation."""
        base_config = {
            "domain": "test",
            "taxonomy_version": "2024.01",
            "input_mapping": {
                "title": ["name"],
                "description": ["desc"],
                "language": ["lang"]
            }
        }
        
        # Missing core_item
        with pytest.raises(ValueError, match="must include 'core_item' section"):
            DomainPack(
                output_schema={
                    "attributes": []
                },
                **base_config
            )
        
        # Missing attributes
        with pytest.raises(ValueError, match="must include 'attributes' section"):
            DomainPack(
                output_schema={
                    "core_item": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "language": {"type": "string"}
                    }
                },
                **base_config
            )
        
        # Missing required core fields
        with pytest.raises(ValueError, match="must include field: title"):
            DomainPack(
                output_schema={
                    "core_item": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "language": {"type": "string"}
                    },
                    "attributes": []
                },
                **base_config
            )
    
    def test_domain_pack_attribute_consistency(self):
        """Test attribute consistency validation."""
        base_config = {
            "domain": "test",
            "taxonomy_version": "2024.01",
            "input_mapping": {
                "title": ["name"],
                "description": ["desc"],
                "language": ["lang"]
            },
            "output_schema": {
                "core_item": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"}
                },
                "attributes": [
                    {"key": "brand", "type": "categorical"}
                ]
            }
        }
        
        # Valid - hints match attributes
        DomainPack(
            attribute_hints={"brand": {"examples": ["Apple"]}},
            **base_config
        )
        
        # Invalid - extra hints
        with pytest.raises(ValueError, match="attribute_hints contains keys not in output_schema.attributes"):
            DomainPack(
                attribute_hints={
                    "brand": {"examples": ["Apple"]},
                    "price": {"required": True}  # Not in attributes
                },
                **base_config
            )
    
    def test_domain_pack_get_attribute_config(self):
        """Test getting attribute configuration."""
        output_schema = {
            "core_item": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "language": {"type": "string"}
            },
            "attributes": [
                {"key": "brand", "type": "categorical"},
                {"key": "price", "type": "numeric"}
            ]
        }
        
        domain_pack = DomainPack(
            domain="test",
            taxonomy_version="2024.01",
            input_mapping={
                "title": ["name"],
                "description": ["desc"],
                "language": ["lang"]
            },
            output_schema=output_schema,
            attribute_hints={
                "brand": {"examples": ["Apple", "Samsung"], "required": True}
            }
        )
        
        # Get existing attribute
        brand_config = domain_pack.get_attribute_config("brand")
        assert brand_config is not None
        assert brand_config.key == "brand"
        assert brand_config.type == "categorical"
        assert brand_config.required is True
        assert "Apple" in brand_config.examples
        
        # Get non-existing attribute
        missing_config = domain_pack.get_attribute_config("missing")
        assert missing_config is None
    
    def test_domain_pack_get_prompt_template(self):
        """Test getting prompt templates."""
        domain_pack = DomainPack(
            domain="ecommerce",
            taxonomy_version="2024.01",
            input_mapping={
                "title": ["name"],
                "description": ["desc"],
                "language": ["lang"]
            },
            output_schema={
                "core_item": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"}
                },
                "attributes": []
            },
            prompts={
                "extract_brand": "Extract brand from {{text}} for domain {{DOMAIN}}",
                "classify": "Classify this item: {{text}}"
            }
        )
        
        # Get existing prompt with substitution
        brand_prompt = domain_pack.get_prompt_template("extract_brand", text="iPhone 15")
        assert "iPhone 15" in brand_prompt
        assert "ecommerce" in brand_prompt
        
        # Get non-existing prompt
        missing_prompt = domain_pack.get_prompt_template("missing")
        assert missing_prompt is None


class TestEdgeCases:
    """Test edge cases and error conditions for domain models."""
    
    def test_unicode_in_domain_models(self):
        """Test Unicode support in domain models."""
        config = AttributeConfig(
            key="brand_zh",
            type="categorical",
            examples=["苹果", "三星", "华为"],
            notes="中文品牌名称"
        )
        
        assert "苹果" in config.examples
        assert "中文" in config.notes
        
        domain_pack = DomainPack(
            domain="zh_ecommerce",
            taxonomy_version="2024.01",
            input_mapping={
                "title": ["产品名称"],
                "description": ["产品描述"],
                "language": ["语言"]
            },
            output_schema={
                "core_item": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "language": {"type": "string"}
                },
                "attributes": []
            }
        )
        
        assert "产品名称" in domain_pack.input_mapping["title"]
    
    def test_model_serialization(self):
        """Test JSON serialization of domain models."""
        config = AttributeConfig(
            key="test",
            type="text"
        )
        
        # Should serialize without error
        json_data = config.model_dump_json()
        assert "test" in json_data
        
        # Should deserialize correctly
        data = config.model_dump()
        recreated = AttributeConfig(**data)
        assert recreated.key == config.key
        assert recreated.type == config.type
