"""Tests for domain adapter components.

Tests for domain pack loading, field mapping, data transformation,
and schema merging functionality.
"""

import pytest
import yaml
import json
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from echo_roots.domain.loader import DomainPackLoader, DomainPackLoadError
from echo_roots.domain.adapter import DomainAdapter, FieldMapper, DataTransformer
from echo_roots.domain.merger import (
    SchemaMerger, AttributeValidator, SchemaConflictResolver,
    ConflictResolution, SchemaValidationError, SchemaConflictError
)
from echo_roots.models.domain import DomainPack
from echo_roots.models.core import IngestionItem


@pytest.fixture
def sample_domain_yaml():
    """Sample domain pack YAML content."""
    return {
        'domain': 'ecommerce',
        'taxonomy_version': '2024.01',
        'input_mapping': {
            'title': ['name', 'product_name'],
            'description': ['description', 'desc'],
            'language': ['lang']
        },
        'output_schema': {
            'core_item': {
                'id': {'type': 'string', 'required': True},
                'title': {'type': 'string', 'required': True},
                'description': {'type': 'string', 'required': False},
                'language': {'type': 'string', 'required': True}
            },
            'attributes': [
                {
                    'key': 'brand',
                    'type': 'categorical',
                    'required': True,
                    'allow_values': ['Nike', 'Adidas', 'Puma']
                },
                {
                    'key': 'price',
                    'type': 'numeric',
                    'required': False
                }
            ]
        },
        'attribute_hints': {
            'brand': {'examples': ['Nike Air Max', 'Adidas Boost']},
            'price': {'pattern': r'\\d+\\.\\d{2}'}
        }
    }


@pytest.fixture
def temp_domain_file(sample_domain_yaml):
    """Create temporary domain YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_domain_yaml, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def temp_domain_dir(sample_domain_yaml):
    """Create temporary directory with domain pack files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        domain_dir = Path(temp_dir) / "ecommerce"
        domain_dir.mkdir()
        
        with open(domain_dir / "domain.yaml", 'w') as f:
            yaml.dump(sample_domain_yaml, f)
        
        yield domain_dir


class TestDomainPackLoader:
    """Test domain pack loading functionality."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = DomainPackLoader()
        assert loader.cache_size == 128
        assert loader.validate_on_load is True
    
    def test_load_valid_domain_file(self, temp_domain_file):
        """Test loading a valid domain pack file."""
        loader = DomainPackLoader()
        domain_pack = loader.load(temp_domain_file)
        
        assert domain_pack.domain == 'ecommerce'
        assert domain_pack.taxonomy_version == '2024.01'
        assert 'title' in domain_pack.input_mapping
    
    def test_load_valid_domain_directory(self, temp_domain_dir):
        """Test loading from a domain directory."""
        loader = DomainPackLoader()
        domain_pack = loader.load_from_directory(str(temp_domain_dir))
        
        assert domain_pack.domain == 'ecommerce'
        assert domain_pack.taxonomy_version == '2024.01'
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        loader = DomainPackLoader()
        
        with pytest.raises(DomainPackLoadError, match="not found"):
            loader.load('nonexistent.yaml')
    
    def test_caching_behavior(self, temp_domain_file):
        """Test that domains are cached after loading."""
        loader = DomainPackLoader()
        
        # First load
        domain_pack1 = loader.load(temp_domain_file)
        
        # Second load should return cached version
        domain_pack2 = loader.load(temp_domain_file)
        assert domain_pack1 is domain_pack2


class TestFieldMapper:
    """Test field mapping functionality."""
    
    def test_map_input_fields_basic(self, sample_domain_yaml):
        """Test basic field mapping."""
        domain_pack = DomainPack(**sample_domain_yaml)
        mapper = FieldMapper(domain_pack)
        
        raw_data = {
            'name': 'Test Product',
            'description': 'A test product',
            'lang': 'en'
        }
        
        mapped = mapper.map_input_fields(raw_data)
        
        assert mapped['title'] == 'Test Product'
        assert mapped['description'] == 'A test product'
        assert mapped['language'] == 'en'
    
    def test_map_input_fields_priority(self, sample_domain_yaml):
        """Test field mapping with priority (first match wins)."""
        domain_pack = DomainPack(**sample_domain_yaml)
        mapper = FieldMapper(domain_pack)
        
        raw_data = {
            'name': 'Primary Name',
            'product_name': 'Secondary Name',
            'description': 'Description text'
        }
        
        mapped = mapper.map_input_fields(raw_data)
        
        # First mapping should win
        assert mapped['title'] == 'Primary Name'
        assert mapped['description'] == 'Description text'
    
    def test_map_input_fields_missing_optional(self, sample_domain_yaml):
        """Test field mapping with missing optional fields."""
        domain_pack = DomainPack(**sample_domain_yaml)
        mapper = FieldMapper(domain_pack)
        
        raw_data = {
            'name': 'Test Product',
            'lang': 'en'
            # Missing description - should be optional
        }
        
        mapped = mapper.map_input_fields(raw_data)
        
        assert mapped['title'] == 'Test Product'
        assert mapped['language'] == 'en'
        assert 'description' not in mapped


class TestDataTransformer:
    """Test data transformation functionality."""
    
    def test_normalize_text_field(self, sample_domain_yaml):
        """Test text field normalization."""
        domain_pack = DomainPack(**sample_domain_yaml)
        transformer = DataTransformer(domain_pack)
        
        # Test trimming and basic normalization
        result = transformer.normalize_text_field("  Test Product  ")
        assert result == "Test Product"
        
        # Test empty string handling
        result = transformer.normalize_text_field("")
        assert result is None
        
        # Test None handling
        result = transformer.normalize_text_field(None)
        assert result is None
    
    def test_apply_normalization_rules(self, sample_domain_yaml):
        """Test applying normalization rules."""
        domain_pack = DomainPack(**sample_domain_yaml)
        transformer = DataTransformer(domain_pack)
        
        data = {
            'title': '  Test Product  ',
            'description': '\\tA great product\\n',
            'language': 'EN',
            'extra_field': 'should_remain'
        }
        
        normalized = transformer.apply_normalization_rules(data)
        
        assert normalized['title'] == 'Test Product'
        assert normalized['description'] == 'A great product'
        assert normalized['language'] == 'en'  # Lowercase normalization
        assert normalized['extra_field'] == 'should_remain'
    
    def test_transform_to_ingestion_item(self, sample_domain_yaml):
        """Test transformation to IngestionItem."""
        domain_pack = DomainPack(**sample_domain_yaml)
        transformer = DataTransformer(domain_pack)
        
        normalized_data = {
            'title': 'Test Product',
            'description': 'A test product',
            'language': 'en',
            'brand': 'Nike',
            'price': '99.99'
        }
        
        item = transformer.transform_to_ingestion_item(normalized_data)
        
        assert isinstance(item, IngestionItem)
        assert item.title == 'Test Product'
        assert item.description == 'A test product'
        assert item.language == 'en'
        
        # Check attributes
        brand_attr = next((attr for attr in item.attributes if attr.name == 'brand'), None)
        assert brand_attr is not None
        assert brand_attr.value == 'Nike'
        
        price_attr = next((attr for attr in item.attributes if attr.name == 'price'), None)
        assert price_attr is not None
        assert price_attr.value == '99.99'


class TestDomainAdapter:
    """Test main domain adapter functionality."""
    
    def test_adapter_initialization(self, sample_domain_yaml):
        """Test adapter initialization."""
        domain_pack = DomainPack(**sample_domain_yaml)
        adapter = DomainAdapter(domain_pack)
        
        assert adapter.domain_pack is not None
        assert adapter.field_mapper is not None
        assert adapter.data_transformer is not None
    
    def test_adapt_single_item(self, sample_domain_yaml):
        """Test adapting a single data item."""
        domain_pack = DomainPack(**sample_domain_yaml)
        adapter = DomainAdapter(domain_pack)
        
        raw_item = {
            'name': 'Nike Air Max',
            'description': 'Great running shoes',
            'lang': 'en',
            'brand': 'Nike',
            'price': '129.99'
        }
        
        result = adapter.adapt(raw_item)
        
        assert isinstance(result, IngestionItem)
        assert result.title == 'Nike Air Max'
        assert result.language == 'en'
        assert len(result.attributes) == 2  # brand and price
    
    def test_adapt_batch_items(self, sample_domain_yaml):
        """Test adapting multiple data items."""
        domain_pack = DomainPack(**sample_domain_yaml)
        adapter = DomainAdapter(domain_pack)
        
        raw_items = [
            {
                'name': 'Nike Air Max',
                'description': 'Running shoes',
                'lang': 'en',
                'brand': 'Nike'
            },
            {
                'name': 'Adidas Boost',
                'description': 'Comfortable shoes',
                'lang': 'en',
                'brand': 'Adidas'
            }
        ]
        
        results = adapter.adapt_batch(raw_items)
        
        assert len(results) == 2
        assert all(isinstance(item, IngestionItem) for item in results)
        assert results[0].title == 'Nike Air Max'
        assert results[1].title == 'Adidas Boost'
    
    def test_from_domain_name_classmethod(self, temp_domain_dir):
        """Test creating adapter from domain name."""
        # Need to adjust for the expected directory structure
        base_path = temp_domain_dir.parent
        adapter = DomainAdapter.from_domain_name('ecommerce', base_path)
        
        assert adapter.domain_pack.domain == 'ecommerce'


class TestAttributeValidator:
    """Test attribute validation functionality."""
    
    def test_validator_initialization(self, sample_domain_yaml):
        """Test validator initialization."""
        domain_pack = DomainPack(**sample_domain_yaml)
        validator = AttributeValidator(domain_pack)
        
        assert 'brand' in validator.attribute_configs
        assert 'price' in validator.attribute_configs
    
    def test_validate_attribute_definition_valid(self, sample_domain_yaml):
        """Test validating valid attribute definitions."""
        domain_pack = DomainPack(**sample_domain_yaml)
        validator = AttributeValidator(domain_pack)
        
        # Valid categorical attribute
        definition = {
            'type': 'categorical',
            'required': True,
            'allow_values': ['A', 'B', 'C']
        }
        
        assert validator.validate_attribute_definition('test_attr', definition)
    
    def test_validate_attribute_definition_invalid(self, sample_domain_yaml):
        """Test validating invalid attribute definitions."""
        domain_pack = DomainPack(**sample_domain_yaml)
        validator = AttributeValidator(domain_pack)
        
        # Missing type
        with pytest.raises(SchemaValidationError, match="missing required 'type' field"):
            validator.validate_attribute_definition('test_attr', {})
        
        # Invalid type
        with pytest.raises(SchemaValidationError, match="invalid type"):
            validator.validate_attribute_definition('test_attr', {'type': 'invalid_type'})
    
    def test_validate_attribute_value_valid(self, sample_domain_yaml):
        """Test validating valid attribute values."""
        domain_pack = DomainPack(**sample_domain_yaml)
        validator = AttributeValidator(domain_pack)
        
        # Valid brand value
        is_valid, error = validator.validate_attribute_value('brand', 'Nike')
        assert is_valid
        assert error is None
        
        # Valid price value
        is_valid, error = validator.validate_attribute_value('price', '99.99')
        assert is_valid
        assert error is None
    
    def test_validate_attribute_value_invalid(self, sample_domain_yaml):
        """Test validating invalid attribute values."""
        domain_pack = DomainPack(**sample_domain_yaml)
        validator = AttributeValidator(domain_pack)
        
        # Invalid brand value (not in allow_values)
        is_valid, error = validator.validate_attribute_value('brand', 'UnknownBrand')
        assert not is_valid
        assert 'not in allowed values' in error


class TestSchemaMerger:
    """Test schema merging functionality."""
    
    def test_merger_initialization(self):
        """Test merger initialization."""
        merger = SchemaMerger()
        assert merger.conflict_resolver is not None
        assert merger.validate_merge is True
    
    def test_merge_schemas_basic(self, sample_domain_yaml):
        """Test basic schema merging."""
        domain_pack = DomainPack(**sample_domain_yaml)
        merger = SchemaMerger()
        
        merged_schema = merger.merge_schemas(domain_pack)
        
        assert 'core_item' in merged_schema
        assert 'attributes' in merged_schema
        assert 'brand' in merged_schema['attributes']
        assert 'price' in merged_schema['attributes']
        assert '_merge_info' in merged_schema
    
    def test_list_attributes(self, sample_domain_yaml):
        """Test listing attributes in merged schema."""
        domain_pack = DomainPack(**sample_domain_yaml)
        merger = SchemaMerger()
        
        merged_schema = merger.merge_schemas(domain_pack)
        attributes = merger.list_attributes(merged_schema)
        
        assert len(attributes) == 2
        brand_attr = next((attr for attr in attributes if attr['key'] == 'brand'), None)
        assert brand_attr is not None
        assert brand_attr['type'] == 'categorical'
        assert brand_attr['required'] is True
    
    def test_export_schema_json(self, sample_domain_yaml):
        """Test exporting schema as JSON."""
        domain_pack = DomainPack(**sample_domain_yaml)
        merger = SchemaMerger()
        
        merged_schema = merger.merge_schemas(domain_pack)
        json_export = merger.export_schema(merged_schema, 'json')
        
        # Should be valid JSON
        parsed = json.loads(json_export)
        assert 'attributes' in parsed
        assert 'brand' in parsed['attributes']


class TestSchemaConflictResolver:
    """Test schema conflict resolution."""
    
    def test_resolver_domain_wins(self):
        """Test domain wins resolution strategy."""
        resolver = SchemaConflictResolver(ConflictResolution.DOMAIN_WINS)
        
        core_def = {'type': 'text', 'required': False}
        domain_def = {'type': 'categorical', 'required': True, 'allow_values': ['A', 'B']}
        
        result = resolver.resolve_attribute_conflict('test_attr', core_def, domain_def)
        
        assert result == domain_def
    
    def test_resolver_core_wins(self):
        """Test core wins resolution strategy."""
        resolver = SchemaConflictResolver(ConflictResolution.CORE_WINS)
        
        core_def = {'type': 'text', 'required': False}
        domain_def = {'type': 'categorical', 'required': True}
        
        result = resolver.resolve_attribute_conflict('test_attr', core_def, domain_def)
        
        assert result == core_def
    
    def test_resolver_strict_mode_conflict(self):
        """Test strict mode with conflicts."""
        resolver = SchemaConflictResolver(ConflictResolution.STRICT)
        
        core_def = {'type': 'text', 'required': False}
        domain_def = {'type': 'categorical', 'required': True}
        
        with pytest.raises(SchemaConflictError, match="conflicting definitions"):
            resolver.resolve_attribute_conflict('test_attr', core_def, domain_def)
    
    def test_resolver_merge_mode(self):
        """Test merge resolution strategy."""
        resolver = SchemaConflictResolver(ConflictResolution.MERGE)
        
        core_def = {'type': 'text', 'examples': ['example1']}
        domain_def = {'required': True, 'examples': ['example2']}
        
        result = resolver.resolve_attribute_conflict('test_attr', core_def, domain_def)
        
        assert result['type'] == 'text'  # From core
        assert result['required'] is True  # From domain
        assert set(result['examples']) == {'example1', 'example2'}  # Merged
