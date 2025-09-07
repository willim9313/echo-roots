"""Schema merging and validation utilities.

This module provides utilities for merging domain-specific schemas with
the core schema, validating attribute definitions, and resolving conflicts.
It supports both schema-level merging and runtime validation.

Key components:
- SchemaMerger: Main class for schema merging operations
- AttributeValidator: Validates attribute definitions and values
- SchemaConflictResolver: Handles conflicts during merging
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from enum import Enum
from datetime import datetime
import json
import copy

from echo_roots.models.domain import DomainPack, AttributeConfig
from echo_roots.models.taxonomy import Attribute, AttributeValue
from echo_roots.models.core import AttributeExtraction


class ConflictResolution(Enum):
    """Strategies for resolving schema conflicts."""
    DOMAIN_WINS = "domain_wins"      # Domain-specific config takes precedence
    CORE_WINS = "core_wins"          # Core schema takes precedence
    STRICT = "strict"                # Raise error on conflicts
    MERGE = "merge"                  # Attempt to merge compatible configs


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class SchemaConflictError(Exception):
    """Raised when schema conflicts cannot be resolved."""
    pass


class AttributeValidator:
    """Validates attribute definitions and values against schema rules.
    
    Provides validation for both domain-specific attribute configurations
    and extracted attribute values.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize validator with domain pack configuration.
        
        Args:
            domain_pack: Domain pack with attribute definitions
        """
        self.domain_pack = domain_pack
        self.attribute_configs = self._build_attribute_configs()
    
    def _build_attribute_configs(self) -> Dict[str, AttributeConfig]:
        """Build attribute configurations from domain pack."""
        configs = {}
        
        if 'attributes' not in self.domain_pack.output_schema:
            return configs
        
        for attr_dict in self.domain_pack.output_schema['attributes']:
            if isinstance(attr_dict, dict) and 'key' in attr_dict:
                key = attr_dict['key']
                
                # Merge with hints if available
                config_data = attr_dict.copy()
                if key in self.domain_pack.attribute_hints:
                    config_data.update(self.domain_pack.attribute_hints[key])
                
                try:
                    configs[key] = AttributeConfig(**config_data)
                except Exception:
                    # Skip invalid configurations
                    continue
        
        return configs
    
    def validate_attribute_definition(self, key: str, definition: Dict[str, Any]) -> bool:
        """Validate an attribute definition.
        
        Args:
            key: Attribute key
            definition: Attribute definition dictionary
            
        Returns:
            True if valid
            
        Raises:
            SchemaValidationError: If validation fails
        """
        try:
            # Check required fields
            if 'type' not in definition:
                raise SchemaValidationError(f"Attribute '{key}' missing required 'type' field")
            
            # Validate type
            valid_types = {'categorical', 'text', 'numeric', 'boolean', 'date'}
            if definition['type'] not in valid_types:
                raise SchemaValidationError(
                    f"Attribute '{key}' has invalid type '{definition['type']}'. "
                    f"Valid types: {valid_types}"
                )
            
            # Validate categorical-specific rules
            if definition['type'] == 'categorical':
                if 'allow_values' in definition and not definition['allow_values']:
                    raise SchemaValidationError(
                        f"Categorical attribute '{key}' must have non-empty 'allow_values'"
                    )
            
            # Validate constraints
            if 'max_length' in definition:
                max_length = definition['max_length']
                if not isinstance(max_length, int) or max_length <= 0:
                    raise SchemaValidationError(
                        f"Attribute '{key}' max_length must be a positive integer"
                    )
            
            return True
            
        except Exception as e:
            if isinstance(e, SchemaValidationError):
                raise
            raise SchemaValidationError(f"Invalid attribute definition for '{key}': {e}")
    
    def validate_attribute_value(self, key: str, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate an attribute value against its configuration.
        
        Args:
            key: Attribute key
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if key not in self.attribute_configs:
            return True, None  # No validation rules defined
        
        config = self.attribute_configs[key]
        
        try:
            # Check required
            if config.required and (value is None or value == ''):
                return False, f"Attribute '{key}' is required but missing"
            
            # Skip validation for null/empty optional values
            if value is None or value == '':
                return True, None
            
            # Type-specific validation
            if config.type == 'categorical':
                if config.allow_values and str(value) not in config.allow_values:
                    return False, f"Value '{value}' not in allowed values for '{key}': {config.allow_values}"
            
            elif config.type == 'text':
                if not isinstance(value, str):
                    return False, f"Attribute '{key}' must be a string"
                
                if config.max_length and len(value) > config.max_length:
                    return False, f"Attribute '{key}' exceeds max length {config.max_length}"
                
                if config.pattern:
                    import re
                    if not re.match(config.pattern, value):
                        return False, f"Attribute '{key}' does not match required pattern"
            
            elif config.type == 'numeric':
                try:
                    float(value)
                except (ValueError, TypeError):
                    return False, f"Attribute '{key}' must be numeric"
            
            elif config.type == 'boolean':
                if not isinstance(value, bool) and str(value).lower() not in ('true', 'false', '1', '0'):
                    return False, f"Attribute '{key}' must be boolean"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error for '{key}': {e}"
    
    def validate_extraction(self, extraction: AttributeExtraction) -> List[str]:
        """Validate an attribute extraction against schema rules.
        
        Args:
            extraction: AttributeExtraction to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        is_valid, error = self.validate_attribute_value(extraction.name, extraction.value)
        if not is_valid:
            errors.append(error)
        
        # Validate confidence range
        if extraction.confidence < 0.0 or extraction.confidence > 1.0:
            errors.append(f"Confidence for '{extraction.name}' must be between 0.0 and 1.0")
        
        return errors


class SchemaConflictResolver:
    """Handles conflicts during schema merging operations."""
    
    def __init__(self, resolution_strategy: ConflictResolution = ConflictResolution.DOMAIN_WINS):
        """Initialize conflict resolver with strategy.
        
        Args:
            resolution_strategy: Strategy for resolving conflicts
        """
        self.resolution_strategy = resolution_strategy
    
    def resolve_attribute_conflict(
        self, 
        key: str, 
        core_definition: Dict[str, Any], 
        domain_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conflicting attribute definitions.
        
        Args:
            key: Attribute key
            core_definition: Core schema attribute definition
            domain_definition: Domain-specific attribute definition
            
        Returns:
            Resolved attribute definition
            
        Raises:
            SchemaConflictError: If conflict cannot be resolved
        """
        if self.resolution_strategy == ConflictResolution.CORE_WINS:
            return core_definition
        
        elif self.resolution_strategy == ConflictResolution.DOMAIN_WINS:
            return domain_definition
        
        elif self.resolution_strategy == ConflictResolution.STRICT:
            # Check for actual conflicts
            conflicts = self._find_conflicts(core_definition, domain_definition)
            if conflicts:
                raise SchemaConflictError(
                    f"Strict mode: conflicting definitions for attribute '{key}': {conflicts}"
                )
            return domain_definition
        
        elif self.resolution_strategy == ConflictResolution.MERGE:
            return self._merge_definitions(core_definition, domain_definition)
        
        else:
            raise ValueError(f"Unknown resolution strategy: {self.resolution_strategy}")
    
    def _find_conflicts(self, core_def: Dict[str, Any], domain_def: Dict[str, Any]) -> List[str]:
        """Find conflicting fields between definitions."""
        conflicts = []
        
        for key in core_def:
            if key in domain_def and core_def[key] != domain_def[key]:
                conflicts.append(f"{key}: core={core_def[key]}, domain={domain_def[key]}")
        
        return conflicts
    
    def _merge_definitions(self, core_def: Dict[str, Any], domain_def: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two attribute definitions intelligently."""
        merged = copy.deepcopy(core_def)
        
        for key, value in domain_def.items():
            if key not in merged:
                # Add new fields from domain definition
                merged[key] = value
            elif key == 'allow_values':
                # Merge allowed values (union)
                if isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = list(set(merged[key] + value))
                else:
                    merged[key] = value  # Domain wins for non-list values
            elif key == 'examples':
                # Merge examples (union)
                if isinstance(merged[key], list) and isinstance(value, list):
                    merged[key] = list(set(merged[key] + value))
                else:
                    merged[key] = value
            elif key in ('required', 'max_length', 'pattern'):
                # Domain-specific constraints take precedence
                merged[key] = value
            else:
                # For other fields, domain wins
                merged[key] = value
        
        return merged


class SchemaMerger:
    """Main class for schema merging operations.
    
    Merges domain-specific schemas with core schemas, validates the result,
    and provides utilities for working with merged schemas.
    """
    
    def __init__(self, 
                 conflict_resolution: ConflictResolution = ConflictResolution.DOMAIN_WINS,
                 validate_merge: bool = True):
        """Initialize schema merger.
        
        Args:
            conflict_resolution: Strategy for resolving conflicts
            validate_merge: Whether to validate merged schemas
        """
        self.conflict_resolver = SchemaConflictResolver(conflict_resolution)
        self.validate_merge = validate_merge
        self._core_schema = self._load_core_schema()
    
    def _load_core_schema(self) -> Dict[str, Any]:
        """Load the core schema definition."""
        # This would typically load from a schema file
        # For now, return the basic core schema structure
        return {
            'core_item': {
                'id': {'type': 'string', 'required': True},
                'title': {'type': 'string', 'required': True},
                'description': {'type': 'string', 'required': False},
                'language': {'type': 'string', 'required': True},
                'source': {'type': 'object', 'required': True},
                'metadata': {'type': 'object', 'required': False},
            },
            'attributes': {}  # Base attributes (can be extended)
        }
    
    def merge_schemas(self, domain_pack: DomainPack) -> Dict[str, Any]:
        """Merge domain pack schema with core schema.
        
        Args:
            domain_pack: Domain pack with schema to merge
            
        Returns:
            Merged schema definition
            
        Raises:
            SchemaValidationError: If validation fails
            SchemaConflictError: If conflicts cannot be resolved
        """
        merged_schema = copy.deepcopy(self._core_schema)
        domain_schema = domain_pack.output_schema
        
        # Merge core_item definitions
        if 'core_item' in domain_schema:
            for field, definition in domain_schema['core_item'].items():
                if field in merged_schema['core_item']:
                    # Resolve conflict
                    merged_schema['core_item'][field] = self.conflict_resolver.resolve_attribute_conflict(
                        field, merged_schema['core_item'][field], definition
                    )
                else:
                    # Add new field
                    merged_schema['core_item'][field] = definition
        
        # Merge attributes
        if 'attributes' in domain_schema:
            validator = AttributeValidator(domain_pack)
            
            for attr_dict in domain_schema['attributes']:
                if isinstance(attr_dict, dict) and 'key' in attr_dict:
                    key = attr_dict['key']
                    
                    # Validate the attribute definition
                    if self.validate_merge:
                        validator.validate_attribute_definition(key, attr_dict)
                    
                    # Add to merged schema
                    if key in merged_schema['attributes']:
                        # Resolve conflict
                        merged_schema['attributes'][key] = self.conflict_resolver.resolve_attribute_conflict(
                            key, merged_schema['attributes'][key], attr_dict
                        )
                    else:
                        # Add new attribute
                        merged_schema['attributes'][key] = attr_dict
        
        # Add metadata about the merge
        merged_schema['_merge_info'] = {
            'domain': domain_pack.domain,
            'taxonomy_version': domain_pack.taxonomy_version,
            'merged_at': str(datetime.now()),
            'total_attributes': len(merged_schema['attributes']),
        }
        
        return merged_schema
    
    def validate_merged_schema(self, schema: Dict[str, Any]) -> List[str]:
        """Validate a merged schema for consistency and completeness.
        
        Args:
            schema: Merged schema to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required core fields
        required_core_fields = {'id', 'title', 'language'}
        if 'core_item' in schema:
            for field in required_core_fields:
                if field not in schema['core_item']:
                    errors.append(f"Missing required core field: {field}")
        else:
            errors.append("Missing core_item section in schema")
        
        # Validate attribute definitions
        if 'attributes' in schema:
            for key, definition in schema['attributes'].items():
                try:
                    # Create a dummy domain pack for validation
                    dummy_pack = DomainPack(
                        domain="validation",
                        taxonomy_version="1.0.0",
                        input_mapping={'title': ['title'], 'description': ['desc'], 'language': ['lang']},
                        output_schema={'core_item': schema['core_item'], 'attributes': []}
                    )
                    validator = AttributeValidator(dummy_pack)
                    validator.validate_attribute_definition(key, definition)
                except Exception as e:
                    errors.append(f"Invalid attribute '{key}': {e}")
        
        return errors
    
    def get_attribute_schema(self, schema: Dict[str, Any], attribute_key: str) -> Optional[Dict[str, Any]]:
        """Get schema definition for a specific attribute.
        
        Args:
            schema: Merged schema
            attribute_key: Attribute key to look up
            
        Returns:
            Attribute schema if found, None otherwise
        """
        if 'attributes' not in schema:
            return None
        
        return schema['attributes'].get(attribute_key)
    
    def list_attributes(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """List all attributes in a merged schema.
        
        Args:
            schema: Merged schema
            
        Returns:
            List of attribute definitions with metadata
        """
        if 'attributes' not in schema:
            return []
        
        attributes = []
        for key, definition in schema['attributes'].items():
            attr_info = {
                'key': key,
                'type': definition.get('type', 'unknown'),
                'required': definition.get('required', False),
                'has_constraints': any(k in definition for k in ('allow_values', 'max_length', 'pattern')),
                'definition': definition
            }
            attributes.append(attr_info)
        
        return attributes
    
    def export_schema(self, schema: Dict[str, Any], format: str = 'json') -> str:
        """Export merged schema in specified format.
        
        Args:
            schema: Merged schema to export
            format: Export format ('json' or 'yaml')
            
        Returns:
            Serialized schema string
        """
        if format.lower() == 'json':
            return json.dumps(schema, indent=2, ensure_ascii=False)
        elif format.lower() == 'yaml':
            import yaml
            return yaml.dump(schema, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")


def merge_domain_schemas(*domain_packs: DomainPack, 
                        conflict_resolution: ConflictResolution = ConflictResolution.DOMAIN_WINS) -> Dict[str, Any]:
    """Convenience function to merge multiple domain pack schemas.
    
    Args:
        *domain_packs: Domain packs to merge
        conflict_resolution: Strategy for resolving conflicts
        
    Returns:
        Merged schema from all domain packs
    """
    merger = SchemaMerger(conflict_resolution=conflict_resolution)
    
    # Start with the first domain pack
    if not domain_packs:
        return merger._core_schema
    
    merged = merger.merge_schemas(domain_packs[0])
    
    # Merge additional domain packs
    for domain_pack in domain_packs[1:]:
        # Create a temporary domain pack with current merged schema
        # This is a simplified approach - in practice, you might want
        # more sophisticated multi-way merging
        domain_schema = merger.merge_schemas(domain_pack)
        
        # Simple merge of attributes
        if 'attributes' in domain_schema:
            merged.setdefault('attributes', {}).update(domain_schema['attributes'])
    
    return merged
