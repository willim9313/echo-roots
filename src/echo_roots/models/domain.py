"""Domain pack models for YAML configuration.

This module defines the Pydantic v2 models for parsing and validating
domain.yaml files. Domain packs provide flexible configuration for
adapting the core framework to specific domains like e-commerce,
media, or knowledge graphs.

Key models:
- DomainPack: Complete domain configuration
- AttributeConfig: Domain-specific attribute definitions
- ValidationRule: Custom validation rules for attributes
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class AttributeConfig(BaseModel):
    """Configuration for a domain-specific attribute.
    
    Defines how an attribute should be processed, validated, and normalized
    within a specific domain context.
    
    Attributes:
        key: Attribute identifier (e.g., "brand", "color")
        type: Data type for validation
        examples: Example values for LLM guidance
        notes: Human-readable notes about the attribute
        normalize_to_lower: Whether to normalize values to lowercase
        allow_values: Restricted set of allowed values
        required: Whether this attribute is required
        max_length: Maximum length for text values
        pattern: Regex pattern for validation
    """
    
    key: str = Field(
        description="Attribute identifier (e.g., 'brand', 'color')",
        min_length=1,
        max_length=50,
        pattern=r'^[a-z][a-z0-9_]*$'
    )
    type: str = Field(
        description="Data type for validation",
        pattern=r'^(categorical|text|numeric|boolean|date)$'
    )
    examples: List[str] = Field(
        default_factory=list,
        description="Example values for LLM guidance"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Human-readable notes about the attribute",
        max_length=500
    )
    normalize_to_lower: bool = Field(
        default=False,
        description="Whether to normalize values to lowercase"
    )
    allow_values: Optional[List[str]] = Field(
        default=None,
        description="Restricted set of allowed values"
    )
    required: bool = Field(
        default=False,
        description="Whether this attribute is required"
    )
    max_length: Optional[int] = Field(
        default=None,
        description="Maximum length for text values",
        ge=1,
        le=5000
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern for validation"
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class ValidationRule(BaseModel):
    """Custom validation rule for domain-specific processing.
    
    Defines rules for validating and normalizing data within
    a domain context.
    
    Attributes:
        field: Target field for validation
        rule_type: Type of validation rule
        parameters: Rule-specific parameters
        error_message: Custom error message for validation failures
    """
    
    field: str = Field(
        description="Target field for validation",
        min_length=1,
        max_length=100
    )
    rule_type: str = Field(
        description="Type of validation rule",
        pattern=r'^(required|length|pattern|range|custom)$'
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule-specific parameters"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Custom error message for validation failures",
        max_length=200
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class MetricConfig(BaseModel):
    """Configuration for domain-specific evaluation metrics.
    
    Defines metrics to track for quality assessment and
    performance monitoring within a domain.
    
    Attributes:
        name: Metric identifier
        params: Metric-specific parameters
        threshold: Optional threshold for alerts/gating
    """
    
    name: str = Field(
        description="Metric identifier",
        min_length=1,
        max_length=100
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metric-specific parameters"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Optional threshold for alerts/gating"
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class RuntimeConfig(BaseModel):
    """Runtime configuration for domain processing.
    
    Defines behavior settings for processing items within
    a specific domain context.
    
    Attributes:
        language_default: Default language when not specified
        dedupe_by: Fields to use for deduplication
        skip_if_missing: Required fields - skip item if missing
        batch_size: Processing batch size
        timeout_seconds: Processing timeout in seconds
    """
    
    language_default: str = Field(
        default="auto",
        description="Default language when not specified",
        pattern=r'^[a-z]{2}(-[A-Z]{2})?|auto$'
    )
    dedupe_by: List[str] = Field(
        default_factory=list,
        description="Fields to use for deduplication"
    )
    skip_if_missing: List[str] = Field(
        default_factory=list,
        description="Required fields - skip item if missing"
    )
    batch_size: int = Field(
        default=100,
        description="Processing batch size",
        ge=1,
        le=10000
    )
    timeout_seconds: int = Field(
        default=300,
        description="Processing timeout in seconds",
        ge=1,
        le=3600
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class DomainPack(BaseModel):
    """Complete domain pack configuration.
    
    Represents a full domain.yaml configuration file with all
    necessary settings for adapting the core framework to a
    specific domain like e-commerce, media, or knowledge graphs.
    
    Attributes:
        domain: Domain identifier (e.g., "ecommerce", "zh-news")
        taxonomy_version: Version identifier for the taxonomy
        input_mapping: Mapping from source fields to core schema fields
        output_schema: Core item schema and domain-specific attributes
        attribute_hints: Hints for LLM processing of attributes
        rules: Normalization and validation rules
        prompts: Domain-specific prompt templates
        evaluation: Evaluation metrics configuration
        runtime: Runtime behavior settings
    """
    
    domain: str = Field(
        description="Domain identifier (e.g., 'ecommerce', 'zh-news')",
        min_length=1,
        max_length=50,
        pattern=r'^[a-z][a-z0-9_-]*$'
    )
    taxonomy_version: str = Field(
        description="Version identifier for the taxonomy",
        min_length=1,
        max_length=20,
        pattern=r'^[0-9]{4}\.[0-9]{2}(-[a-z0-9]+)?$'
    )
    input_mapping: Dict[str, List[str]] = Field(
        description="Mapping from source fields to core schema fields"
    )
    output_schema: Dict[str, Any] = Field(
        description="Core item schema and domain-specific attributes"
    )
    attribute_hints: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Hints for LLM processing of attributes"
    )
    rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Normalization and validation rules"
    )
    prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Domain-specific prompt templates"
    )
    evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evaluation metrics configuration"
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Runtime behavior settings"
    )
    
    @field_validator('input_mapping')
    @classmethod
    def validate_input_mapping(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Ensure input mapping has required core fields."""
        required_core_fields = ['title', 'description', 'language']
        
        for field in required_core_fields:
            if field not in v:
                raise ValueError(f"input_mapping must include core field: {field}")
            
            if not v[field] or not isinstance(v[field], list):
                raise ValueError(f"input_mapping[{field}] must be a non-empty list")
        
        return v
    
    @field_validator('output_schema')
    @classmethod
    def validate_output_schema(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure output schema has required structure."""
        if 'core_item' not in v:
            raise ValueError("output_schema must include 'core_item' section")
        
        if 'attributes' not in v:
            raise ValueError("output_schema must include 'attributes' section")
        
        core_item = v['core_item']
        required_core_fields = ['id', 'title', 'description', 'language']
        
        for field in required_core_fields:
            if field not in core_item:
                raise ValueError(f"output_schema.core_item must include field: {field}")
        
        return v
    
    @model_validator(mode='after')
    def validate_attribute_consistency(self) -> 'DomainPack':
        """Ensure attributes in output_schema have corresponding hints if provided."""
        if 'attributes' not in self.output_schema:
            return self
        
        attribute_keys = {attr['key'] for attr in self.output_schema['attributes'] 
                         if isinstance(attr, dict) and 'key' in attr}
        
        # Check for hints without corresponding attributes
        extra_hints = set(self.attribute_hints.keys()) - attribute_keys
        if extra_hints:
            raise ValueError(f"attribute_hints contains keys not in output_schema.attributes: {extra_hints}")
        
        return self
    
    def get_attribute_config(self, attribute_key: str) -> Optional[AttributeConfig]:
        """Get configuration for a specific attribute.
        
        Args:
            attribute_key: The attribute key to look up
            
        Returns:
            AttributeConfig object if found, None otherwise
        """
        if 'attributes' not in self.output_schema:
            return None
        
        for attr_dict in self.output_schema['attributes']:
            if isinstance(attr_dict, dict) and attr_dict.get('key') == attribute_key:
                # Merge with hints if available
                config_data = attr_dict.copy()
                if attribute_key in self.attribute_hints:
                    config_data.update(self.attribute_hints[attribute_key])
                
                try:
                    return AttributeConfig(**config_data)
                except Exception:
                    return None
        
        return None
    
    def get_prompt_template(self, prompt_type: str, **kwargs) -> Optional[str]:
        """Get a prompt template with variable substitution.
        
        Args:
            prompt_type: Type of prompt to retrieve
            **kwargs: Variables for template substitution
            
        Returns:
            Formatted prompt string if found, None otherwise
        """
        if prompt_type not in self.prompts:
            return None
        
        template = self.prompts[prompt_type]
        
        # Add domain pack context to kwargs
        kwargs.update({
            'DOMAIN': self.domain,
            'TAXONOMY_VERSION': self.taxonomy_version,
            'OUTPUT_KEYS_JSON': str(list(self.attribute_hints.keys())),
            'ATTRIBUTE_HINTS': str(self.attribute_hints)
        })
        
        try:
            # Simple template substitution
            for key, value in kwargs.items():
                template = template.replace(f'{{{{{key}}}}}', str(value))
            return template
        except Exception:
            return template
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"
