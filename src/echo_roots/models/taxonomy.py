"""Taxonomy framework models for A/C/D layers.

This module defines the Pydantic v2 models for the taxonomy framework
as described in docs/TAXONOMY.md. It implements the A/C/D layer structure:

- A Layer: Classification skeleton (taxonomy tree)
- C Layer: Controlled attributes and values
- D Layer: Semantic candidate network

Key models:
- Category: Hierarchical taxonomy nodes (A layer)
- Attribute: Controlled vocabulary attributes (C layer)
- SemanticCandidate: Candidate terms for semantic layer (D layer)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class Category(BaseModel):
    """Hierarchical taxonomy category node (A layer).
    
    Represents a node in the classification tree with support for
    multilingual labels, metadata, and hierarchy management.
    
    Attributes:
        category_id: Unique identifier for the category
        name: Primary category name
        parent_id: Optional parent category ID for hierarchy
        level: Depth level in the taxonomy tree (0 = root)
        path: Full path from root (e.g., ["Electronics", "Mobile", "Smartphones"])
        labels: Multilingual labels for the category
        description: Optional detailed description
        status: Category status (active, deprecated, merged)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional category-specific metadata
    """
    
    category_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the category"
    )
    name: str = Field(
        description="Primary category name",
        min_length=1,
        max_length=100
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Optional parent category ID for hierarchy"
    )
    level: int = Field(
        description="Depth level in the taxonomy tree (0 = root)",
        ge=0,
        le=10
    )
    path: List[str] = Field(
        description="Full path from root",
        min_length=1,
        max_length=10
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Multilingual labels for the category"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional detailed description",
        max_length=1000
    )
    status: str = Field(
        default="active",
        description="Category status",
        pattern=r'^(active|deprecated|merged)$'
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional category-specific metadata"
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not just whitespace."""
        if not v.strip():
            raise ValueError("Category name cannot be empty or whitespace")
        return v.strip()
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v: List[str]) -> List[str]:
        """Ensure path components are not empty."""
        if not v:
            raise ValueError("Path cannot be empty")
        
        for component in v:
            if not component.strip():
                raise ValueError("Path components cannot be empty or whitespace")
        
        return [component.strip() for component in v]
    
    @model_validator(mode='after')
    def validate_hierarchy_consistency(self) -> 'Category':
        """Ensure hierarchy consistency between level, path, and parent."""
        if self.level == 0 and self.parent_id is not None:
            raise ValueError("Root categories (level 0) cannot have a parent")
        
        if self.level > 0 and self.parent_id is None:
            raise ValueError("Non-root categories must have a parent_id")
        
        if len(self.path) != self.level + 1:
            raise ValueError(f"Path length ({len(self.path)}) must equal level + 1 ({self.level + 1})")
        
        if self.path[-1] != self.name:
            raise ValueError("Last path component must match category name")
        
        return self
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"


class AttributeValue(BaseModel):
    """Individual value within a controlled attribute.
    
    Represents a specific allowed value for a controlled vocabulary
    attribute, with multilingual support and normalization rules.
    
    Attributes:
        value: The normalized attribute value
        labels: Multilingual labels for the value
        aliases: Alternative forms that map to this value
        description: Optional description of the value
        status: Value status (active, deprecated, merged)
        metadata: Additional value-specific metadata
    """
    
    value: str = Field(
        description="The normalized attribute value",
        min_length=1,
        max_length=200
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Multilingual labels for the value"
    )
    aliases: Set[str] = Field(
        default_factory=set,
        description="Alternative forms that map to this value"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the value",
        max_length=500
    )
    status: str = Field(
        default="active",
        description="Value status",
        pattern=r'^(active|deprecated|merged)$'
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional value-specific metadata"
    )
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v: str) -> str:
        """Ensure value is not just whitespace."""
        if not v.strip():
            raise ValueError("Attribute value cannot be empty or whitespace")
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class Attribute(BaseModel):
    """Controlled vocabulary attribute (C layer).
    
    Represents a managed attribute with controlled values, validation rules,
    and governance workflows. Part of the normalization layer between
    raw input and semantic candidates.
    
    Attributes:
        attribute_id: Unique identifier for the attribute
        name: Attribute name (e.g., "brand", "color", "size")
        display_name: Human-readable display name
        data_type: Data type for validation (categorical, text, numeric, boolean)
        values: Controlled values for categorical attributes
        validation_rules: Optional validation patterns/rules
        labels: Multilingual labels for the attribute
        description: Detailed description of the attribute
        required: Whether this attribute is required for items
        status: Attribute status (active, deprecated, merged)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional attribute-specific metadata
    """
    
    attribute_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the attribute"
    )
    name: str = Field(
        description="Attribute name (e.g., 'brand', 'color', 'size')",
        min_length=1,
        max_length=50,
        pattern=r'^[a-z][a-z0-9_]*$'
    )
    display_name: str = Field(
        description="Human-readable display name",
        min_length=1,
        max_length=100
    )
    data_type: str = Field(
        description="Data type for validation",
        pattern=r'^(categorical|text|numeric|boolean|date)$'
    )
    values: List[AttributeValue] = Field(
        default_factory=list,
        description="Controlled values for categorical attributes"
    )
    validation_rules: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional validation patterns/rules"
    )
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Multilingual labels for the attribute"
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed description of the attribute",
        max_length=1000
    )
    required: bool = Field(
        default=False,
        description="Whether this attribute is required for items"
    )
    status: str = Field(
        default="active",
        description="Attribute status",
        pattern=r'^(active|deprecated|merged)$'
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional attribute-specific metadata"
    )
    
    @field_validator('display_name')
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        """Ensure display_name is not just whitespace."""
        if not v.strip():
            raise ValueError("Display name cannot be empty or whitespace")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_categorical_values(self) -> 'Attribute':
        """Ensure categorical attributes have values."""
        if self.data_type == "categorical" and not self.values:
            raise ValueError("Categorical attributes must have at least one value")
        
        if self.data_type != "categorical" and self.values:
            raise ValueError("Only categorical attributes can have predefined values")
        
        # Ensure value uniqueness
        if self.values:
            value_names = [v.value for v in self.values]
            if len(value_names) != len(set(value_names)):
                raise ValueError("Attribute values must be unique")
        
        return self
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"


class SemanticRelation(BaseModel):
    """Relationship between semantic candidates.
    
    Represents connections in the semantic candidate network (D layer)
    such as similarity, co-occurrence, or hierarchical relationships.
    
    Attributes:
        relation_id: Unique identifier for the relationship
        from_term: Source term in the relationship
        to_term: Target term in the relationship
        relation_type: Type of semantic relationship
        strength: Relationship strength score (0.0-1.0)
        evidence_count: Number of supporting evidence instances
        context: Optional contextual information
        metadata: Additional relation-specific metadata
    """
    
    relation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the relationship"
    )
    from_term: str = Field(
        description="Source term in the relationship",
        min_length=1,
        max_length=200
    )
    to_term: str = Field(
        description="Target term in the relationship",
        min_length=1,
        max_length=200
    )
    relation_type: str = Field(
        description="Type of semantic relationship",
        pattern=r'^(similar|related|variant|broader|narrower|co_occurs)$'
    )
    strength: float = Field(
        description="Relationship strength score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    evidence_count: int = Field(
        description="Number of supporting evidence instances",
        ge=1
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional contextual information",
        max_length=500
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional relation-specific metadata"
    )
    
    @model_validator(mode='after')
    def validate_relation_terms(self) -> 'SemanticRelation':
        """Ensure from_term and to_term are different."""
        if self.from_term.strip().lower() == self.to_term.strip().lower():
            raise ValueError("from_term and to_term cannot be the same")
        return self
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class SemanticCandidate(BaseModel):
    """Semantic candidate term in the D layer.
    
    Represents a term in the semantic candidate network that may
    eventually be elevated to controlled vocabulary. Includes
    clustering, scoring, and relationship information.
    
    Attributes:
        candidate_id: Unique identifier for the candidate
        term: The candidate term text
        normalized_term: Normalized form for matching
        frequency: Occurrence frequency in the dataset
        contexts: Sample contexts where the term appears
        cluster_id: Optional cluster assignment for grouping
        score: Overall quality/stability score (0.0-1.0)
        relations: Semantic relationships to other candidates
        language: Primary language of the term
        status: Candidate status in the workflow
        created_at: First seen timestamp
        updated_at: Last update timestamp
        metadata: Additional candidate-specific metadata
    """
    
    candidate_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the candidate"
    )
    term: str = Field(
        description="The candidate term text",
        min_length=1,
        max_length=200
    )
    normalized_term: str = Field(
        description="Normalized form for matching",
        min_length=1,
        max_length=200
    )
    frequency: int = Field(
        description="Occurrence frequency in the dataset",
        ge=1
    )
    contexts: List[str] = Field(
        default_factory=list,
        description="Sample contexts where the term appears",
        max_length=10
    )
    cluster_id: Optional[str] = Field(
        default=None,
        description="Optional cluster assignment for grouping"
    )
    score: float = Field(
        description="Overall quality/stability score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    relations: List[SemanticRelation] = Field(
        default_factory=list,
        description="Semantic relationships to other candidates"
    )
    language: str = Field(
        default="auto",
        description="Primary language of the term",
        pattern=r'^[a-z]{2}(-[A-Z]{2})?|auto$'
    )
    status: str = Field(
        default="active",
        description="Candidate status in the workflow",
        pattern=r'^(active|clustered|merged|elevated|deprecated)$'
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="First seen timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional candidate-specific metadata"
    )
    
    @field_validator('term', 'normalized_term')
    @classmethod
    def validate_terms(cls, v: str) -> str:
        """Ensure terms are not just whitespace."""
        if not v.strip():
            raise ValueError("Terms cannot be empty or whitespace")
        return v.strip()
    
    @field_validator('contexts')
    @classmethod
    def validate_contexts(cls, v: List[str]) -> List[str]:
        """Ensure contexts are not empty."""
        return [ctx.strip() for ctx in v if ctx.strip()]
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"
