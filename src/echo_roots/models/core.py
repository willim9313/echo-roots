"""Core data models for echo-roots framework.

This module defines the fundamental Pydantic v2 models that serve as the
backbone for data contracts throughout the system. All models follow the
JSON schemas defined in docs/DATA_SCHEMA.md.

Key models:
- IngestionItem: Raw input data from various sources
- ExtractionResult: LLM-processed attributes and terms
- ElevationProposal: D→C layer promotion requests  
- Mapping: Versioned alias/merge/replace operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import UUID4


class ProcessingStatus(str, Enum):
    """Status of item processing through the ingestion pipeline."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class IngestionItem(BaseModel):
    """Raw input item for ingestion into the taxonomy system.
    
    Represents unprocessed data from various sources (APIs, files, databases)
    before domain adaptation and normalization. Follows the core ingestion
    schema from DATA_SCHEMA.md.
    
    Attributes:
        item_id: Unique identifier for the item
        title: Primary title or name of the item
        description: Optional detailed description
        raw_category: Original category from source system
        raw_attributes: Domain-specific attributes as key-value pairs
        source: Data origin identifier (API, DB, CSV, etc.)
        language: Language code (e.g., 'en', 'zh-tw', 'zh-cn')
        metadata: Additional metadata and collection info
        ingested_at: Timestamp when item was ingested
    """
    
    item_id: str = Field(
        description="Unique identifier for the item",
        min_length=1,
        max_length=255
    )
    title: str = Field(
        description="Primary title or name of the item",
        min_length=1,
        max_length=1000
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional detailed description",
        max_length=5000
    )
    raw_category: Optional[str] = Field(
        default=None,
        description="Original category from source system",
        max_length=500
    )
    raw_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific attributes as key-value pairs"
    )
    source: str = Field(
        description="Data origin identifier (API, DB, CSV, etc.)",
        min_length=1,
        max_length=100
    )
    language: str = Field(
        default="auto",
        description="Language code (e.g., 'en', 'zh-tw', 'zh-cn')",
        pattern=r'^[a-z]{2}(-[A-Z]{2})?|auto$'
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata and collection info"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when item was ingested"
    )
    
    @field_validator('item_id')
    @classmethod
    def validate_item_id(cls, v: str) -> str:
        """Ensure item_id is not just whitespace."""
        if not v.strip():
            raise ValueError("item_id cannot be empty or whitespace")
        return v.strip()
    
    @field_validator('title')
    @classmethod 
    def validate_title(cls, v: str) -> str:
        """Ensure title is not just whitespace."""
        if not v.strip():
            raise ValueError("title cannot be empty or whitespace")
        return v.strip()

    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        validate_assignment = True
        extra = "forbid"


class AttributeExtraction(BaseModel):
    """Individual attribute extracted from an item.
    
    Represents a single normalized attribute-value pair with evidence
    from the source text.
    
    Attributes:
        name: Normalized attribute name
        value: Extracted attribute value
        evidence: Source text that supports this extraction
        confidence: Optional confidence score (0.0-1.0)
    """
    
    name: str = Field(
        description="Normalized attribute name",
        min_length=1,
        max_length=100
    )
    value: str = Field(
        description="Extracted attribute value", 
        min_length=1,
        max_length=500
    )
    evidence: str = Field(
        description="Source text that supports this extraction",
        min_length=1,
        max_length=1000
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class SemanticTerm(BaseModel):
    """Semantic term extracted from item content.
    
    Represents candidate terms for the D (semantic) layer that may
    eventually be elevated to controlled vocabulary.
    
    Attributes:
        term: The extracted semantic term
        context: Surrounding context from source
        confidence: Extraction confidence score (0.0-1.0)
        frequency: Optional frequency count in dataset
    """
    
    term: str = Field(
        description="The extracted semantic term",
        min_length=1,
        max_length=200
    )
    context: str = Field(
        description="Surrounding context from source",
        min_length=1,
        max_length=1000
    )
    confidence: float = Field(
        description="Extraction confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    frequency: Optional[int] = Field(
        default=None,
        description="Optional frequency count in dataset",
        ge=1
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class ExtractionMetadata(BaseModel):
    """Metadata for extraction operations.
    
    Tracks the model, run, and timing information for LLM extractions.
    
    Attributes:
        model: LLM model identifier used for extraction
        run_id: Unique identifier for this extraction run
        extracted_at: Timestamp when extraction was performed
        processing_time_ms: Optional processing duration in milliseconds
    """
    
    model: str = Field(
        description="LLM model identifier used for extraction",
        min_length=1,
        max_length=100
    )
    run_id: str = Field(
        description="Unique identifier for this extraction run",
        min_length=1,
        max_length=100
    )
    extracted_at: datetime = Field(
        description="Timestamp when extraction was performed"
    )
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Optional processing duration in milliseconds",
        ge=0
    )
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class ExtractionResult(BaseModel):
    """Result of LLM attribute and term extraction.
    
    Contains the structured output from LLM processing of an ingestion item,
    including normalized attributes, semantic terms, and extraction metadata.
    Follows the LLM extraction schema from DATA_SCHEMA.md.
    
    Attributes:
        item_id: Reference to the source ingestion item
        attributes: List of extracted and normalized attributes
        terms: List of semantic terms found in the content
        metadata: Extraction operation metadata
    """
    
    item_id: str = Field(
        description="Reference to the source ingestion item",
        min_length=1,
        max_length=255
    )
    attributes: List[AttributeExtraction] = Field(
        default_factory=list,
        description="List of extracted and normalized attributes"
    )
    terms: List[SemanticTerm] = Field(
        default_factory=list, 
        description="List of semantic terms found in the content"
    )
    metadata: ExtractionMetadata = Field(
        description="Extraction operation metadata"
    )
    
    @field_validator('attributes')
    @classmethod
    def validate_unique_attributes(cls, v: List[AttributeExtraction]) -> List[AttributeExtraction]:
        """Ensure attribute names are unique within an extraction."""
        names = [attr.name for attr in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate attribute names are not allowed")
        return v
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class ElevationMetrics(BaseModel):
    """Metrics supporting a D→C elevation proposal.
    
    Quantitative data to support promoting a semantic term to 
    controlled vocabulary.
    
    Attributes:
        frequency: Number of times term appears in dataset
        coverage: Percentage of items that contain this term (0.0-1.0)
        stability_score: Consistency of usage across contexts (0.0-1.0)
        co_occurrence_strength: Average association with known terms (0.0-1.0)
    """
    
    frequency: int = Field(
        description="Number of times term appears in dataset",
        ge=1
    )
    coverage: float = Field(
        description="Percentage of items that contain this term (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    stability_score: float = Field(
        description="Consistency of usage across contexts (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    co_occurrence_strength: Optional[float] = Field(
        default=None,
        description="Average association with known terms (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class ElevationProposal(BaseModel):
    """Proposal to elevate a term from D (semantic) to C (controlled) layer.
    
    Represents a request to promote a semantic candidate term to the
    controlled vocabulary, including justification and metrics.
    Follows the elevation proposal schema from DATA_SCHEMA.md.
    
    Attributes:
        proposal_id: Unique identifier for this proposal
        term: The semantic term to be elevated
        proposed_attribute: Target attribute name in controlled vocabulary
        justification: Human-readable explanation for the promotion
        metrics: Supporting quantitative metrics
        submitted_by: Identifier of who submitted the proposal
        submitted_at: Timestamp of proposal submission
        status: Current approval status
        reviewed_at: Optional timestamp of review completion
        reviewer_notes: Optional reviewer comments
    """
    
    proposal_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this proposal"
    )
    term: str = Field(
        description="The semantic term to be elevated",
        min_length=1,
        max_length=200
    )
    proposed_attribute: str = Field(
        description="Target attribute name in controlled vocabulary",
        min_length=1,
        max_length=100
    )
    justification: str = Field(
        description="Human-readable explanation for the promotion",
        min_length=10,
        max_length=2000
    )
    metrics: ElevationMetrics = Field(
        description="Supporting quantitative metrics"
    )
    submitted_by: str = Field(
        description="Identifier of who submitted the proposal",
        min_length=1,
        max_length=100
    )
    submitted_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of proposal submission"
    )
    status: str = Field(
        default="pending",
        description="Current approval status",
        pattern=r'^(pending|approved|rejected|withdrawn)$'
    )
    reviewed_at: Optional[datetime] = Field(
        default=None,
        description="Optional timestamp of review completion"
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Optional reviewer comments",
        max_length=2000
    )
    
    @model_validator(mode='after')
    def validate_review_consistency(self) -> 'ElevationProposal':
        """Ensure review fields are consistent with status."""
        if self.status in ['approved', 'rejected'] and not self.reviewed_at:
            raise ValueError(f"reviewed_at is required when status is {self.status}")
        
        if self.status == 'pending' and self.reviewed_at:
            raise ValueError("reviewed_at should not be set for pending proposals")
            
        return self
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"


class Mapping(BaseModel):
    """Versioned mapping between terms for alias, merge, or replace operations.
    
    Tracks the evolution of taxonomy terms over time with full versioning
    support for rollback and audit trails.
    Follows the mapping schema from DATA_SCHEMA.md.
    
    Attributes:
        mapping_id: Unique identifier for this mapping
        from_term: Source term being mapped
        to_term: Target term for the mapping
        relation_type: Type of mapping relationship
        valid_from: Start timestamp for mapping validity
        valid_to: Optional end timestamp for mapping validity
        created_by: Identifier of who created the mapping
        created_at: Timestamp of mapping creation
        notes: Optional explanatory notes
        metadata: Additional mapping-specific metadata
    """
    
    mapping_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this mapping"
    )
    from_term: str = Field(
        description="Source term being mapped",
        min_length=1,
        max_length=200
    )
    to_term: str = Field(
        description="Target term for the mapping",
        min_length=1,
        max_length=200
    )
    relation_type: str = Field(
        description="Type of mapping relationship",
        pattern=r'^(alias|merge|replace|deprecate)$'
    )
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="Start timestamp for mapping validity"
    )
    valid_to: Optional[datetime] = Field(
        default=None,
        description="Optional end timestamp for mapping validity"
    )
    created_by: str = Field(
        description="Identifier of who created the mapping",
        min_length=1,
        max_length=100
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of mapping creation"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional explanatory notes",
        max_length=1000
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional mapping-specific metadata"
    )
    
    @field_validator('from_term', 'to_term')
    @classmethod
    def validate_terms_not_empty(cls, v: str) -> str:
        """Ensure terms are not just whitespace."""
        if not v.strip():
            raise ValueError("Terms cannot be empty or whitespace")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_term_mapping(self) -> 'Mapping':
        """Ensure from_term and to_term are different for non-deprecate mappings."""
        if self.relation_type != 'deprecate' and self.from_term == self.to_term:
            raise ValueError("from_term and to_term cannot be the same for non-deprecate mappings")
        
        if self.valid_to and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
            
        return self
    
    class Config:
        """Pydantic configuration."""
        str_strip_whitespace = True
        extra = "forbid"
