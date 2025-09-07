# Echo-Roots API Reference

**Version:** 1.0.0  
**Updated:** 2025-09-07  
**Authors:** Echo-Roots Team  
**Tags:** api, reference, python  

---

## Overview

This document provides a comprehensive reference for the Echo-Roots taxonomy framework API.

## Module: __init__

Echo-Roots: Practical taxonomy construction and semantic enrichment framework.

This package provides tools for building, managing, and evolving taxonomies
with controlled vocabularies and semantic layers across domains like e-commerce,
media, and knowledge graphs.

Key components:
- Domain adapters for flexible input mapping
- A/C/D taxonomy framework (Classification/Controlled/Dynamic)
- Multi-storage backend support (DuckDB core + optional Neo4j/Qdrant)
- LLM-powered attribute extraction and normalization
- Governance workflows for taxonomy evolution

**Module Path:** `src/echo_roots/__init__.py`



## Module: __init__

Comprehensive documentation system with automated generation, knowledge management,
interactive help, and integrated learning resources for the Echo-Roots taxonomy framework.

**Module Path:** `src/echo_roots/documentation/__init__.py`



### Classes

#### `DocumentationType`

Types of documentation that can be generated.

**Inherits from:** Enum

#### `ContentFormat`

Supported content formats.

**Inherits from:** Enum

#### `DocumentSection`

Represents a section within a document.

**Methods:**

- `to_markdown(self)`: Convert section to markdown format
- `to_dict(self)`: Convert section to dictionary

#### `Document`

Represents a complete document.

**Methods:**

- `add_section(self, section)`: Add a section to the document
- `to_markdown(self)`: Convert entire document to markdown
- `to_html(self)`: Convert document to HTML

#### `CodeAnalyzer`

Analyzes Python code to extract documentation information.

**Methods:**

- `__init__(self)`
- `analyze_module(self, module_path)`: Analyze a Python module and extract documentation info
- `_analyze_class(self, node)`: Analyze a class definition
- `_analyze_function(self, node)`: Analyze a function definition

#### `DocumentGenerator`

Generates documentation from various sources.

**Methods:**

- `__init__(self)`
- `generate_api_reference(self, module_paths)`: Generate API reference documentation from Python modules
- `_create_module_section(self, module_info)`: Create a documentation section for a module
- `_format_class_docs(self, class_info)`: Format class documentation
- `_format_function_docs(self, func_info)`: Format function documentation
- `generate_user_guide(self)`: Generate comprehensive user guide
- `generate_developer_guide(self)`: Generate developer guide with architecture and extension information
- `generate_changelog(self, version_history)`: Generate changelog from version history

#### `KnowledgeBase`

Manages knowledge base with searchable documentation and learning resources.

**Methods:**

- `__init__(self, base_path)`
- `add_document(self, doc_id, document)`: Add a document to the knowledge base
- `_update_index(self, doc_id, document)`: Update search index with document content
- `_save_document(self, doc_id, document)`: Save document to disk in multiple formats
- `search(self, query, limit)`: Search the knowledge base
- `get_document(self, doc_id)`: Get a document by ID
- `list_documents(self)`: List all documents
- `generate_index_page(self)`: Generate HTML index page for the knowledge base

#### `DocumentationManager`

Main documentation management system.

**Methods:**

- `__init__(self, docs_path)`
- `initialize(self)`: Initialize the documentation system
- `generate_all_docs(self)`: Generate comprehensive documentation set
- `search_docs(self, query, limit)`: Search documentation
- `get_doc_stats(self)`: Get documentation statistics

#### `InteractiveHelp`

Interactive help system for CLI and API.

**Methods:**

- `__init__(self)`
- `show_command_help(self, command)`: Show detailed help for a specific command
- `show_topic_help(self, topic)`: Show help for a specific topic



## Module: ingestion

T5: Ingestion Pipeline

Main orchestration pipeline that connects domain adaptation (T2), 
LLM extraction (T3), and storage (T4) for end-to-end data processing.

This module provides:
- IngestionConfig: Configuration for ingestion workflows
- IngestionPipeline: Main orchestrator for end-to-end processing
- BatchProcessor: Handles batch ingestion with progress tracking
- StreamProcessor: Handles continuous/streaming ingestion
- PipelineCoordinator: High-level workflow management

**Module Path:** `src/echo_roots/pipelines/ingestion.py`



### Classes

#### `IngestionConfig`

Configuration for ingestion pipeline operations.

**Methods:**

- `__post_init__(self)`: Validate configuration after initialization

#### `IngestionStats`

Statistics tracking for ingestion operations.

**Methods:**

- `success_rate(self)`: Calculate success rate as percentage
- `average_quality(self)`: Calculate average quality score
- `processing_time(self)`: Calculate total processing time in seconds
- `add_error(self, item_id, error, context)`: Add an error to the tracking
- `to_dict(self)`: Convert stats to dictionary for serialization

#### `IngestionPipeline`

Main ingestion pipeline orchestrator.

Coordinates the complete data flow:
Raw Data → Domain Adaptation → LLM Extraction → Storage

**Methods:**

- `__init__(self, config, llm_client, storage_backend)`

#### `BatchProcessor`

Specialized processor for large batch operations.

**Methods:**

- `__init__(self, pipeline)`

#### `StreamProcessor`

Processor for streaming/continuous ingestion.

**Methods:**

- `__init__(self, pipeline)`
- `stop_stream(self)`: Stop stream processing

#### `PipelineCoordinator`

High-level coordinator for managing multiple ingestion pipelines.

Useful for scenarios with multiple domains or processing configurations.

**Methods:**

- `__init__(self)`
- `register_pipeline(self, name, config)`: Register a named pipeline configuration



## Module: __init__

Data processing and transformation pipelines.

**Module Path:** `src/echo_roots/pipelines/__init__.py`



## Module: extraction

LLM-based extraction pipeline for attribute and term extraction.

This module provides the core extraction pipeline that processes IngestionItem objects
through LLM models to extract structured attributes and semantic terms according to
domain-specific configurations.

Key components:
- ExtractorConfig: Configuration for LLM extraction settings
- PromptBuilder: Builds domain-specific prompts for LLM calls
- LLMExtractor: Main extraction engine with LLM integration
- ExtractionPipeline: High-level pipeline orchestrator
- BatchProcessor: Efficient batch processing of multiple items

**Module Path:** `src/echo_roots/pipelines/extraction.py`



### Classes

#### `ExtractorConfig`

Configuration for LLM-based extraction.

Controls model selection, prompt behavior, and processing parameters
for the extraction pipeline.

**Inherits from:** BaseModel

#### `ExtractionError`

Base exception for extraction pipeline errors.

**Inherits from:** Exception

**Methods:**

- `__init__(self, message, item_id, cause)`

#### `PromptBuilder`

Builds domain-specific prompts for LLM extraction.

Constructs prompts by combining base templates with domain-specific
configurations, attribute hints, and output schema specifications.

**Methods:**

- `__init__(self, domain_pack)`: Initialize prompt builder with domain pack
- `_load_base_template(self)`: Load the base extraction prompt template
- `build_prompt(self, item)`: Build extraction prompt for a specific item
- `build_batch_prompt(self, items)`: Build prompt for batch extraction of multiple items

#### `LLMClient`

Protocol for LLM client implementations.

Defines the interface that LLM clients must implement to work
with the extraction pipeline.

**Inherits from:** Protocol

#### `MockLLMClient`

Mock LLM client for testing and development.

Provides deterministic responses for testing the extraction pipeline
without requiring actual LLM API calls.

**Methods:**

- `__init__(self, delay_seconds)`: Initialize mock client

#### `LLMExtractor`

Main LLM extraction engine.

Orchestrates the extraction process including prompt building,
LLM calls, response parsing, and validation.

**Methods:**

- `__init__(self, domain_pack, llm_client, config)`: Initialize extractor
- `_parse_llm_response(self, response_text)`: Parse LLM response into structured data
- `_validate_extraction(self, result)`: Validate extraction result against domain schema

#### `ExtractionPipeline`

High-level extraction pipeline orchestrator.

Provides a simple interface for domain-based extraction with
automatic domain loading and configuration.

**Methods:**

- `__init__(self, domains_path, llm_client, config)`: Initialize extraction pipeline
- `get_extractor(self, domain)`: Get or create extractor for a domain
- `list_available_domains(self)`: List available domains for extraction



## Module: openai_client

OpenAI client implementation for LLM extraction.

Provides integration with OpenAI's API for real LLM-based extraction.
Supports both synchronous and asynchronous calls with proper error handling.

**Module Path:** `src/echo_roots/pipelines/openai_client.py`



### Classes

#### `OpenAIClient`

OpenAI client implementation for LLM extraction.

Provides integration with OpenAI's GPT models for attribute and term extraction.
Handles API authentication, rate limiting, and error recovery.

**Methods:**

- `__init__(self, api_key, organization, base_url)`: Initialize OpenAI client
- `__str__(self)`

#### `AzureOpenAIClient`

Azure OpenAI client implementation.

Specialized client for Azure's OpenAI service with endpoint and deployment handling.

**Inherits from:** OpenAIClient

**Methods:**

- `__init__(self, azure_endpoint, deployment_name, api_version, api_key)`: Initialize Azure OpenAI client
- `__str__(self)`

#### `MockLLMClient`

Mock LLM client for testing purposes.

Provides realistic but deterministic responses for testing
the extraction pipeline without making actual API calls.

**Methods:**

- `__init__(self)`: Initialize mock client
- `_extract_mock_title(self, prompt)`: Extract a mock title from the prompt text



### Functions

#### `create_openai_client(api_key)`

Create a standard OpenAI client.

Args:
    api_key: OpenAI API key (uses environment variable if None)
    
Returns:
    Configured OpenAIClient

**Returns:** `OpenAIClient`

#### `create_azure_client(azure_endpoint, deployment_name, api_version, api_key)`

Create an Azure OpenAI client.

Args:
    azure_endpoint: Azure OpenAI endpoint URL
    deployment_name: Azure deployment name  
    api_version: Azure API version
    api_key: Azure API key (uses environment variable if None)
    
Returns:
    Configured AzureOpenAIClient

**Returns:** `AzureOpenAIClient`



## Module: validation

Validation and post-processing for extraction results.

This module provides validation, normalization, and quality assurance
for extraction results from LLM processing.

Key components:
- ExtractionValidator: Validates extraction results against domain schemas
- ResultNormalizer: Normalizes and cleans extraction data
- QualityAnalyzer: Analyzes extraction quality and confidence
- PostProcessor: Orchestrates validation and normalization pipeline

**Module Path:** `src/echo_roots/pipelines/validation.py`



### Classes

#### `ValidationIssue`

Represents a validation issue found during extraction validation.

#### `QualityMetrics`

Quality metrics for an extraction result.

#### `ExtractionValidator`

Validates extraction results against domain schemas and quality standards.

Performs comprehensive validation including schema compliance,
data quality checks, and domain-specific validation rules.

**Methods:**

- `__init__(self, domain_pack)`: Initialize validator with domain configuration
- `_get_expected_attributes(self)`: Extract expected attributes from domain pack schema
- `_get_validation_rules(self)`: Extract validation rules from domain pack
- `validate(self, result)`: Validate an extraction result comprehensively
- `_validate_schema_compliance(self, result)`: Validate that result complies with expected schema
- `_validate_attributes(self, attributes)`: Validate individual attributes
- `_validate_terms(self, terms)`: Validate semantic terms
- `_validate_quality(self, result)`: Validate overall extraction quality
- `_validate_domain_rules(self, result)`: Validate against domain-specific rules

#### `ResultNormalizer`

Normalizes and cleans extraction results.

Applies normalization rules to ensure consistent formatting
and clean up common extraction issues.

**Methods:**

- `__init__(self, domain_pack)`: Initialize normalizer with domain configuration
- `normalize(self, result)`: Normalize an extraction result
- `_normalize_attribute(self, attr)`: Normalize a single attribute
- `_normalize_term(self, term)`: Normalize a semantic term
- `_normalize_attribute_name(self, name)`: Normalize attribute name to standard format
- `_normalize_attribute_value(self, attr_name, value)`: Normalize attribute value based on type and hints
- `_clean_text(self, text)`: Clean and normalize text

#### `QualityAnalyzer`

Analyzes extraction result quality and provides metrics.

**Methods:**

- `__init__(self, domain_pack)`: Initialize quality analyzer
- `analyze(self, result, validation_issues)`: Analyze extraction result quality
- `_calculate_schema_compliance_score(self, result, issues)`: Calculate how well the result complies with the schema
- `_calculate_coverage_score(self, result)`: Calculate attribute coverage score
- `_calculate_evidence_quality_score(self, result)`: Calculate evidence quality score based on evidence length and content
- `_calculate_total_quality_score(self, schema_score, coverage_score, evidence_score, issues)`: Calculate overall quality score

#### `PostProcessor`

Orchestrates validation and normalization of extraction results.

Provides a unified interface for post-processing extraction results
with validation, normalization, and quality analysis.

**Methods:**

- `__init__(self, domain_pack)`: Initialize post-processor
- `process(self, result, normalize, validate)`: Process an extraction result with validation and normalization
- `process_batch(self, results, normalize, validate)`: Process multiple extraction results



## Module: __init__

Utility functions and helpers.

**Module Path:** `src/echo_roots/utils/__init__.py`



## Module: domain

Domain pack models for YAML configuration.

This module defines the Pydantic v2 models for parsing and validating
domain.yaml files. Domain packs provide flexible configuration for
adapting the core framework to specific domains like e-commerce,
media, or knowledge graphs.

Key models:
- DomainPack: Complete domain configuration
- AttributeConfig: Domain-specific attribute definitions
- ValidationRule: Custom validation rules for attributes

**Module Path:** `src/echo_roots/models/domain.py`



### Classes

#### `AttributeConfig`

Configuration for a domain-specific attribute.

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

**Inherits from:** BaseModel

#### `ValidationRule`

Custom validation rule for domain-specific processing.

Defines rules for validating and normalizing data within
a domain context.

Attributes:
    field: Target field for validation
    rule_type: Type of validation rule
    parameters: Rule-specific parameters
    error_message: Custom error message for validation failures

**Inherits from:** BaseModel

#### `MetricConfig`

Configuration for domain-specific evaluation metrics.

Defines metrics to track for quality assessment and
performance monitoring within a domain.

Attributes:
    name: Metric identifier
    params: Metric-specific parameters
    threshold: Optional threshold for alerts/gating

**Inherits from:** BaseModel

#### `RuntimeConfig`

Runtime configuration for domain processing.

Defines behavior settings for processing items within
a specific domain context.

Attributes:
    language_default: Default language when not specified
    dedupe_by: Fields to use for deduplication
    skip_if_missing: Required fields - skip item if missing
    batch_size: Processing batch size
    timeout_seconds: Processing timeout in seconds

**Inherits from:** BaseModel

#### `DomainPack`

Complete domain pack configuration.

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

**Inherits from:** BaseModel

**Methods:**

- `validate_input_mapping(cls, v)`: Ensure input mapping has required core fields
- `validate_output_schema(cls, v)`: Ensure output schema has required structure
- `validate_attribute_consistency(self)`: Ensure attributes in output_schema have corresponding hints if provided
- `get_attribute_config(self, attribute_key)`: Get configuration for a specific attribute
- `get_prompt_template(self, prompt_type)`: Get a prompt template with variable substitution



## Module: __init__

Core data models for echo-roots taxonomy framework.

**Module Path:** `src/echo_roots/models/__init__.py`



## Module: core

Core data models for echo-roots framework.

This module defines the fundamental Pydantic v2 models that serve as the
backbone for data contracts throughout the system. All models follow the
JSON schemas defined in docs/DATA_SCHEMA.md.

Key models:
- IngestionItem: Raw input data from various sources
- ExtractionResult: LLM-processed attributes and terms
- ElevationProposal: D→C layer promotion requests  
- Mapping: Versioned alias/merge/replace operations

**Module Path:** `src/echo_roots/models/core.py`



### Classes

#### `ProcessingStatus`

Status of item processing through the ingestion pipeline.

**Inherits from:** str, Enum

#### `IngestionItem`

Raw input item for ingestion into the taxonomy system.

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

**Inherits from:** BaseModel

**Methods:**

- `validate_item_id(cls, v)`: Ensure item_id is not just whitespace
- `validate_title(cls, v)`: Ensure title is not just whitespace

#### `AttributeExtraction`

Individual attribute extracted from an item.

Represents a single normalized attribute-value pair with evidence
from the source text.

Attributes:
    name: Normalized attribute name
    value: Extracted attribute value
    evidence: Source text that supports this extraction
    confidence: Optional confidence score (0.0-1.0)

**Inherits from:** BaseModel

#### `SemanticTerm`

Semantic term extracted from item content.

Represents candidate terms for the D (semantic) layer that may
eventually be elevated to controlled vocabulary.

Attributes:
    term: The extracted semantic term
    context: Surrounding context from source
    confidence: Extraction confidence score (0.0-1.0)
    frequency: Optional frequency count in dataset

**Inherits from:** BaseModel

#### `ExtractionMetadata`

Metadata for extraction operations.

Tracks the model, run, and timing information for LLM extractions.

Attributes:
    model: LLM model identifier used for extraction
    run_id: Unique identifier for this extraction run
    extracted_at: Timestamp when extraction was performed
    processing_time_ms: Optional processing duration in milliseconds

**Inherits from:** BaseModel

#### `ExtractionResult`

Result of LLM attribute and term extraction.

Contains the structured output from LLM processing of an ingestion item,
including normalized attributes, semantic terms, and extraction metadata.
Follows the LLM extraction schema from DATA_SCHEMA.md.

Attributes:
    item_id: Reference to the source ingestion item
    attributes: List of extracted and normalized attributes
    terms: List of semantic terms found in the content
    metadata: Extraction operation metadata

**Inherits from:** BaseModel

**Methods:**

- `validate_unique_attributes(cls, v)`: Ensure attribute names are unique within an extraction

#### `ElevationMetrics`

Metrics supporting a D→C elevation proposal.

Quantitative data to support promoting a semantic term to 
controlled vocabulary.

Attributes:
    frequency: Number of times term appears in dataset
    coverage: Percentage of items that contain this term (0.0-1.0)
    stability_score: Consistency of usage across contexts (0.0-1.0)
    co_occurrence_strength: Average association with known terms (0.0-1.0)

**Inherits from:** BaseModel

#### `ElevationProposal`

Proposal to elevate a term from D (semantic) to C (controlled) layer.

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

**Inherits from:** BaseModel

**Methods:**

- `validate_review_consistency(self)`: Ensure review fields are consistent with status

#### `Mapping`

Versioned mapping between terms for alias, merge, or replace operations.

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

**Inherits from:** BaseModel

**Methods:**

- `validate_terms_not_empty(cls, v)`: Ensure terms are not just whitespace
- `validate_term_mapping(self)`: Ensure from_term and to_term are different for non-deprecate mappings



## Module: taxonomy

Taxonomy framework models for A/C/D layers.

This module defines the Pydantic v2 models for the taxonomy framework
as described in docs/TAXONOMY.md. It implements the A/C/D layer structure:

- A Layer: Classification skeleton (taxonomy tree)
- C Layer: Controlled attributes and values
- D Layer: Semantic candidate network

Key models:
- Category: Hierarchical taxonomy nodes (A layer)
- Attribute: Controlled vocabulary attributes (C layer)
- SemanticCandidate: Candidate terms for semantic layer (D layer)

**Module Path:** `src/echo_roots/models/taxonomy.py`



### Classes

#### `Category`

Hierarchical taxonomy category node (A layer).

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

**Inherits from:** BaseModel

**Methods:**

- `validate_name(cls, v)`: Ensure name is not just whitespace
- `validate_path(cls, v)`: Ensure path components are not empty
- `validate_hierarchy_consistency(self)`: Ensure hierarchy consistency between level, path, and parent

#### `AttributeValue`

Individual value within a controlled attribute.

Represents a specific allowed value for a controlled vocabulary
attribute, with multilingual support and normalization rules.

Attributes:
    value: The normalized attribute value
    labels: Multilingual labels for the value
    aliases: Alternative forms that map to this value
    description: Optional description of the value
    status: Value status (active, deprecated, merged)
    metadata: Additional value-specific metadata

**Inherits from:** BaseModel

**Methods:**

- `validate_value(cls, v)`: Ensure value is not just whitespace

#### `Attribute`

Controlled vocabulary attribute (C layer).

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

**Inherits from:** BaseModel

**Methods:**

- `validate_display_name(cls, v)`: Ensure display_name is not just whitespace
- `validate_categorical_values(self)`: Ensure categorical attributes have values

#### `SemanticRelation`

Relationship between semantic candidates.

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

**Inherits from:** BaseModel

**Methods:**

- `validate_relation_terms(self)`: Ensure from_term and to_term are different

#### `SemanticCandidate`

Semantic candidate term in the D layer.

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

**Inherits from:** BaseModel

**Methods:**

- `validate_terms(cls, v)`: Ensure terms are not just whitespace
- `validate_contexts(cls, v)`: Ensure contexts are not empty



## Module: __init__

Command-line interface for echo-roots.

**Module Path:** `src/echo_roots/cli/__init__.py`



## Module: api_server

FastAPI-based REST API server for Echo-Roots taxonomy system.
Provides HTTP endpoints for query, search, and system operations.

**Module Path:** `src/echo_roots/cli/api_server.py`



### Classes

#### `DateTimeJSONEncoder`

JSON encoder that can handle datetime objects.

**Methods:**

- `default(self, obj)`

#### `APIQueryFilter`

API model for query filters.

**Inherits from:** BaseModel

#### `APISortCriterion`

API model for sort criteria.

**Inherits from:** BaseModel

#### `APIQueryRequest`

API model for query requests.

**Inherits from:** BaseModel

#### `APIQueryResult`

API model for query results.

**Inherits from:** BaseModel

#### `APIQueryResponse`

API model for query responses.

**Inherits from:** BaseModel

#### `APIErrorResponse`

API error response model.

**Inherits from:** BaseModel

#### `SystemStatus`

System status model.

**Inherits from:** BaseModel



### Functions

#### `convert_api_filter_to_query_filter(api_filter)`

Convert API filter to internal query filter.

**Returns:** `QueryFilter`

#### `convert_api_sort_to_sort_criterion(api_sort)`

Convert API sort to internal sort criterion.

**Returns:** `SortCriterion`

#### `convert_query_result_to_api(result)`

Convert internal query result to API model.

**Returns:** `APIQueryResult`

#### `start_api_server(host, port, reload, workers)`

Start the API server (called from CLI).



## Module: main

Main CLI entry point for echo-roots.

**Module Path:** `src/echo_roots/cli/main.py`



### Classes

#### `QueryTypeChoice`

CLI-friendly query type choices.

**Inherits from:** str, Enum

#### `OutputFormat`

Output format choices.

**Inherits from:** str, Enum



### Functions

#### `version()`

Show the version information.

#### `status()`

Show system status and configuration.

#### `init(output_dir, with_examples)`

Initialize a new echo-roots workspace.

#### `search_command(query_text, query_type, limit, threshold, entity_types, output_format, workspace)`

Search for entities using various query strategies.

#### `interactive_query(workspace)`

Start interactive query session.

#### `query_history(limit, success_only, workspace)`

Show recent query history.

#### `start_api_server(host, port, reload, workers)`

Start the API server.

#### `test_api_endpoints(base_url)`

Test API endpoints.

#### `open_api_docs(base_url)`

Open API documentation in browser.

#### `governance_status()`

Show system governance and monitoring status.

#### `show_metrics()`

Show detailed system metrics.

#### `show_alerts(severity, resolved)`

Show system alerts.

#### `show_users()`

Show user accounts and access control.

#### `show_audit_logs(user, action, limit)`

Show audit logs.

#### `generate_docs(output_dir, format, force)`

Generate comprehensive documentation.

#### `search_docs(query, limit)`

Search documentation.

#### `list_docs()`

List all available documentation.

#### `show_doc(doc_id, format)`

Show a specific document.

#### `show_help(topic)`

Show interactive help and guidance.

#### `open_docs(doc_id, browser)`

Open documentation in browser or file manager.

#### `show_doc_stats()`

Show documentation statistics and health.



## Module: interfaces

Storage interface protocols for the echo-roots taxonomy system.

This module defines the abstract storage interfaces that enable
pluggable storage backends while maintaining type safety and
consistency across the application.

Based on ADR-0001: Hybrid Storage Model
- DuckDB: Core ingestion, normalization, analytics
- Neo4j: Graph operations, hierarchy navigation  
- Qdrant: Vector search, semantic similarity

The interfaces provide a unified API while allowing specialized
backends to optimize for their strengths.

**Module Path:** `src/echo_roots/storage/interfaces.py`



### Classes

#### `StorageBackend`

Protocol defining the core storage interface.

All storage backends must implement this protocol to ensure
consistent behavior across different storage technologies.

**Inherits from:** Protocol

#### `IngestionRepository`

Repository for managing raw ingestion data.

**Inherits from:** Protocol

#### `ExtractionRepository`

Repository for managing LLM extraction results.

**Inherits from:** Protocol

#### `TaxonomyRepository`

Repository for managing the canonical taxonomy.

**Inherits from:** Protocol

#### `MappingRepository`

Repository for managing domain mappings and transformations.

**Inherits from:** Protocol

#### `ElevationRepository`

Repository for managing elevation proposals and feedback.

**Inherits from:** Protocol

#### `AnalyticsRepository`

Repository for analytics and reporting queries.

**Inherits from:** Protocol

#### `StorageManager`

Abstract base class for storage management.

Coordinates multiple storage backends and provides
high-level repository interfaces.

**Inherits from:** ABC

**Methods:**

- `__init__(self, config)`
- `ingestion(self)`: Access to ingestion data repository
- `extraction(self)`: Access to extraction results repository
- `taxonomy(self)`: Access to taxonomy repository
- `mappings(self)`: Access to mappings repository
- `elevation(self)`: Access to elevation repository
- `analytics(self)`: Access to analytics repository

#### `TransactionContext`

Protocol for transaction management across repositories.

**Inherits from:** Protocol

#### `StorageError`

Base exception for storage operations.

**Inherits from:** Exception

#### `ConnectionError`

Raised when storage backend connection fails.

**Inherits from:** StorageError

#### `IntegrityError`

Raised when data integrity constraints are violated.

**Inherits from:** StorageError

#### `NotFoundError`

Raised when requested entity is not found.

**Inherits from:** StorageError

#### `ConflictError`

Raised when operation conflicts with existing data.

**Inherits from:** StorageError



## Module: __init__

Storage layer for echo-roots taxonomy system.

This package provides a flexible storage abstraction with pluggable
backends for different storage technologies. The primary backend is
DuckDB for analytics and ingestion workloads.

Usage:
    from echo_roots.storage import create_storage, query_ingestion
    
    # Create storage manager
    storage = await create_storage()
    
    # Store and query data
    item_id = await storage.ingestion.store_item(item)
    items = await query_ingestion().filter_by_domain("ecommerce").execute(storage)

**Module Path:** `src/echo_roots/storage/__init__.py`



## Module: duckdb_backend

DuckDB storage backend implementation.

This module provides the core DuckDB-based storage implementation
for the echo-roots taxonomy system. DuckDB serves as the primary
backend for ingestion, normalization, and analytics workloads.

Features:
- High-performance analytical queries
- JSON support for flexible schemas
- In-memory and persistent storage modes
- SQL-based operations with Python integration
- Schema versioning and migrations

**Module Path:** `src/echo_roots/storage/duckdb_backend.py`



### Classes

#### `DuckDBTransaction`

Transaction context for DuckDB operations.

**Inherits from:** TransactionContext

**Methods:**

- `__init__(self, connection)`

#### `DuckDBBackend`

Core DuckDB storage backend.

**Methods:**

- `__init__(self, database_path)`
- `transaction(self)`: Create a new transaction context

#### `DuckDBIngestionRepository`

DuckDB implementation of IngestionRepository.

**Methods:**

- `__init__(self, backend)`

#### `DuckDBStorageManager`

Main storage manager using DuckDB backend.

**Inherits from:** StorageManager

**Methods:**

- `__init__(self, config)`
- `ingestion(self)`: Access to ingestion repository
- `extraction(self)`: Access to extraction repository
- `taxonomy(self)`: Access to taxonomy repository
- `mappings(self)`: Access to mappings repository
- `elevation(self)`: Access to elevation repository
- `analytics(self)`: Access to analytics repository



## Module: migrations

Database migration system for echo-roots storage.

This module provides schema versioning and migration capabilities
to support evolution of the storage layer over time while maintaining
data integrity and backward compatibility.

Features:
- Version-controlled schema changes
- Forward and backward migrations
- Data preservation during schema updates
- Migration rollback capabilities
- Environment-specific configurations

**Module Path:** `src/echo_roots/storage/migrations.py`



### Classes

#### `Migration`

Represents a single database migration.

**Methods:**

- `__init__(self, version, description, up_sql, down_sql, pre_migration, post_migration)`
- `__str__(self)`

#### `MigrationManager`

Manages database schema migrations.

**Methods:**

- `__init__(self, backend)`
- `_register_migrations(self)`: Register all available migrations
- `_get_initial_schema(self)`: Get the initial database schema
- `add_migration(self, migration)`: Add a new migration to the manager



## Module: repository

High-level repository patterns for echo-roots storage.

This module provides convenience classes and utilities that build
on top of the storage interfaces to provide common patterns and
workflows for working with the taxonomy data.

Features:
- Repository composition and coordination
- Transaction management across repositories
- Bulk operations and batch processing
- Query builders and advanced filtering
- Data validation and integrity checks

**Module Path:** `src/echo_roots/storage/repository.py`



### Classes

#### `RepositoryCoordinator`

Coordinates operations across multiple repositories.

Provides high-level workflows that involve multiple storage
components and ensures data consistency across repositories.

**Methods:**

- `__init__(self, storage_manager)`

#### `QueryBuilder`

Builder pattern for constructing complex queries.

Provides a fluent interface for building filtered queries
across different repositories.

**Methods:**

- `__init__(self, repository_type)`
- `filter_by_source(self, source)`: Filter results by source
- `filter_by_status(self, status)`: Filter results by status
- `filter_by_date_range(self, start_date, end_date)`: Filter results by date range
- `filter_by_confidence(self, min_confidence, max_confidence)`: Filter results by confidence range
- `sort_by(self, field, ascending)`: Add sorting criteria
- `limit(self, limit)`: Set result limit
- `offset(self, offset)`: Set result offset
- `page(self, page, page_size)`: Set pagination by page number

#### `DataValidator`

Provides data validation and integrity checking.

Ensures data consistency and validates relationships
between different entities in the storage system.

**Methods:**

- `__init__(self, storage_manager)`

#### `StorageFactory`

Factory for creating and configuring storage managers.

Provides convenient methods for setting up storage with
different configurations and backends.



### Functions

#### `query_ingestion()`

Create a query builder for ingestion items.

**Returns:** `QueryBuilder`

#### `query_extraction()`

Create a query builder for extraction results.

**Returns:** `QueryBuilder`



## Module: __init__

**Module Path:** `src/echo_roots/retrieval/__init__.py`



### Classes

#### `QueryType`

Types of queries supported by the retrieval system.

**Inherits from:** str, Enum

#### `SortOrder`

Sort order options.

**Inherits from:** str, Enum

#### `FilterOperator`

Filter operation types.

**Inherits from:** str, Enum

#### `AggregationType`

Types of aggregations.

**Inherits from:** str, Enum

#### `QueryFilter`

Individual filter criterion.

#### `SortCriterion`

Sort criterion for query results.

#### `AggregationRequest`

Aggregation specification.

#### `QueryRequest`

Comprehensive query request specification.

#### `QueryResult`

Individual query result item.

#### `AggregationResult`

Result of an aggregation query.

#### `QueryResponse`

Complete query response.

#### `QueryHistory`

Query execution history for analytics.

#### `FacetConfiguration`

Configuration for faceted search.

#### `QueryProcessor`

Abstract base class for query processors.

**Inherits from:** ABC

**Methods:**

- `supports_query_type(self, query_type)`: Check if processor supports the given query type

#### `RetrievalRepository`

Abstract repository for retrieval operations.

**Inherits from:** ABC

#### `FilterValidator`

Validates and normalizes query filters.

**Methods:**

- `__init__(self)`
- `validate_filters(self, filters)`: Validate a list of filters and return validation errors
- `validate_filter(self, filter_item)`: Validate a single filter
- `normalize_filter(self, filter_item)`: Normalize filter values and formats

#### `QueryOptimizer`

Optimizes query requests for better performance.

**Methods:**

- `__init__(self)`
- `optimize_query(self, request)`: Optimize query request for better performance
- `_deep_copy_request(self, request)`: Create a deep copy of the query request
- `_optimize_filters(self, request)`: Optimize filter conditions
- `_optimize_sorting(self, request)`: Optimize sort criteria
- `_optimize_aggregations(self, request)`: Optimize aggregation requests
- `_optimize_limits(self, request)`: Optimize limit and offset values

#### `ExactMatchProcessor`

Processor for exact match queries.

**Inherits from:** QueryProcessor

**Methods:**

- `__init__(self, repository)`
- `supports_query_type(self, query_type)`: Check if processor supports the given query type
- `_is_exact_match(self, query_text, result)`: Check if result is an exact match for the query
- `_apply_sorting(self, results, sort_criteria)`: Apply sorting to results

#### `FuzzySearchProcessor`

Processor for fuzzy search queries using Levenshtein distance.

**Inherits from:** QueryProcessor

**Methods:**

- `__init__(self, repository)`
- `supports_query_type(self, query_type)`: Check if processor supports the given query type
- `_calculate_fuzzy_score(self, query_text, result)`: Calculate fuzzy match score using Levenshtein distance
- `_levenshtein_similarity(self, s1, s2)`: Calculate similarity using Levenshtein distance

#### `SemanticSearchProcessor`

Processor for semantic search queries using embeddings.

**Inherits from:** QueryProcessor

**Methods:**

- `__init__(self, repository, semantic_engine)`
- `supports_query_type(self, query_type)`: Check if processor supports the given query type
- `_cosine_similarity(self, vec1, vec2)`: Calculate cosine similarity between two vectors

#### `QueryEngine`

Main query engine that orchestrates different processors.

**Methods:**

- `__init__(self, repository)`
- `register_processor(self, query_type, processor)`: Register a query processor for a specific query type
- `set_semantic_engine(self, semantic_engine)`: Set semantic engine and register semantic processor
- `get_supported_query_types(self)`: Get list of supported query types
- `get_query_history(self, limit, query_type, success_only)`: Get query execution history



## Module: __init__

Taxonomy management and hierarchy operations.

**Module Path:** `src/echo_roots/taxonomy/__init__.py`



## Module: navigator

Taxonomy Navigation and Tree Operations.

This module provides utilities for navigating and querying the taxonomy hierarchy,
including tree traversal, search, and structural analysis.

**Module Path:** `src/echo_roots/taxonomy/navigator.py`



### Classes

#### `TraversalOrder`

Tree traversal order options.

**Inherits from:** Enum

#### `TreeNode`

Tree representation of a category with navigation utilities.

**Methods:**

- `is_root(self)`: Check if this is a root node
- `is_leaf(self)`: Check if this is a leaf node
- `depth(self)`: Get the depth of this node in the tree
- `subtree_size(self)`: Get the total number of nodes in this subtree
- `find_child(self, name)`: Find a direct child by name
- `get_path_to_root(self)`: Get the path from this node to the root
- `get_common_ancestor(self, other)`: Find the lowest common ancestor with another node

#### `TaxonomyNavigator`

Navigation utilities for taxonomy hierarchies.

Provides tree-based operations, search capabilities, and structural analysis
for taxonomy hierarchies.

**Methods:**

- `__init__(self, taxonomy_repo)`: Initialize the navigator
- `_calculate_category_similarity(self, cat1, cat2)`: Calculate similarity score between two categories
- `clear_cache(self)`: Clear navigation caches



## Module: manager

Taxonomy Management Layer (A Layer) - High-level taxonomy operations.

This module provides comprehensive management for the A-layer taxonomy hierarchy,
including creation, navigation, validation, and governance operations.

**Module Path:** `src/echo_roots/taxonomy/manager.py`



### Classes

#### `TaxonomyStats`

Statistics for taxonomy structure and health.

**Inherits from:** BaseModel

#### `TaxonomyPath`

Utility class for working with taxonomy paths.

**Methods:**

- `build_path(parent_path, category_name)`: Build a complete path for a category given its parent path
- `calculate_level(path)`: Calculate the hierarchy level from a path
- `validate_path_consistency(path, level, name)`: Validate that path is consistent with level and category name
- `get_parent_path(path)`: Get the parent path from a category path

#### `CategoryCreationRequest`

Request model for creating a new category.

**Inherits from:** BaseModel

#### `CategoryMoveRequest`

Request model for moving a category to a new parent.

**Inherits from:** BaseModel

#### `CategoryMergeRequest`

Request model for merging categories.

**Inherits from:** BaseModel

#### `TaxonomyManager`

High-level manager for taxonomy operations and governance.

Provides comprehensive operations for managing the A-layer taxonomy hierarchy
including creation, navigation, validation, and governance workflows.

**Methods:**

- `__init__(self, taxonomy_repo)`: Initialize the taxonomy manager
- `clear_cache(self)`: Clear internal caches



## Module: graph

**Module Path:** `src/echo_roots/semantic/graph.py`



### Classes

#### `GraphQueryType`

Types of knowledge graph queries.

**Inherits from:** str, Enum

#### `GraphMetric`

Knowledge graph metrics.

**Inherits from:** str, Enum

#### `IntegrationType`

Types of semantic integration.

**Inherits from:** str, Enum

#### `GraphNode`

Node in the knowledge graph.

#### `GraphEdge`

Edge in the knowledge graph.

#### `GraphPath`

Path through the knowledge graph.

#### `GraphCluster`

Cluster of related nodes in the graph.

#### `GraphMetrics`

Comprehensive graph metrics.

#### `IntegrationTask`

Task for semantic integration with existing systems.

#### `KnowledgeGraphBuilder`

Builds and maintains the semantic knowledge graph.

**Methods:**

- `__init__(self, repository)`
- `_calculate_cluster_cohesion(self, cluster_nodes, all_edges)`: Calculate internal cohesion of a cluster
- `_get_dominant_entity_types(self, cluster_nodes, all_nodes)`: Get dominant entity types in a cluster
- `_get_representative_terms(self, cluster_nodes, all_nodes)`: Get representative terms for a cluster

#### `SemanticIntegrator`

Integrates semantic enrichment with existing systems.

**Methods:**

- `__init__(self, repository, graph_builder)`



## Module: __init__

**Module Path:** `src/echo_roots/semantic/__init__.py`



### Classes

#### `EmbeddingModel`

Supported embedding models.

**Inherits from:** str, Enum

#### `SemanticRelationType`

Types of semantic relationships.

**Inherits from:** str, Enum

#### `EnrichmentStatus`

Status of semantic enrichment process.

**Inherits from:** str, Enum

#### `ConfidenceLevel`

Confidence levels for semantic relationships.

**Inherits from:** str, Enum

#### `SemanticEmbedding`

Semantic embedding representation.

#### `SemanticRelationship`

Semantic relationship between entities.

#### `SemanticConcept`

High-level semantic concept extracted from data.

#### `EnrichmentTask`

Task for semantic enrichment processing.

#### `SemanticQuery`

Query for semantic search and analysis.

#### `SemanticSearchResult`

Result from semantic search.

#### `EnrichmentStats`

Statistics for semantic enrichment.

#### `EmbeddingProvider`

Abstract base class for embedding providers.

**Inherits from:** ABC

**Methods:**

- `get_embedding_dimensions(self, model)`: Get dimensions for specific model

#### `SemanticRepository`

Abstract repository for semantic data storage.

**Inherits from:** ABC

#### `TextProcessor`

Advanced text processing for semantic analysis.

**Methods:**

- `__init__(self)`
- `extract_keywords(self, text, max_keywords)`: Extract key terms from text
- `extract_phrases(self, text, min_length, max_length)`: Extract meaningful phrases from text
- `clean_text_for_embedding(self, text)`: Clean and prepare text for embedding generation
- `calculate_text_similarity(self, text1, text2)`: Calculate basic text similarity using word overlap

#### `RelationshipExtractor`

Extract semantic relationships between entities.

**Methods:**

- `__init__(self, embedding_provider)`
- `_is_hyponym(self, source, target)`: Check if source is a more specific term than target
- `_is_hypernym(self, source, target)`: Check if source is a more general term than target
- `_is_meronym(self, source, target)`: Check if source is part of target
- `_is_holonym(self, source, target)`: Check if source contains target as part
- `_score_to_confidence_level(self, score)`: Convert numeric score to confidence level
- `_generate_relationship_id(self)`: Generate unique relationship ID

#### `ConceptExtractor`

Extract high-level semantic concepts from entity collections.

**Methods:**

- `__init__(self, embedding_provider)`
- `_generate_concept_name(self, keywords, entities)`: Generate meaningful concept name
- `_determine_concept_type(self, entity_types)`: Determine concept type based on entity types
- `_extract_domains(self, entities)`: Extract domain information from entities

#### `SemanticEnrichmentEngine`

Main engine for semantic enrichment operations.

**Methods:**

- `__init__(self, repository, embedding_provider)`



## Module: search

**Module Path:** `src/echo_roots/semantic/search.py`



### Classes

#### `SearchStrategy`

Different semantic search strategies.

**Inherits from:** str, Enum

#### `RankingStrategy`

Ranking strategies for search results.

**Inherits from:** str, Enum

#### `SearchScope`

Scope of semantic search.

**Inherits from:** str, Enum

#### `SearchConfiguration`

Configuration for semantic search operations.

#### `RankingFactors`

Factors used in result ranking.

#### `SearchContext`

Context for semantic search session.

#### `SearchMetrics`

Metrics for search performance tracking.

#### `QueryExpander`

Expands search queries with related terms.

**Methods:**

- `__init__(self, repository, embedding_provider)`
- `_get_relationship_weight(self, rel_type)`: Get weight for relationship type in expansion

#### `ResultRanker`

Ranks search results using multiple factors.

**Methods:**

- `__init__(self, repository)`
- `_calculate_freshness_score(self, result)`: Calculate freshness score based on recency
- `_extract_confidence_score(self, result)`: Extract confidence score from result metadata
- `_calculate_domain_relevance(self, result, context)`: Calculate domain relevance score
- `_calculate_final_score(self, factors, config)`: Calculate final ranking score based on strategy

#### `SemanticSearchEngine`

Advanced semantic search engine with multiple strategies.

**Methods:**

- `__init__(self, repository, embedding_provider)`
- `_generate_cache_key(self, query, config)`: Generate cache key for search query and config



## Module: __init__

System governance, monitoring, access control, and operational management.
Provides administrative oversight and production-ready operational capabilities.

**Module Path:** `src/echo_roots/governance/__init__.py`



### Classes

#### `AccessLevel`

Access control levels.

**Inherits from:** Enum

#### `AlertSeverity`

Alert severity levels.

**Inherits from:** Enum

#### `User`

User representation for access control.

**Methods:**

- `__post_init__(self)`
- `_generate_api_key(self)`: Generate secure API key
- `has_permission(self, permission)`: Check if user has specific permission
- `can_access_resource(self, resource_access_level)`: Check if user can access resource based on access level

#### `AuditLogEntry`

Audit log entry for tracking system activities.

**Methods:**

- `__post_init__(self)`

#### `SystemMetrics`

System performance and health metrics.

**Methods:**

- `to_dict(self)`: Convert to dictionary for JSON serialization

#### `Alert`

System alert representation.

**Methods:**

- `__post_init__(self)`
- `resolve(self, resolved_by)`: Mark alert as resolved

#### `UserManager`

User management and access control.

**Methods:**

- `__init__(self)`
- `_setup_default_admin(self)`: Create default admin user if none exists
- `add_user(self, user)`: Add new user to the system
- `authenticate_user(self, username, password)`: Authenticate user by username and password
- `authenticate_api_key(self, api_key)`: Authenticate user by API key
- `create_session(self, user)`: Create user session
- `validate_session(self, session_id)`: Validate session and return user
- `revoke_session(self, session_id)`: Revoke user session
- `get_active_users(self)`: Get list of active users

#### `AuditLogger`

Audit logging for tracking system activities.

**Methods:**

- `__init__(self, log_file)`
- `_load_existing_logs(self)`: Load existing audit logs from file
- `log_action(self, user_id, action, resource, details, success, error_message, ip_address, user_agent)`: Log user action for audit trail
- `_write_entry_to_file(self, entry)`: Write audit entry to log file
- `get_logs(self, user_id, action, start_time, end_time, limit)`: Retrieve audit logs with filtering

#### `SystemMonitor`

System monitoring and health checking.

**Methods:**

- `__init__(self)`
- `collect_metrics(self)`: Collect current system metrics
- `_check_alert_conditions(self, metrics)`: Check for conditions that should trigger alerts
- `create_alert(self, severity, title, message, component, metadata)`: Create system alert
- `resolve_alert(self, alert_id, resolved_by)`: Resolve an alert
- `get_active_alerts(self)`: Get all unresolved alerts
- `get_alerts_by_severity(self, severity)`: Get alerts by severity level
- `record_query_time(self, duration)`: Record query execution time for metrics
- `record_error(self)`: Record an error occurrence
- `get_system_health(self)`: Get comprehensive system health report

#### `GovernanceManager`

Main governance and monitoring coordinator.

**Methods:**

- `__init__(self)`
- `_load_config(self)`: Load governance configuration
- `_start_monitoring_tasks(self)`: Start background monitoring tasks
- `log_request(self, user_id, action, resource, details, success, error_message)`: Log API request for audit trail
- `get_dashboard_data(self)`: Get data for governance dashboard



### Functions

#### `require_permission(permission)`

Decorator to require specific permission for API endpoint.



## Module: adapter

Domain adapter for field mapping and data transformation.

This module provides the core domain adaptation functionality, transforming
raw input data according to domain pack specifications. It handles field
mapping, normalization, validation, and prompt template resolution.

Key components:
- DomainAdapter: Main adapter class with transformation logic
- FieldMapper: Handles input field mapping to core schema
- DataTransformer: Applies normalization rules and transformations

**Module Path:** `src/echo_roots/domain/adapter.py`



### Classes

#### `DomainAdapterError`

Raised when domain adaptation fails.

**Inherits from:** Exception

#### `FieldMapper`

Handles mapping of input fields to core schema fields.

Uses the input_mapping configuration from domain packs to transform
raw data dictionaries into standardized core schema format.

**Methods:**

- `__init__(self, domain_pack)`: Initialize field mapper with domain pack configuration
- `map_fields(self, raw_data)`: Map raw input fields to core schema fields
- `generate_stable_id(self, mapped_data)`: Generate a stable ID for an item based on title and source URI

#### `DataTransformer`

Applies normalization rules and data transformations.

Uses rules from domain packs to normalize values, apply mappings,
and filter blocked terms.

**Methods:**

- `__init__(self, domain_pack)`: Initialize data transformer with domain pack rules
- `normalize_text(self, text)`: Apply text normalization rules
- `normalize_attribute_value(self, attribute_key, value)`: Apply attribute-specific value normalization
- `is_blocked_term(self, text)`: Check if text contains blocked terms
- `filter_blocked_content(self, data)`: Filter out content with blocked terms

#### `DomainAdapter`

Main domain adapter for transforming raw data using domain packs.

Combines field mapping, data transformation, and validation to convert
raw input data into standardized IngestionItem objects according to
domain pack specifications.

Example:
    >>> adapter = DomainAdapter.from_file("domains/ecommerce/domain.yaml")
    >>> raw_item = {"product_name": "iPhone", "desc": "Great phone"}
    >>> item = adapter.adapt(raw_item)
    >>> print(item.title, item.description)

**Methods:**

- `__init__(self, domain_pack)`: Initialize domain adapter with a domain pack
- `from_file(cls, path)`: Create domain adapter from a domain pack file
- `from_domain_name(cls, domain_name, base_path)`: Create domain adapter by domain name
- `adapt(self, raw_data, source)`: Adapt raw input data to an IngestionItem
- `adapt_batch(self, raw_items, source)`: Adapt a batch of raw items
- `get_prompt_template(self, prompt_type)`: Get a formatted prompt template from the domain pack
- `get_attribute_config(self, attribute_key)`: Get configuration for a specific attribute
- `get_runtime_config(self)`: Get runtime configuration from the domain pack
- `validate_required_fields(self, data)`: Check if data has required fields according to runtime config
- `should_dedupe_items(self, item1, item2)`: Check if two items should be considered duplicates
- `_get_nested_value(self, data, field_path)`: Get value from nested dictionary using dot notation



## Module: __init__

Domain adaptation and configuration management.

**Module Path:** `src/echo_roots/domain/__init__.py`



## Module: loader

Domain pack loader for YAML configuration files.

This module handles loading, validation, and caching of domain.yaml files.
Domain packs provide flexible configuration for adapting the core framework
to specific domains like e-commerce, media, or knowledge graphs.

Key components:
- DomainPackLoader: Main loader with caching and validation
- load_domain_pack: Convenience function for simple loading
- validate_domain_pack: Standalone validation function

**Module Path:** `src/echo_roots/domain/loader.py`



### Classes

#### `DomainPackLoadError`

Raised when domain pack loading fails.

**Inherits from:** Exception

**Methods:**

- `__init__(self, message, path, cause)`

#### `DomainPackLoader`

Loads and validates domain pack YAML files with caching.

Provides loading, validation, and caching of domain.yaml files.
Supports both file paths and directory scanning.

Example:
    >>> loader = DomainPackLoader()
    >>> pack = loader.load("domains/ecommerce/domain.yaml")
    >>> print(pack.domain, pack.taxonomy_version)
    
    >>> # Load from directory (looks for domain.yaml)
    >>> pack = loader.load_from_directory("domains/ecommerce")

**Methods:**

- `__init__(self, cache_size, validate_on_load)`: Initialize the domain pack loader
- `load(self, path)`: Load a domain pack from a YAML file
- `load_from_directory(self, directory)`: Load a domain pack from a directory (looks for domain
- `reload(self, path)`: Reload a domain pack, bypassing cache
- `clear_cache(self)`: Clear the domain pack cache
- `list_cached(self)`: List currently cached domain packs



### Functions

#### `load_domain_pack(path, validate)`

Load a domain pack from a YAML file (convenience function).

Args:
    path: Path to the domain.yaml file
    validate: Whether to validate the pack during loading
    
Returns:
    Loaded DomainPack
    
Raises:
    DomainPackLoadError: If loading or validation fails
    
Example:
    >>> pack = load_domain_pack("domains/ecommerce/domain.yaml")
    >>> print(f"Loaded {pack.domain} v{pack.taxonomy_version}")

**Returns:** `DomainPack`

#### `validate_domain_pack(data)`

Validate raw domain pack data.

Args:
    data: Raw domain pack data (from YAML)
    
Returns:
    Validated DomainPack
    
Raises:
    ValidationError: If validation fails
    
Example:
    >>> with open("domain.yaml") as f:
    ...     raw_data = yaml.safe_load(f)
    >>> pack = validate_domain_pack(raw_data)

**Returns:** `DomainPack`

#### `load_cached_domain_pack(path)`

Load a domain pack with simple LRU caching.

Args:
    path: Path to the domain.yaml file (as string for hashing)
    
Returns:
    Cached DomainPack
    
Note:
    This is a simple caching function. For more control,
    use DomainPackLoader directly.

**Decorators:** lru_cache

**Returns:** `DomainPack`

#### `scan_domain_packs(base_directory)`

Scan a directory tree for domain packs.

Args:
    base_directory: Root directory to scan
    
Returns:
    Dict mapping domain names to loaded DomainPacks
    
Raises:
    DomainPackLoadError: If any domain pack fails to load
    
Example:
    >>> packs = scan_domain_packs("domains/")
    >>> print(f"Found domains: {list(packs.keys())}")

#### `get_domain_pack_info(path)`

Get basic info about a domain pack without full validation.

Args:
    path: Path to the domain.yaml file
    
Returns:
    Dict with basic domain pack information
    
Raises:
    DomainPackLoadError: If file cannot be read
    
Example:
    >>> info = get_domain_pack_info("domains/ecommerce/domain.yaml")
    >>> print(f"Domain: {info['domain']}, Version: {info['taxonomy_version']}")



## Module: merger

Schema merging and validation utilities.

This module provides utilities for merging domain-specific schemas with
the core schema, validating attribute definitions, and resolving conflicts.
It supports both schema-level merging and runtime validation.

Key components:
- SchemaMerger: Main class for schema merging operations
- AttributeValidator: Validates attribute definitions and values
- SchemaConflictResolver: Handles conflicts during merging

**Module Path:** `src/echo_roots/domain/merger.py`



### Classes

#### `ConflictResolution`

Strategies for resolving schema conflicts.

**Inherits from:** Enum

#### `SchemaValidationError`

Raised when schema validation fails.

**Inherits from:** Exception

#### `SchemaConflictError`

Raised when schema conflicts cannot be resolved.

**Inherits from:** Exception

#### `AttributeValidator`

Validates attribute definitions and values against schema rules.

Provides validation for both domain-specific attribute configurations
and extracted attribute values.

**Methods:**

- `__init__(self, domain_pack)`: Initialize validator with domain pack configuration
- `_build_attribute_configs(self)`: Build attribute configurations from domain pack
- `validate_attribute_definition(self, key, definition)`: Validate an attribute definition
- `validate_attribute_value(self, key, value)`: Validate an attribute value against its configuration
- `validate_extraction(self, extraction)`: Validate an attribute extraction against schema rules

#### `SchemaConflictResolver`

Handles conflicts during schema merging operations.

**Methods:**

- `__init__(self, resolution_strategy)`: Initialize conflict resolver with strategy
- `resolve_attribute_conflict(self, key, core_definition, domain_definition)`: Resolve conflicting attribute definitions
- `_find_conflicts(self, core_def, domain_def)`: Find conflicting fields between definitions
- `_merge_definitions(self, core_def, domain_def)`: Merge two attribute definitions intelligently

#### `SchemaMerger`

Main class for schema merging operations.

Merges domain-specific schemas with core schemas, validates the result,
and provides utilities for working with merged schemas.

**Methods:**

- `__init__(self, conflict_resolution, validate_merge)`: Initialize schema merger
- `_load_core_schema(self)`: Load the core schema definition
- `merge_schemas(self, domain_pack)`: Merge domain pack schema with core schema
- `validate_merged_schema(self, schema)`: Validate a merged schema for consistency and completeness
- `get_attribute_schema(self, schema, attribute_key)`: Get schema definition for a specific attribute
- `list_attributes(self, schema)`: List all attributes in a merged schema
- `export_schema(self, schema, format)`: Export merged schema in specified format



### Functions

#### `merge_domain_schemas()`

Convenience function to merge multiple domain pack schemas.

Args:
    *domain_packs: Domain packs to merge
    conflict_resolution: Strategy for resolving conflicts
    
Returns:
    Merged schema from all domain packs



## Module: __init__

**Module Path:** `src/echo_roots/vocabulary/__init__.py`



## Module: navigator

**Module Path:** `src/echo_roots/vocabulary/navigator.py`



### Classes

#### `VocabularyHierarchy`

Represents a vocabulary hierarchy tree.

#### `VocabularyCluster`

A cluster of related vocabulary terms.

#### `VocabularyRecommendation`

Recommendation for vocabulary improvement.

#### `VocabularyNavigator`

Navigation utilities for vocabulary hierarchies and relationships.

**Methods:**

- `__init__(self, repository)`
- `_calculate_depths(self, hierarchy)`: Calculate depth for each term in hierarchy
- `_calculate_string_similarity(self, str1, str2)`: Calculate string similarity using simple algorithm

#### `VocabularyAnalyzer`

Analyzes vocabulary for quality, coverage, and improvement opportunities.

**Methods:**

- `__init__(self, repository, navigator)`



## Module: manager

**Module Path:** `src/echo_roots/vocabulary/manager.py`



### Classes

#### `VocabularyType`

Types of controlled vocabularies.

**Inherits from:** str, Enum

#### `ValidationLevel`

Validation strictness levels.

**Inherits from:** str, Enum

#### `MappingConfidence`

Confidence levels for vocabulary mappings.

**Inherits from:** str, Enum

#### `VocabularyTerm`

A controlled vocabulary term with metadata.

#### `VocabularyMapping`

Mapping between raw input and controlled vocabulary.

#### `ValidationResult`

Result of vocabulary validation.

#### `VocabularyRequest`

Request to create or update vocabulary terms.

#### `VocabularyStats`

Statistics about vocabulary coverage and quality.

#### `VocabularyRepository`

Abstract repository for vocabulary storage.

**Inherits from:** ABC

#### `VocabularyNormalizer`

Normalizes vocabulary values for consistent matching.

**Methods:**

- `__init__(self)`
- `normalize_term(self, term)`: Normalize a term for consistent matching
- `extract_units(self, text)`: Extract units and measurements from text
- `_expand_abbreviations(self, text)`: Expand common abbreviations

#### `VocabularyMatcher`

Matches raw input to controlled vocabulary terms.

**Methods:**

- `__init__(self, normalizer)`
- `_score_to_confidence(self, score)`: Convert numeric score to confidence level

#### `VocabularyValidator`

Validates vocabulary values against controlled vocabularies.

**Methods:**

- `__init__(self, matcher)`

#### `VocabularyManager`

High-level manager for controlled vocabulary operations.

**Methods:**

- `__init__(self, repository)`
- `_generate_term_id(self)`: Generate a unique term ID
- `_generate_mapping_id(self)`: Generate a unique mapping ID
- `_score_to_confidence(self, score)`: Convert numeric score to confidence level
- `_clear_cache(self)`: Clear internal caches



