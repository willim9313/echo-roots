"""LLM-based extraction pipeline for attribute and term extraction.

This module provides the core extraction pipeline that processes IngestionItem objects
through LLM models to extract structured attributes and semantic terms according to
domain-specific configurations.

Key components:
- ExtractorConfig: Configuration for LLM extraction settings
- PromptBuilder: Builds domain-specific prompts for LLM calls
- LLMExtractor: Main extraction engine with LLM integration
- ExtractionPipeline: High-level pipeline orchestrator
- BatchProcessor: Efficient batch processing of multiple items
"""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Protocol
from pathlib import Path
import asyncio
import time

from pydantic import BaseModel, Field, validator
from echo_roots.models.core import (
    IngestionItem, 
    ExtractionResult, 
    AttributeExtraction, 
    SemanticTerm, 
    ExtractionMetadata
)
from echo_roots.models.domain import DomainPack
from echo_roots.domain.loader import DomainPackLoader


class ExtractorConfig(BaseModel):
    """Configuration for LLM-based extraction.
    
    Controls model selection, prompt behavior, and processing parameters
    for the extraction pipeline.
    """
    
    model_name: str = Field(
        default="gpt-4",
        description="LLM model identifier for extraction"
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for extraction (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for LLM response",
        gt=0
    )
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout in seconds",
        gt=0
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests",
        ge=0
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for concurrent processing",
        gt=0
    )
    enable_validation: bool = Field(
        default=True,
        description="Whether to validate extracted data against domain schema"
    )


class ExtractionError(Exception):
    """Base exception for extraction pipeline errors."""
    
    def __init__(self, message: str, item_id: Optional[str] = None, cause: Optional[Exception] = None):
        self.message = message
        self.item_id = item_id
        self.cause = cause
        super().__init__(message)


class PromptBuilder:
    """Builds domain-specific prompts for LLM extraction.
    
    Constructs prompts by combining base templates with domain-specific
    configurations, attribute hints, and output schema specifications.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize prompt builder with domain pack.
        
        Args:
            domain_pack: Domain configuration for prompt customization
        """
        self.domain_pack = domain_pack
        self.base_template = self._load_base_template()
    
    def _load_base_template(self) -> str:
        """Load the base extraction prompt template."""
        # Default template - in production this might be loaded from a file
        return '''You are an expert at extracting structured attributes and semantic terms from text.

Domain: {{DOMAIN}}
Taxonomy Version: {{TAXONOMY_VERSION}}

Expected Attributes: {{OUTPUT_KEYS_JSON}}
Attribute Hints: {{ATTRIBUTE_HINTS}}

Instructions:
1. Extract attributes with name, value, and supporting evidence from the text
2. Extract semantic terms with context and confidence scores (0.0-1.0)
3. Use the expected attributes list as guidance but don't force extractions
4. Preserve original language and casing where appropriate
5. Provide clear evidence text for each extraction

Input Item:
Title: {{TITLE}}
Description: {{DESCRIPTION}}
Language: {{LANGUAGE}}

Return ONLY a valid JSON object in this exact format:
{
    "attributes": [
        {"name": "attribute_name", "value": "extracted_value", "evidence": "supporting text"}
    ],
    "terms": [
        {"term": "semantic_term", "context": "surrounding context", "confidence": 0.85}
    ]
}'''
    
    def build_prompt(self, item: IngestionItem) -> str:
        """Build extraction prompt for a specific item.
        
        Args:
            item: IngestionItem to extract from
            
        Returns:
            Formatted prompt string ready for LLM
        """
        # Get expected attribute keys from domain pack
        output_keys = []
        if 'attributes' in self.domain_pack.output_schema:
            for attr in self.domain_pack.output_schema['attributes']:
                if isinstance(attr, dict) and 'key' in attr:
                    output_keys.append(attr['key'])
        
        # Prepare attribute hints
        hints_text = "None provided"
        if self.domain_pack.attribute_hints:
            hints_lines = []
            for key, hints in self.domain_pack.attribute_hints.items():
                hint_parts = []
                if 'examples' in hints:
                    hint_parts.append(f"examples: {hints['examples']}")
                if 'pattern' in hints:
                    hint_parts.append(f"pattern: {hints['pattern']}")
                if hint_parts:
                    hints_lines.append(f"- {key}: {', '.join(hint_parts)}")
            if hints_lines:
                hints_text = "\\n".join(hints_lines)
        
        # Fill template variables
        prompt = self.base_template.replace('{{DOMAIN}}', self.domain_pack.domain)
        prompt = prompt.replace('{{TAXONOMY_VERSION}}', self.domain_pack.taxonomy_version)
        prompt = prompt.replace('{{OUTPUT_KEYS_JSON}}', json.dumps(output_keys))
        prompt = prompt.replace('{{ATTRIBUTE_HINTS}}', hints_text)
        prompt = prompt.replace('{{TITLE}}', item.title)
        prompt = prompt.replace('{{DESCRIPTION}}', item.description or "")
        prompt = prompt.replace('{{LANGUAGE}}', item.language)
        
        return prompt
    
    def build_batch_prompt(self, items: List[IngestionItem]) -> str:
        """Build prompt for batch extraction of multiple items.
        
        Args:
            items: List of IngestionItems to extract from
            
        Returns:
            Formatted batch prompt string
        """
        if not items:
            raise ValueError("Cannot build prompt for empty item list")
        
        # Use domain pack from the builder
        output_keys = []
        if 'attributes' in self.domain_pack.output_schema:
            for attr in self.domain_pack.output_schema['attributes']:
                if isinstance(attr, dict) and 'key' in attr:
                    output_keys.append(attr['key'])
        
        # Build batch template
        batch_template = '''You are an expert at extracting structured attributes and semantic terms from text.

Domain: {{DOMAIN}}
Taxonomy Version: {{TAXONOMY_VERSION}}
Expected Attributes: {{OUTPUT_KEYS_JSON}}

Instructions:
1. Process each item separately
2. Extract attributes with name, value, and supporting evidence
3. Extract semantic terms with context and confidence scores
4. Return a JSON array with one object per item in the same order

Items to process:
{{ITEMS}}

Return ONLY a valid JSON array in this format:
[
    {
        "item_id": "item_1_id",
        "attributes": [...],
        "terms": [...]
    },
    ...
]'''
        
        # Format items section
        items_text = ""
        for i, item in enumerate(items, 1):
            items_text += f"\\nItem {i} (ID: {item.item_id}):\\n"
            items_text += f"Title: {item.title}\\n"
            if item.description:
                items_text += f"Description: {item.description}\\n"
            items_text += f"Language: {item.language}\\n"
        
        # Fill template
        prompt = batch_template.replace('{{DOMAIN}}', self.domain_pack.domain)
        prompt = prompt.replace('{{TAXONOMY_VERSION}}', self.domain_pack.taxonomy_version)
        prompt = prompt.replace('{{OUTPUT_KEYS_JSON}}', json.dumps(output_keys))
        prompt = prompt.replace('{{ITEMS}}', items_text)
        
        return prompt


class LLMClient(Protocol):
    """Protocol for LLM client implementations.
    
    Defines the interface that LLM clients must implement to work
    with the extraction pipeline.
    """
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt with the LLM.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            LLM response text
        """
        ...


class MockLLMClient:
    """Mock LLM client for testing and development.
    
    Provides deterministic responses for testing the extraction pipeline
    without requiring actual LLM API calls.
    """
    
    def __init__(self, delay_seconds: float = 0.1):
        """Initialize mock client.
        
        Args:
            delay_seconds: Simulated API call delay
        """
        self.delay_seconds = delay_seconds
    
    async def complete(self, prompt: str, **kwargs) -> str:
        """Mock completion that returns structured test data.
        
        Args:
            prompt: Input prompt (analyzed for test data generation)
            **kwargs: Ignored additional parameters
            
        Returns:
            Mock JSON response
        """
        await asyncio.sleep(self.delay_seconds)
        
        # Extract item ID from prompt for consistent responses
        if "iPhone" in prompt or "Apple" in prompt:
            return json.dumps({
                "attributes": [
                    {"name": "brand", "value": "Apple", "evidence": "Apple iPhone mentioned in title", "confidence": 0.95},
                    {"name": "category", "value": "smartphone", "evidence": "iPhone is a smartphone", "confidence": 0.90}
                ],
                "terms": [
                    {"term": "smartphone", "context": "mobile device technology", "confidence": 0.95},
                    {"term": "Apple", "context": "technology brand", "confidence": 0.99}
                ]
            })
        else:
            # Generic response
            return json.dumps({
                "attributes": [
                    {"name": "category", "value": "unknown", "evidence": "category not clearly identified", "confidence": 0.60}
                ],
                "terms": [
                    {"term": "product", "context": "general product reference", "confidence": 0.7}
                ]
            })


class LLMExtractor:
    """Main LLM extraction engine.
    
    Orchestrates the extraction process including prompt building,
    LLM calls, response parsing, and validation.
    """
    
    def __init__(self, 
                 domain_pack: DomainPack,
                 llm_client: LLMClient,
                 config: Optional[ExtractorConfig] = None):
        """Initialize extractor.
        
        Args:
            domain_pack: Domain configuration
            llm_client: LLM client implementation
            config: Extraction configuration (uses defaults if None)
        """
        self.domain_pack = domain_pack
        self.llm_client = llm_client
        self.config = config or ExtractorConfig()
        self.prompt_builder = PromptBuilder(domain_pack)
    
    async def extract_single(self, item: IngestionItem) -> ExtractionResult:
        """Extract attributes and terms from a single item.
        
        Args:
            item: IngestionItem to process
            
        Returns:
            ExtractionResult with extracted data
            
        Raises:
            ExtractionError: If extraction fails
        """
        try:
            start_time = time.time()
            run_id = f"extract_{uuid.uuid4().hex[:8]}"
            
            # Build prompt
            prompt = self.prompt_builder.build_prompt(item)
            
            # Call LLM with retries
            response_text = await self._call_llm_with_retry(prompt)
            
            # Parse response
            extraction_data = self._parse_llm_response(response_text)
            
            # Create extraction objects
            attributes = [
                AttributeExtraction(**attr_data) 
                for attr_data in extraction_data.get('attributes', [])
            ]
            
            terms = [
                SemanticTerm(**term_data)
                for term_data in extraction_data.get('terms', [])
            ]
            
            # Create metadata
            processing_time_ms = int((time.time() - start_time) * 1000)
            metadata = ExtractionMetadata(
                model=self.config.model_name,
                run_id=run_id,
                extracted_at=datetime.now(),
                processing_time_ms=processing_time_ms
            )
            
            # Create result
            result = ExtractionResult(
                item_id=item.item_id,
                attributes=attributes,
                terms=terms,
                metadata=metadata
            )
            
            # Validate if enabled
            if self.config.enable_validation:
                self._validate_extraction(result)
            
            return result
            
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract from item {item.item_id}: {e}",
                item_id=item.item_id,
                cause=e
            )
    
    async def extract_batch(self, items: List[IngestionItem]) -> List[ExtractionResult]:
        """Extract from multiple items efficiently.
        
        Args:
            items: List of IngestionItems to process
            
        Returns:
            List of ExtractionResults in same order as input
        """
        if not items:
            return []
        
        # Process in batches
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.extract_single(item) for item in batch],
                return_exceptions=True
            )
            
            # Handle exceptions in batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Create empty result for failed items
                    failed_item = batch[j]
                    result = ExtractionResult(
                        item_id=failed_item.item_id,
                        attributes=[],
                        terms=[],
                        metadata=ExtractionMetadata(
                            model=self.config.model_name,
                            run_id=f"failed_{uuid.uuid4().hex[:8]}",
                            extracted_at=datetime.now()
                        )
                    )
                
                results.append(result)
        
        return results
    
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with retry logic.
        
        Args:
            prompt: Prompt text to send
            
        Returns:
            LLM response text
            
        Raises:
            ExtractionError: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self.llm_client.complete(
                        prompt,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    ),
                    timeout=self.config.timeout_seconds
                )
                return response
                
            except Exception as e:
                last_exception = e
                if attempt < self.config.retry_attempts:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    continue
                break
        
        raise ExtractionError(
            f"LLM call failed after {self.config.retry_attempts + 1} attempts",
            cause=last_exception
        )
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured data.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed extraction data
            
        Raises:
            ExtractionError: If parsing fails
        """
        try:
            # Clean response text - remove any surrounding markdown or extra text
            response_text = response_text.strip()
            
            # Find JSON block
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end > start:
                    response_text = response_text[start:end].strip()
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Response must be a JSON object")
            
            # Ensure required fields exist
            if 'attributes' not in data:
                data['attributes'] = []
            if 'terms' not in data:
                data['terms'] = []
                
            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            raise ExtractionError(f"Failed to parse LLM response: {e}")
    
    def _validate_extraction(self, result: ExtractionResult) -> None:
        """Validate extraction result against domain schema.
        
        Args:
            result: ExtractionResult to validate
            
        Raises:
            ExtractionError: If validation fails
        """
        try:
            # Import here to avoid circular imports
            from echo_roots.domain.merger import AttributeValidator
            
            validator = AttributeValidator(self.domain_pack)
            
            for attr in result.attributes:
                errors = validator.validate_extraction(attr)
                if errors:
                    raise ExtractionError(
                        f"Validation failed for attribute '{attr.name}': {'; '.join(errors)}",
                        item_id=result.item_id
                    )
        
        except ImportError:
            # If validator not available, skip validation
            pass


class ExtractionPipeline:
    """High-level extraction pipeline orchestrator.
    
    Provides a simple interface for domain-based extraction with
    automatic domain loading and configuration.
    """
    
    def __init__(self, 
                 domains_path: Union[str, Path] = "domains",
                 llm_client: Optional[LLMClient] = None,
                 config: Optional[ExtractorConfig] = None):
        """Initialize extraction pipeline.
        
        Args:
            domains_path: Path to domains directory
            llm_client: LLM client (uses MockLLMClient if None)
            config: Extraction configuration
        """
        self.domains_path = Path(domains_path)
        self.llm_client = llm_client or MockLLMClient()
        self.config = config or ExtractorConfig()
        self.loader = DomainPackLoader()
        self._extractors: Dict[str, LLMExtractor] = {}
    
    def get_extractor(self, domain: str) -> LLMExtractor:
        """Get or create extractor for a domain.
        
        Args:
            domain: Domain name
            
        Returns:
            LLMExtractor for the domain
        """
        if domain not in self._extractors:
            # Load domain pack
            domain_file = self.domains_path / domain / "domain.yaml"
            domain_pack = self.loader.load(str(domain_file))
            
            # Create extractor
            self._extractors[domain] = LLMExtractor(
                domain_pack=domain_pack,
                llm_client=self.llm_client,
                config=self.config
            )
        
        return self._extractors[domain]
    
    async def extract(self, 
                     items: Union[IngestionItem, List[IngestionItem]], 
                     domain: str) -> Union[ExtractionResult, List[ExtractionResult]]:
        """Extract attributes and terms from items.
        
        Args:
            items: Single item or list of items to process
            domain: Domain name for extraction configuration
            
        Returns:
            Single result or list of results matching input type
        """
        extractor = self.get_extractor(domain)
        
        if isinstance(items, IngestionItem):
            return await extractor.extract_single(items)
        else:
            return await extractor.extract_batch(items)
    
    def list_available_domains(self) -> List[str]:
        """List available domains for extraction.
        
        Returns:
            List of available domain names
        """
        if not self.domains_path.exists():
            return []
        
        domains = []
        for path in self.domains_path.iterdir():
            if path.is_dir() and (path / "domain.yaml").exists():
                domains.append(path.name)
        
        return sorted(domains)


# Convenience functions for common use cases
async def extract_single(item: IngestionItem, 
                        domain: str,
                        domains_path: Union[str, Path] = "domains",
                        llm_client: Optional[LLMClient] = None) -> ExtractionResult:
    """Convenience function to extract from a single item.
    
    Args:
        item: IngestionItem to process
        domain: Domain name
        domains_path: Path to domains directory
        llm_client: LLM client (uses mock if None)
        
    Returns:
        ExtractionResult
    """
    pipeline = ExtractionPipeline(domains_path, llm_client)
    return await pipeline.extract(item, domain)


async def extract_batch(items: List[IngestionItem],
                       domain: str, 
                       domains_path: Union[str, Path] = "domains",
                       llm_client: Optional[LLMClient] = None) -> List[ExtractionResult]:
    """Convenience function to extract from multiple items.
    
    Args:
        items: List of IngestionItems to process
        domain: Domain name
        domains_path: Path to domains directory
        llm_client: LLM client (uses mock if None)
        
    Returns:
        List of ExtractionResults
    """
    pipeline = ExtractionPipeline(domains_path, llm_client)
    return await pipeline.extract(items, domain)
