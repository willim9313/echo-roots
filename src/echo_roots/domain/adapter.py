"""Domain adapter for field mapping and data transformation.

This module provides the core domain adaptation functionality, transforming
raw input data according to domain pack specifications. It handles field
mapping, normalization, validation, and prompt template resolution.

Key components:
- DomainAdapter: Main adapter class with transformation logic
- FieldMapper: Handles input field mapping to core schema
- DataTransformer: Applies normalization rules and transformations
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import re
import hashlib

from echo_roots.models.domain import DomainPack
from echo_roots.models.core import IngestionItem
from echo_roots.domain.loader import DomainPackLoader, load_domain_pack


class DomainAdapterError(Exception):
    """Raised when domain adaptation fails."""
    pass


class FieldMapper:
    """Handles mapping of input fields to core schema fields.
    
    Uses the input_mapping configuration from domain packs to transform
    raw data dictionaries into standardized core schema format.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize field mapper with domain pack configuration.
        
        Args:
            domain_pack: Domain pack with input_mapping configuration
        """
        self.domain_pack = domain_pack
        self.input_mapping = domain_pack.input_mapping
    
    def map_fields(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map raw input fields to core schema fields.
        
        Args:
            raw_data: Raw input data dictionary
            
        Returns:
            Mapped data with core schema field names
            
        Example:
            >>> raw = {"product_name": "iPhone", "desc": "Great phone"}
            >>> mapped = mapper.map_fields(raw)
            >>> # mapped = {"title": "iPhone", "description": "Great phone"}
        """
        mapped = {}
        unmapped_fields = {}
        
        # Process each core field mapping
        for target_field, source_fields in self.input_mapping.items():
            value = None
            
            # Try each possible source field in order
            for source_field in source_fields:
                if source_field in raw_data and raw_data[source_field] is not None:
                    value = raw_data[source_field]
                    break
            
            if value is not None:
                mapped[target_field] = value
        
        # Collect unmapped fields for raw_attributes
        for key, value in raw_data.items():
            # Check if this field was used in any mapping
            used_in_mapping = any(
                key in source_fields 
                for source_fields in self.input_mapping.values()
            )
            
            if not used_in_mapping and value is not None:
                unmapped_fields[key] = value
        
        # Add unmapped fields to raw_attributes if any exist
        if unmapped_fields:
            mapped['raw_attributes'] = unmapped_fields
        
        return mapped
    
    def generate_stable_id(self, mapped_data: Dict[str, Any]) -> str:
        """Generate a stable ID for an item based on title and source URI.
        
        Args:
            mapped_data: Mapped data dictionary
            
        Returns:
            Stable hash-based ID
        """
        # Use title + source URI for stable hashing
        title = mapped_data.get('title', '')
        source_uri = mapped_data.get('source_uri', '')
        
        # Create stable hash
        content = f"{title}|{source_uri}".encode('utf-8')
        hash_hex = hashlib.sha256(content).hexdigest()
        
        # Return first 12 characters for readability
        return f"item_{hash_hex[:12]}"


class DataTransformer:
    """Applies normalization rules and data transformations.
    
    Uses rules from domain packs to normalize values, apply mappings,
    and filter blocked terms.
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize data transformer with domain pack rules.
        
        Args:
            domain_pack: Domain pack with transformation rules
        """
        self.domain_pack = domain_pack
        self.rules = domain_pack.rules
    
    def normalize_text(self, text: str) -> str:
        """Apply text normalization rules.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str) or not text.strip():
            return text
        
        normalized = text.strip()
        
        # Apply normalize_map rules
        normalize_map = self.rules.get('normalize_map', {})
        for original, replacement in normalize_map.items():
            normalized = re.sub(
                re.escape(original), 
                replacement, 
                normalized, 
                flags=re.IGNORECASE
            )
        
        return normalized
    
    def normalize_attribute_value(self, attribute_key: str, value: str) -> str:
        """Apply attribute-specific value normalization.
        
        Args:
            attribute_key: The attribute key (e.g., 'color', 'brand')
            value: The value to normalize
            
        Returns:
            Normalized value
        """
        if not isinstance(value, str) or not value.strip():
            return value
        
        normalized = value.strip()
        
        # Apply general text normalization first
        normalized = self.normalize_text(normalized)
        
        # Apply attribute-specific value maps
        value_maps = self.rules.get('value_maps', {})
        if attribute_key in value_maps:
            attr_map = value_maps[attribute_key]
            for original, replacement in attr_map.items():
                if normalized.lower() == original.lower():
                    normalized = replacement
                    break
        
        return normalized
    
    def is_blocked_term(self, text: str) -> bool:
        """Check if text contains blocked terms.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains blocked terms
        """
        if not isinstance(text, str):
            return False
        
        blocked_terms = self.rules.get('blocked_terms', [])
        text_lower = text.lower()
        
        return any(blocked.lower() in text_lower for blocked in blocked_terms)
    
    def filter_blocked_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out content with blocked terms.
        
        Args:
            data: Data dictionary to filter
            
        Returns:
            Filtered data (may be empty if blocked)
        """
        # Check key text fields for blocked terms
        text_fields = ['title', 'description']
        
        for field in text_fields:
            if field in data and self.is_blocked_term(str(data[field])):
                # Return empty dict to indicate blocked content
                return {}
        
        return data


class DomainAdapter:
    """Main domain adapter for transforming raw data using domain packs.
    
    Combines field mapping, data transformation, and validation to convert
    raw input data into standardized IngestionItem objects according to
    domain pack specifications.
    
    Example:
        >>> adapter = DomainAdapter.from_file("domains/ecommerce/domain.yaml")
        >>> raw_item = {"product_name": "iPhone", "desc": "Great phone"}
        >>> item = adapter.adapt(raw_item)
        >>> print(item.title, item.description)
    """
    
    def __init__(self, domain_pack: DomainPack):
        """Initialize domain adapter with a domain pack.
        
        Args:
            domain_pack: Domain pack configuration
        """
        self.domain_pack = domain_pack
        self.field_mapper = FieldMapper(domain_pack)
        self.data_transformer = DataTransformer(domain_pack)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'DomainAdapter':
        """Create domain adapter from a domain pack file.
        
        Args:
            path: Path to domain.yaml file
            
        Returns:
            Configured DomainAdapter
        """
        domain_pack = load_domain_pack(path)
        return cls(domain_pack)
    
    @classmethod
    def from_domain_name(cls, domain_name: str, base_path: Union[str, Path] = "domains") -> 'DomainAdapter':
        """Create domain adapter by domain name.
        
        Args:
            domain_name: Name of the domain (e.g., 'ecommerce')
            base_path: Base directory to search for domain packs
            
        Returns:
            Configured DomainAdapter
        """
        domain_dir = Path(base_path) / domain_name
        loader = DomainPackLoader()
        domain_pack = loader.load_from_directory(domain_dir)
        return cls(domain_pack)
    
    def adapt(self, raw_data: Dict[str, Any], source: str = "unknown") -> Optional[IngestionItem]:
        """Adapt raw input data to an IngestionItem.
        
        Args:
            raw_data: Raw input data dictionary
            source: Source identifier for the data
            
        Returns:
            IngestionItem if successful, None if blocked or invalid
            
        Raises:
            DomainAdapterError: If adaptation fails
        """
        try:
            # Map fields to core schema
            mapped_data = self.field_mapper.map_fields(raw_data)
            
            # Apply content filtering
            filtered_data = self.data_transformer.filter_blocked_content(mapped_data)
            if not filtered_data:
                return None  # Content was blocked
            
            # Apply text normalization
            if 'title' in filtered_data:
                filtered_data['title'] = self.data_transformer.normalize_text(filtered_data['title'])
            
            if 'description' in filtered_data:
                filtered_data['description'] = self.data_transformer.normalize_text(filtered_data['description'])
            
            # Generate stable ID if missing
            if 'id' not in filtered_data or not filtered_data['id']:
                filtered_data['id'] = self.field_mapper.generate_stable_id(filtered_data)
            
            # Set default values according to domain pack runtime config
            runtime = self.domain_pack.runtime
            
            if 'language' not in filtered_data:
                filtered_data['language'] = runtime.language_default
            
            # Prepare source information
            source_info = {
                'uri': filtered_data.pop('source_uri', ''),
                'collected_at': filtered_data.pop('collected_at', datetime.now()),
            }
            
            # Handle raw attributes
            raw_attributes = filtered_data.pop('raw_attributes', {})
            
            # Create IngestionItem
            item = IngestionItem(
                item_id=filtered_data['id'],
                title=filtered_data['title'],
                description=filtered_data.get('description'),
                language=filtered_data.get('language', 'auto'),
                source=source,
                source_uri=source_info['uri'],
                collected_at=source_info['collected_at'],
                raw_attributes=raw_attributes,
                metadata={
                    'domain': self.domain_pack.domain,
                    'taxonomy_version': self.domain_pack.taxonomy_version,
                    'adapted_at': datetime.now().isoformat(),
                }
            )
            
            return item
            
        except Exception as e:
            raise DomainAdapterError(f"Failed to adapt data: {e}") from e
    
    def adapt_batch(self, raw_items: List[Dict[str, Any]], source: str = "unknown") -> List[IngestionItem]:
        """Adapt a batch of raw items.
        
        Args:
            raw_items: List of raw data dictionaries
            source: Source identifier for the data
            
        Returns:
            List of successfully adapted IngestionItems
        """
        adapted_items = []
        
        for raw_item in raw_items:
            try:
                adapted = self.adapt(raw_item, source)
                if adapted:  # Skip None (blocked) items
                    adapted_items.append(adapted)
            except DomainAdapterError:
                # Skip items that fail to adapt
                continue
        
        return adapted_items
    
    def get_prompt_template(self, prompt_type: str, **kwargs) -> Optional[str]:
        """Get a formatted prompt template from the domain pack.
        
        Args:
            prompt_type: Type of prompt (e.g., 'attribute_extraction')
            **kwargs: Variables for template substitution
            
        Returns:
            Formatted prompt string if found, None otherwise
        """
        return self.domain_pack.get_prompt_template(prompt_type, **kwargs)
    
    def get_attribute_config(self, attribute_key: str):
        """Get configuration for a specific attribute.
        
        Args:
            attribute_key: The attribute key to look up
            
        Returns:
            AttributeConfig object if found, None otherwise
        """
        return self.domain_pack.get_attribute_config(attribute_key)
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration from the domain pack.
        
        Returns:
            Runtime configuration dictionary
        """
        return {
            'domain': self.domain_pack.domain,
            'taxonomy_version': self.domain_pack.taxonomy_version,
            'language_default': self.domain_pack.runtime.language_default,
            'dedupe_by': self.domain_pack.runtime.dedupe_by,
            'skip_if_missing': self.domain_pack.runtime.skip_if_missing,
            'batch_size': self.domain_pack.runtime.batch_size,
            'timeout_seconds': self.domain_pack.runtime.timeout_seconds,
        }
    
    def validate_required_fields(self, data: Dict[str, Any]) -> bool:
        """Check if data has required fields according to runtime config.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            True if all required fields are present
        """
        skip_if_missing = self.domain_pack.runtime.skip_if_missing
        
        for field in skip_if_missing:
            if field not in data or not data[field]:
                return False
        
        return True
    
    def should_dedupe_items(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two items should be considered duplicates.
        
        Args:
            item1: First item data
            item2: Second item data
            
        Returns:
            True if items are duplicates according to dedupe_by config
        """
        dedupe_by = self.domain_pack.runtime.dedupe_by
        
        if not dedupe_by:
            return False
        
        for field in dedupe_by:
            # Handle nested fields like 'source.uri'
            value1 = self._get_nested_value(item1, field)
            value2 = self._get_nested_value(item2, field)
            
            if value1 != value2:
                return False
        
        return True
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation.
        
        Args:
            data: Data dictionary
            field_path: Field path (e.g., 'source.uri')
            
        Returns:
            Value if found, None otherwise
        """
        keys = field_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
