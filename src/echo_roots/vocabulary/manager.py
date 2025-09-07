# T7 Controlled Vocabulary Management (C Layer) Implementation

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class VocabularyType(str, Enum):
    """Types of controlled vocabularies."""
    ATTRIBUTE = "attribute"         # Product attributes (color, size, brand)
    VALUE = "value"                # Attribute values (red, large, Nike)
    SYNONYM = "synonym"            # Synonyms and aliases
    UNIT = "unit"                  # Units of measurement (kg, cm, USD)
    RELATIONSHIP = "relationship"   # Inter-attribute relationships


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"       # Must match exactly
    MODERATE = "moderate"   # Allow normalized matches
    FLEXIBLE = "flexible"   # Allow fuzzy matches
    PERMISSIVE = "permissive"  # Allow any value with warnings


class MappingConfidence(str, Enum):
    """Confidence levels for vocabulary mappings."""
    EXACT = "exact"         # 1.0 - Perfect match
    HIGH = "high"           # 0.8-0.99 - Very confident
    MEDIUM = "medium"       # 0.6-0.79 - Moderately confident
    LOW = "low"             # 0.4-0.59 - Uncertain
    POOR = "poor"           # 0.0-0.39 - Very uncertain


@dataclass
class VocabularyTerm:
    """A controlled vocabulary term with metadata."""
    term_id: str
    term: str
    vocabulary_type: VocabularyType
    category_id: Optional[str] = None      # Associated category
    parent_term_id: Optional[str] = None   # For hierarchical terms
    aliases: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)  # Multi-language
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    confidence_score: float = 1.0
    usage_count: int = 0
    domain: Optional[str] = None


@dataclass
class VocabularyMapping:
    """Mapping between raw input and controlled vocabulary."""
    mapping_id: str
    raw_value: str
    mapped_term_id: str
    confidence: MappingConfidence
    confidence_score: float
    mapping_type: str                       # exact, fuzzy, semantic, rules
    context: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = "pending"      # pending, approved, rejected
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of vocabulary validation."""
    is_valid: bool
    term_id: Optional[str] = None
    mapped_value: Optional[str] = None
    confidence_score: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STRICT
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VocabularyRequest:
    """Request to create or update vocabulary terms."""
    term: str
    vocabulary_type: VocabularyType
    category_id: Optional[str] = None
    parent_term_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None


@dataclass
class VocabularyStats:
    """Statistics about vocabulary coverage and quality."""
    total_terms: int = 0
    terms_by_type: Dict[VocabularyType, int] = field(default_factory=dict)
    terms_by_category: Dict[str, int] = field(default_factory=dict)
    coverage_rate: float = 0.0              # % of mappings successfully resolved
    confidence_distribution: Dict[MappingConfidence, int] = field(default_factory=dict)
    validation_success_rate: float = 0.0
    most_common_terms: List[Tuple[str, int]] = field(default_factory=list)
    orphaned_terms: int = 0                 # Terms without category
    outdated_terms: int = 0                 # Terms not used recently
    domain_coverage: Dict[str, float] = field(default_factory=dict)


class VocabularyRepository(ABC):
    """Abstract repository for vocabulary storage."""
    
    @abstractmethod
    async def store_term(self, term: VocabularyTerm) -> str:
        """Store a vocabulary term."""
        pass
    
    @abstractmethod
    async def get_term(self, term_id: str) -> Optional[VocabularyTerm]:
        """Retrieve a vocabulary term by ID."""
        pass
    
    @abstractmethod
    async def find_terms(
        self,
        vocabulary_type: Optional[VocabularyType] = None,
        category_id: Optional[str] = None,
        domain: Optional[str] = None,
        is_active: bool = True
    ) -> List[VocabularyTerm]:
        """Find vocabulary terms by criteria."""
        pass
    
    @abstractmethod
    async def search_terms(self, query: str, limit: int = 50) -> List[VocabularyTerm]:
        """Search vocabulary terms by text."""
        pass
    
    @abstractmethod
    async def store_mapping(self, mapping: VocabularyMapping) -> str:
        """Store a vocabulary mapping."""
        pass
    
    @abstractmethod
    async def get_mapping(self, mapping_id: str) -> Optional[VocabularyMapping]:
        """Retrieve a vocabulary mapping by ID."""
        pass
    
    @abstractmethod
    async def find_mappings(
        self,
        raw_value: Optional[str] = None,
        term_id: Optional[str] = None,
        confidence: Optional[MappingConfidence] = None
    ) -> List[VocabularyMapping]:
        """Find vocabulary mappings by criteria."""
        pass
    
    @abstractmethod
    async def update_term(self, term: VocabularyTerm) -> bool:
        """Update a vocabulary term."""
        pass
    
    @abstractmethod
    async def delete_term(self, term_id: str) -> bool:
        """Delete a vocabulary term."""
        pass


class VocabularyNormalizer:
    """Normalizes vocabulary values for consistent matching."""
    
    def __init__(self):
        self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.unit_patterns = {
            'weight': re.compile(r'\b(\d+(?:\.\d+)?)\s*(kg|g|lb|oz|pound|gram|kilogram)\b', re.IGNORECASE),
            'length': re.compile(r'\b(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|ft|foot|yard)\b', re.IGNORECASE),
            'volume': re.compile(r'\b(\d+(?:\.\d+)?)\s*(ml|l|liter|oz|gallon|cup)\b', re.IGNORECASE)
        }
    
    def normalize_term(self, term: str) -> str:
        """Normalize a term for consistent matching."""
        if not term:
            return ""
        
        # Convert to lowercase
        normalized = term.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations BEFORE removing special characters
        normalized = self._expand_abbreviations(normalized)
        
        # Replace underscores with spaces
        normalized = normalized.replace('_', ' ')
        
        # Remove special characters (keep alphanumeric, spaces, hyphens)
        normalized = re.sub(r'[^a-z0-9\s\-]', ' ', normalized)
        
        # Clean up multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def extract_units(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract units and measurements from text."""
        units = defaultdict(list)
        
        for unit_type, pattern in self.unit_patterns.items():
            matches = pattern.findall(text)
            for value, unit in matches:
                units[unit_type].append((value, unit))
        
        return dict(units)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        abbreviations = {
            r'w/(?=\s|$)': 'with',    # w/ followed by space or end
            r'w/o(?=\s|$)': 'without', # w/o followed by space or end
            r'\s&\s': ' and ',        # & surrounded by spaces
            r'\s\+\s': ' plus ',      # + surrounded by spaces  
            r'\bsz\b': 'size',
            r'\bclr\b': 'color',
            r'\bqty\b': 'quantity'
        }
        
        for abbrev, expansion in abbreviations.items():
            text = re.sub(abbrev, expansion, text)
        
        return text


class VocabularyMatcher:
    """Matches raw input to controlled vocabulary terms."""
    
    def __init__(self, normalizer: VocabularyNormalizer):
        self.normalizer = normalizer
        self._term_cache: Dict[str, List[VocabularyTerm]] = {}
        self._mapping_cache: Dict[str, VocabularyMapping] = {}
    
    async def find_best_match(
        self,
        raw_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.MODERATE
    ) -> Optional[Tuple[VocabularyTerm, float]]:
        """Find the best matching vocabulary term."""
        if not raw_value:
            return None
        
        # Try cache first
        cache_key = f"{raw_value}:{vocabulary_type}:{category_id}"
        if cache_key in self._mapping_cache:
            mapping = self._mapping_cache[cache_key]
            term = await self._get_term_by_id(mapping.mapped_term_id)
            if term:
                return term, mapping.confidence_score
        
        normalized_value = self.normalizer.normalize_term(raw_value)
        
        # Try different matching strategies
        matches = []
        
        # 1. Exact match
        exact_match = await self._find_exact_match(normalized_value, vocabulary_type, category_id)
        if exact_match:
            matches.append((exact_match, 1.0, "exact"))
        
        # 2. Alias/synonym match
        alias_match = await self._find_alias_match(normalized_value, vocabulary_type, category_id)
        if alias_match:
            matches.append((alias_match, 0.95, "alias"))
        
        # 3. Fuzzy match (if not strict)
        if validation_level in [ValidationLevel.MODERATE, ValidationLevel.FLEXIBLE, ValidationLevel.PERMISSIVE]:
            fuzzy_matches = await self._find_fuzzy_matches(normalized_value, vocabulary_type, category_id)
            matches.extend(fuzzy_matches)
        
        # 4. Semantic match (if flexible/permissive)
        if validation_level in [ValidationLevel.FLEXIBLE, ValidationLevel.PERMISSIVE]:
            semantic_matches = await self._find_semantic_matches(normalized_value, vocabulary_type, category_id)
            matches.extend(semantic_matches)
        
        if not matches:
            return None
        
        # Return best match
        best_match = max(matches, key=lambda x: x[1])
        
        # Cache the result
        mapping = VocabularyMapping(
            mapping_id=f"cache_{raw_value}_{vocabulary_type}",
            raw_value=raw_value,
            mapped_term_id=best_match[0].term_id,
            confidence=self._score_to_confidence(best_match[1]),
            confidence_score=best_match[1],
            mapping_type=best_match[2]
        )
        self._mapping_cache[cache_key] = mapping
        
        return best_match[0], best_match[1]
    
    async def _find_exact_match(
        self,
        normalized_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> Optional[VocabularyTerm]:
        """Find exact normalized match."""
        # This would search the repository for exact matches
        # Implementation depends on the repository
        return None
    
    async def _find_alias_match(
        self,
        normalized_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> Optional[VocabularyTerm]:
        """Find match in aliases or synonyms."""
        # This would search aliases and synonyms
        return None
    
    async def _find_fuzzy_matches(
        self,
        normalized_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> List[Tuple[VocabularyTerm, float, str]]:
        """Find fuzzy string matches."""
        # This would use string similarity algorithms
        return []
    
    async def _find_semantic_matches(
        self,
        normalized_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> List[Tuple[VocabularyTerm, float, str]]:
        """Find semantic/embedding matches."""
        # This would use embeddings for semantic similarity
        return []
    
    async def _get_term_by_id(self, term_id: str) -> Optional[VocabularyTerm]:
        """Get term by ID (placeholder for repository call)."""
        return None
    
    def _score_to_confidence(self, score: float) -> MappingConfidence:
        """Convert numeric score to confidence level."""
        if score >= 0.95:
            return MappingConfidence.EXACT
        elif score >= 0.8:
            return MappingConfidence.HIGH
        elif score >= 0.6:
            return MappingConfidence.MEDIUM
        elif score >= 0.4:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.POOR


class VocabularyValidator:
    """Validates vocabulary values against controlled vocabularies."""
    
    def __init__(self, matcher: VocabularyMatcher):
        self.matcher = matcher
    
    async def validate_value(
        self,
        raw_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.MODERATE
    ) -> ValidationResult:
        """Validate a raw value against controlled vocabulary."""
        result = ValidationResult(
            is_valid=False,
            validation_level=validation_level
        )
        
        if not raw_value or not raw_value.strip():
            result.errors.append("Empty value provided")
            return result
        
        try:
            # Find best match
            match = await self.matcher.find_best_match(
                raw_value, vocabulary_type, category_id, validation_level
            )
            
            if match:
                term, confidence = match
                result.is_valid = True
                result.term_id = term.term_id
                result.mapped_value = term.term
                result.confidence_score = confidence
                
                # Add warnings based on confidence
                if confidence < 0.8:
                    result.warnings.append(f"Low confidence match: {confidence:.2f}")
                
                if confidence < 0.6 and validation_level == ValidationLevel.STRICT:
                    result.is_valid = False
                    result.errors.append("Match confidence too low for strict validation")
            
            else:
                # No match found
                result.errors.append(f"No matching term found for '{raw_value}'")
                
                # Generate suggestions for permissive mode
                if validation_level == ValidationLevel.PERMISSIVE:
                    result.is_valid = True
                    result.mapped_value = raw_value
                    result.warnings.append("Using raw value - no controlled vocabulary match")
                    
                    # Add suggestions for similar terms
                    suggestions = await self._generate_suggestions(raw_value, vocabulary_type, category_id)
                    result.suggestions.extend(suggestions)
        
        except Exception as e:
            logger.error(f"Error validating value '{raw_value}': {e}")
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    async def _generate_suggestions(
        self,
        raw_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> List[str]:
        """Generate suggestions for unmatched values."""
        # This would generate suggestions based on similar terms
        return []


class VocabularyManager:
    """High-level manager for controlled vocabulary operations."""
    
    def __init__(self, repository: VocabularyRepository):
        self.repository = repository
        self.normalizer = VocabularyNormalizer()
        self.matcher = VocabularyMatcher(self.normalizer)
        self.validator = VocabularyValidator(self.matcher)
        self._stats_cache: Optional[VocabularyStats] = None
        self._cache_expiry: Optional[datetime] = None
    
    async def create_term(self, request: VocabularyRequest) -> VocabularyTerm:
        """Create a new vocabulary term."""
        # Validate request
        await self._validate_term_request(request)
        
        # Normalize term
        normalized_term = self.normalizer.normalize_term(request.term)
        
        # Check for duplicates
        existing = await self._find_duplicate_term(normalized_term, request.vocabulary_type, request.category_id)
        if existing:
            raise ValueError(f"Term '{request.term}' already exists as '{existing.term}'")
        
        # Create term
        term = VocabularyTerm(
            term_id=self._generate_term_id(),
            term=request.term,
            vocabulary_type=request.vocabulary_type,
            category_id=request.category_id,
            parent_term_id=request.parent_term_id,
            aliases=request.aliases,
            synonyms=request.synonyms,
            description=request.description,
            labels=request.labels,
            metadata=request.metadata,
            validation_rules=request.validation_rules,
            domain=request.domain
        )
        
        # Store term
        term_id = await self.repository.store_term(term)
        term.term_id = term_id
        
        # Clear cache
        self._clear_cache()
        
        logger.info(f"Created vocabulary term: {term.term} ({term.vocabulary_type})")
        return term
    
    async def validate_value(
        self,
        raw_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.MODERATE
    ) -> ValidationResult:
        """Validate a value against controlled vocabulary."""
        return await self.validator.validate_value(
            raw_value, vocabulary_type, category_id, validation_level
        )
    
    async def map_value(
        self,
        raw_value: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str] = None,
        validation_level: ValidationLevel = ValidationLevel.MODERATE,
        store_mapping: bool = True
    ) -> Optional[VocabularyMapping]:
        """Map a raw value to controlled vocabulary."""
        # Validate and find match
        validation_result = await self.validate_value(
            raw_value, vocabulary_type, category_id, validation_level
        )
        
        if not validation_result.is_valid and validation_level != ValidationLevel.PERMISSIVE:
            return None
        
        # Create mapping
        mapping = VocabularyMapping(
            mapping_id=self._generate_mapping_id(),
            raw_value=raw_value,
            mapped_term_id=validation_result.term_id or "",
            confidence=self._score_to_confidence(validation_result.confidence_score),
            confidence_score=validation_result.confidence_score,
            mapping_type="system",
            context={
                "vocabulary_type": vocabulary_type,
                "category_id": category_id,
                "validation_level": validation_level
            },
            metadata=validation_result.metadata
        )
        
        # Store mapping if requested
        if store_mapping:
            mapping_id = await self.repository.store_mapping(mapping)
            mapping.mapping_id = mapping_id
        
        return mapping
    
    async def get_vocabulary_stats(self, force_refresh: bool = False) -> VocabularyStats:
        """Get vocabulary statistics."""
        if not force_refresh and self._stats_cache and self._cache_expiry and datetime.utcnow() < self._cache_expiry:
            return self._stats_cache
        
        stats = VocabularyStats()
        
        # Get all terms
        all_terms = await self.repository.find_terms()
        stats.total_terms = len(all_terms)
        
        # Count by type
        for term in all_terms:
            stats.terms_by_type[term.vocabulary_type] = stats.terms_by_type.get(term.vocabulary_type, 0) + 1
            if term.category_id:
                stats.terms_by_category[term.category_id] = stats.terms_by_category.get(term.category_id, 0) + 1
            else:
                stats.orphaned_terms += 1
        
        # Get mapping statistics
        all_mappings = await self.repository.find_mappings()
        confidence_counts = defaultdict(int)
        successful_mappings = 0
        
        for mapping in all_mappings:
            confidence_counts[mapping.confidence] += 1
            if mapping.confidence in [MappingConfidence.EXACT, MappingConfidence.HIGH]:
                successful_mappings += 1
        
        stats.confidence_distribution = dict(confidence_counts)
        if all_mappings:
            stats.coverage_rate = successful_mappings / len(all_mappings)
        
        # Cache stats
        self._stats_cache = stats
        self._cache_expiry = datetime.utcnow().replace(hour=datetime.utcnow().hour + 1)  # Cache for 1 hour
        
        return stats
    
    async def _validate_term_request(self, request: VocabularyRequest) -> None:
        """Validate a term creation request."""
        if not request.term or not request.term.strip():
            raise ValueError("Term cannot be empty")
        
        if request.parent_term_id:
            parent = await self.repository.get_term(request.parent_term_id)
            if not parent:
                raise ValueError(f"Parent term '{request.parent_term_id}' not found")
            
            if parent.vocabulary_type != request.vocabulary_type:
                raise ValueError("Parent term must be of the same vocabulary type")
    
    async def _find_duplicate_term(
        self,
        normalized_term: str,
        vocabulary_type: VocabularyType,
        category_id: Optional[str]
    ) -> Optional[VocabularyTerm]:
        """Find duplicate terms."""
        terms = await self.repository.find_terms(vocabulary_type, category_id)
        for term in terms:
            if self.normalizer.normalize_term(term.term) == normalized_term:
                return term
        return None
    
    def _generate_term_id(self) -> str:
        """Generate a unique term ID."""
        import uuid
        return f"term_{uuid.uuid4().hex[:12]}"
    
    def _generate_mapping_id(self) -> str:
        """Generate a unique mapping ID."""
        import uuid
        return f"mapping_{uuid.uuid4().hex[:12]}"
    
    def _score_to_confidence(self, score: float) -> MappingConfidence:
        """Convert numeric score to confidence level."""
        if score >= 0.95:
            return MappingConfidence.EXACT
        elif score >= 0.8:
            return MappingConfidence.HIGH
        elif score >= 0.6:
            return MappingConfidence.MEDIUM
        elif score >= 0.4:
            return MappingConfidence.LOW
        else:
            return MappingConfidence.POOR
    
    def _clear_cache(self) -> None:
        """Clear internal caches."""
        self._stats_cache = None
        self._cache_expiry = None
        self.matcher._term_cache.clear()
        self.matcher._mapping_cache.clear()
