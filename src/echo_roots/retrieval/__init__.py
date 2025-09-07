# T9 Retrieval & Query Interface Implementation

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union, AsyncGenerator, Callable
import logging
from collections import defaultdict
import asyncio
import json

logger = logging.getLogger(__name__)

# Package exports
__all__ = [
    # Core enums
    "QueryType",
    "SortOrder", 
    "FilterOperator",
    "AggregationType",
    
    # Data models
    "QueryFilter",
    "SortCriterion",
    "AggregationRequest",
    "QueryRequest",
    "QueryResult",
    "AggregationResult", 
    "QueryResponse",
    "QueryHistory",
    "FacetConfiguration",
    
    # Abstract interfaces
    "QueryProcessor",
    "RetrievalRepository",
    
    # Core components
    "FilterValidator",
    "QueryOptimizer",
    "ExactMatchProcessor",
    "FuzzySearchProcessor", 
    "SemanticSearchProcessor",
    "QueryEngine",
]


class QueryType(str, Enum):
    """Types of queries supported by the retrieval system."""
    EXACT_MATCH = "exact_match"              # Exact string matching
    FUZZY_SEARCH = "fuzzy_search"           # Fuzzy string matching
    SEMANTIC_SEARCH = "semantic_search"      # Semantic similarity search
    HYBRID_SEARCH = "hybrid_search"         # Combined approach
    FACETED_SEARCH = "faceted_search"       # Multi-faceted filtering
    GRAPH_TRAVERSAL = "graph_traversal"     # Relationship-based search
    FULL_TEXT_SEARCH = "full_text_search"   # Full-text search
    AGGREGATION = "aggregation"             # Statistical queries


class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "ascending"
    DESC = "descending"


class FilterOperator(str, Enum):
    """Filter operation types."""
    EQUALS = "eq"                   # Exact equality
    NOT_EQUALS = "ne"              # Not equal
    CONTAINS = "contains"          # String contains
    NOT_CONTAINS = "not_contains"  # String does not contain
    STARTS_WITH = "starts_with"    # String starts with
    ENDS_WITH = "ends_with"        # String ends with
    IN = "in"                      # Value in list
    NOT_IN = "not_in"             # Value not in list
    GREATER_THAN = "gt"           # Greater than
    GREATER_EQUAL = "gte"         # Greater than or equal
    LESS_THAN = "lt"              # Less than
    LESS_EQUAL = "lte"            # Less than or equal
    BETWEEN = "between"           # Between two values
    IS_NULL = "is_null"           # Is null/empty
    IS_NOT_NULL = "is_not_null"   # Is not null/empty
    REGEX = "regex"               # Regular expression match


class AggregationType(str, Enum):
    """Types of aggregations."""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    DISTINCT_COUNT = "distinct_count"
    GROUP_BY = "group_by"
    PERCENTILE = "percentile"


@dataclass
class QueryFilter:
    """Individual filter criterion."""
    field: str
    operator: FilterOperator
    value: Any
    case_sensitive: bool = False
    weight: float = 1.0  # For weighted filters
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SortCriterion:
    """Sort criterion for query results."""
    field: str
    order: SortOrder = SortOrder.ASC
    weight: float = 1.0  # For weighted sorting
    null_handling: str = "last"  # "first" or "last"


@dataclass
class AggregationRequest:
    """Aggregation specification."""
    aggregation_type: AggregationType
    field: str
    group_by_fields: List[str] = field(default_factory=list)
    filters: List[QueryFilter] = field(default_factory=list)
    having_conditions: List[QueryFilter] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryRequest:
    """Comprehensive query request specification."""
    query_id: str
    query_type: QueryType
    search_text: Optional[str] = None
    target_types: List[str] = field(default_factory=list)  # Entity types to search
    filters: List[QueryFilter] = field(default_factory=list)
    sort_criteria: List[SortCriterion] = field(default_factory=list)
    limit: int = 100
    offset: int = 0
    include_metadata: bool = True
    include_relationships: bool = False
    include_embeddings: bool = False
    similarity_threshold: float = 0.5
    fuzzy_threshold: float = 0.8
    boost_fields: Dict[str, float] = field(default_factory=dict)  # Field boosting
    aggregations: List[AggregationRequest] = field(default_factory=list)
    timeout_seconds: int = 30
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Individual query result item."""
    entity_id: str
    entity_type: str
    score: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: Dict[str, List[float]] = field(default_factory=dict)
    highlights: Dict[str, List[str]] = field(default_factory=dict)  # Search highlights
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of an aggregation query."""
    aggregation_type: AggregationType
    field: str
    value: Any
    group_values: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResponse:
    """Complete query response."""
    query_id: str
    total_results: int
    returned_results: int
    execution_time_ms: float
    results: List[QueryResult]
    aggregations: List[AggregationResult] = field(default_factory=list)
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)  # Facet counts
    suggestions: List[str] = field(default_factory=list)  # Query suggestions
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QueryHistory:
    """Query execution history for analytics."""
    query_id: str
    request: QueryRequest
    response: QueryResponse
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    executed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FacetConfiguration:
    """Configuration for faceted search."""
    field: str
    display_name: str
    facet_type: str = "terms"  # terms, range, date_histogram
    size: int = 10
    min_count: int = 1
    sort_order: str = "count_desc"  # count_desc, count_asc, key_desc, key_asc
    ranges: List[Dict[str, Any]] = field(default_factory=list)  # For range facets
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryProcessor(ABC):
    """Abstract base class for query processors."""
    
    @abstractmethod
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query request and return results."""
        pass
    
    @abstractmethod
    async def validate_query(self, request: QueryRequest) -> List[str]:
        """Validate query request and return validation errors."""
        pass
    
    @abstractmethod
    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if processor supports the given query type."""
        pass


class RetrievalRepository(ABC):
    """Abstract repository for retrieval operations."""
    
    @abstractmethod
    async def search_entities(
        self, 
        query: str, 
        entity_types: List[str] = None,
        filters: List[QueryFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[QueryResult]:
        """Search for entities matching the query."""
        pass
    
    @abstractmethod
    async def get_entity_by_id(self, entity_id: str, include_relationships: bool = False) -> Optional[QueryResult]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_entities_by_ids(self, entity_ids: List[str]) -> List[QueryResult]:
        """Get multiple entities by IDs."""
        pass
    
    @abstractmethod
    async def aggregate_data(self, aggregation: AggregationRequest) -> AggregationResult:
        """Perform data aggregation."""
        pass
    
    @abstractmethod
    async def get_facets(
        self, 
        facet_configs: List[FacetConfiguration],
        base_filters: List[QueryFilter] = None
    ) -> Dict[str, Dict[str, int]]:
        """Get facet counts for search results."""
        pass


class FilterValidator:
    """Validates and normalizes query filters."""
    
    def __init__(self):
        self.valid_operators = set(FilterOperator)
        self.field_type_mapping = {}  # Field name -> expected type
    
    def validate_filters(self, filters: List[QueryFilter]) -> List[str]:
        """Validate a list of filters and return validation errors."""
        errors = []
        
        for i, filter_item in enumerate(filters):
            filter_errors = self.validate_filter(filter_item)
            for error in filter_errors:
                errors.append(f"Filter {i}: {error}")
        
        return errors
    
    def validate_filter(self, filter_item: QueryFilter) -> List[str]:
        """Validate a single filter."""
        errors = []
        
        # Validate field name
        if not filter_item.field or not isinstance(filter_item.field, str):
            errors.append("Field name must be a non-empty string")
        
        # Validate operator
        if filter_item.operator not in self.valid_operators:
            errors.append(f"Invalid operator: {filter_item.operator}")
        
        # Validate value based on operator
        if filter_item.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            # These operators don't need a value
            pass
        elif filter_item.operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(filter_item.value, (list, tuple, set)):
                errors.append(f"Operator {filter_item.operator} requires a list value")
        elif filter_item.operator == FilterOperator.BETWEEN:
            if not isinstance(filter_item.value, (list, tuple)) or len(filter_item.value) != 2:
                errors.append("BETWEEN operator requires a list/tuple with exactly 2 values")
        elif filter_item.operator == FilterOperator.REGEX:
            if not isinstance(filter_item.value, str):
                errors.append("REGEX operator requires a string value")
        else:
            if filter_item.value is None:
                errors.append(f"Operator {filter_item.operator} requires a non-null value")
        
        return errors
    
    def normalize_filter(self, filter_item: QueryFilter) -> QueryFilter:
        """Normalize filter values and formats."""
        normalized = QueryFilter(
            field=filter_item.field.strip().lower() if isinstance(filter_item.field, str) else filter_item.field,
            operator=filter_item.operator,
            value=filter_item.value,
            case_sensitive=filter_item.case_sensitive,
            weight=max(0.0, min(10.0, filter_item.weight)),  # Clamp weight between 0 and 10
            metadata=filter_item.metadata
        )
        
        # Normalize string values for case-insensitive operations
        if not normalized.case_sensitive and isinstance(normalized.value, str):
            normalized.value = normalized.value.lower()
        elif not normalized.case_sensitive and isinstance(normalized.value, (list, tuple)):
            normalized.value = [v.lower() if isinstance(v, str) else v for v in normalized.value]
        
        return normalized


class QueryOptimizer:
    """Optimizes query requests for better performance."""
    
    def __init__(self):
        self.optimization_rules = []
    
    def optimize_query(self, request: QueryRequest) -> QueryRequest:
        """Optimize query request for better performance."""
        optimized = self._deep_copy_request(request)
        
        # Apply optimization rules
        optimized = self._optimize_filters(optimized)
        optimized = self._optimize_sorting(optimized)
        optimized = self._optimize_aggregations(optimized)
        optimized = self._optimize_limits(optimized)
        
        return optimized
    
    def _deep_copy_request(self, request: QueryRequest) -> QueryRequest:
        """Create a deep copy of the query request."""
        return QueryRequest(
            query_id=request.query_id,
            query_type=request.query_type,
            search_text=request.search_text,
            target_types=request.target_types.copy(),
            filters=[
                QueryFilter(
                    field=f.field,
                    operator=f.operator,
                    value=f.value,
                    case_sensitive=f.case_sensitive,
                    weight=f.weight,
                    metadata=f.metadata.copy()
                ) for f in request.filters
            ],
            sort_criteria=[
                SortCriterion(
                    field=s.field,
                    order=s.order,
                    weight=s.weight,
                    null_handling=s.null_handling
                ) for s in request.sort_criteria
            ],
            limit=request.limit,
            offset=request.offset,
            include_metadata=request.include_metadata,
            include_relationships=request.include_relationships,
            include_embeddings=request.include_embeddings,
            similarity_threshold=request.similarity_threshold,
            fuzzy_threshold=request.fuzzy_threshold,
            boost_fields=request.boost_fields.copy(),
            aggregations=[
                AggregationRequest(
                    aggregation_type=a.aggregation_type,
                    field=a.field,
                    group_by_fields=a.group_by_fields.copy(),
                    filters=a.filters.copy(),
                    having_conditions=a.having_conditions.copy(),
                    metadata=a.metadata.copy()
                ) for a in request.aggregations
            ],
            timeout_seconds=request.timeout_seconds,
            metadata=request.metadata.copy()
        )
    
    def _optimize_filters(self, request: QueryRequest) -> QueryRequest:
        """Optimize filter conditions."""
        if not request.filters:
            return request
        
        # Remove duplicate filters
        seen_filters = set()
        unique_filters = []
        
        for filter_item in request.filters:
            filter_key = (filter_item.field, filter_item.operator, str(filter_item.value))
            if filter_key not in seen_filters:
                seen_filters.add(filter_key)
                unique_filters.append(filter_item)
        
        # Sort filters by selectivity (more selective filters first)
        # This is a simple heuristic - in practice, you'd use statistics
        def filter_selectivity(f):
            if f.operator in [FilterOperator.EQUALS, FilterOperator.IN]:
                return 1  # High selectivity
            elif f.operator in [FilterOperator.CONTAINS, FilterOperator.STARTS_WITH]:
                return 2  # Medium selectivity
            else:
                return 3  # Lower selectivity
        
        unique_filters.sort(key=filter_selectivity)
        request.filters = unique_filters
        
        return request
    
    def _optimize_sorting(self, request: QueryRequest) -> QueryRequest:
        """Optimize sort criteria."""
        if not request.sort_criteria:
            return request
        
        # Remove duplicate sort criteria
        seen_fields = set()
        unique_sorts = []
        
        for sort_item in request.sort_criteria:
            if sort_item.field not in seen_fields:
                seen_fields.add(sort_item.field)
                unique_sorts.append(sort_item)
        
        request.sort_criteria = unique_sorts
        return request
    
    def _optimize_aggregations(self, request: QueryRequest) -> QueryRequest:
        """Optimize aggregation requests."""
        # Remove duplicate aggregations
        if not request.aggregations:
            return request
        
        seen_aggs = set()
        unique_aggs = []
        
        for agg in request.aggregations:
            agg_key = (agg.aggregation_type, agg.field, tuple(agg.group_by_fields))
            if agg_key not in seen_aggs:
                seen_aggs.add(agg_key)
                unique_aggs.append(agg)
        
        request.aggregations = unique_aggs
        return request
    
    def _optimize_limits(self, request: QueryRequest) -> QueryRequest:
        """Optimize limit and offset values."""
        # Ensure reasonable limits
        request.limit = max(1, min(1000, request.limit))  # Between 1 and 1000
        request.offset = max(0, request.offset)  # Non-negative
        
        # Adjust timeout based on complexity
        base_timeout = 30
        if len(request.filters) > 10:
            base_timeout += 10
        if len(request.aggregations) > 5:
            base_timeout += 15
        if request.include_embeddings:
            base_timeout += 20
        
        request.timeout_seconds = min(request.timeout_seconds, base_timeout)
        
        return request


class ExactMatchProcessor(QueryProcessor):
    """Processor for exact match queries."""
    
    def __init__(self, repository: RetrievalRepository):
        self.repository = repository
        self.filter_validator = FilterValidator()
    
    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if processor supports the given query type."""
        return query_type == QueryType.EXACT_MATCH
    
    async def validate_query(self, request: QueryRequest) -> List[str]:
        """Validate exact match query request."""
        errors = []
        
        if not request.search_text:
            errors.append("Exact match queries require search text")
        
        # Validate filters
        filter_errors = self.filter_validator.validate_filters(request.filters)
        errors.extend(filter_errors)
        
        return errors
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process exact match query."""
        start_time = datetime.now(UTC)
        errors = await self.validate_query(request)
        
        if errors:
            return QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=0,
                results=[],
                errors=errors
            )
        
        try:
            # Normalize filters
            normalized_filters = [
                self.filter_validator.normalize_filter(f) for f in request.filters
            ]
            
            # Search for exact matches
            results = await self.repository.search_entities(
                query=request.search_text,
                entity_types=request.target_types,
                filters=normalized_filters,
                limit=request.limit,
                offset=request.offset
            )
            
            # Filter results for exact matches only
            exact_results = []
            for result in results:
                if self._is_exact_match(request.search_text, result):
                    exact_results.append(result)
            
            # Apply sorting if specified
            if request.sort_criteria:
                exact_results = self._apply_sorting(exact_results, request.sort_criteria)
            
            # Calculate execution time
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            return QueryResponse(
                query_id=request.query_id,
                total_results=len(exact_results),
                returned_results=len(exact_results),
                execution_time_ms=execution_time,
                results=exact_results
            )
            
        except Exception as e:
            logger.error(f"Error processing exact match query {request.query_id}: {e}")
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            return QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=execution_time,
                results=[],
                errors=[f"Query processing error: {str(e)}"]
            )
    
    def _is_exact_match(self, query_text: str, result: QueryResult) -> bool:
        """Check if result is an exact match for the query."""
        query_lower = query_text.lower()
        
        # Check in main data fields
        for field_value in result.data.values():
            if isinstance(field_value, str) and field_value.lower() == query_lower:
                return True
        
        return False
    
    def _apply_sorting(self, results: List[QueryResult], sort_criteria: List[SortCriterion]) -> List[QueryResult]:
        """Apply sorting to results."""
        def sort_key(result):
            keys = []
            for criterion in sort_criteria:
                value = result.data.get(criterion.field)
                if value is None:
                    value = "" if criterion.null_handling == "first" else "zzz"
                keys.append(value)
            return keys
        
        # Apply sorting
        for criterion in reversed(sort_criteria):
            reverse = criterion.order == SortOrder.DESC
            results.sort(key=lambda r: r.data.get(criterion.field, ""), reverse=reverse)
        
        return results


class FuzzySearchProcessor(QueryProcessor):
    """Processor for fuzzy search queries using Levenshtein distance."""
    
    def __init__(self, repository: RetrievalRepository):
        self.repository = repository
        self.filter_validator = FilterValidator()
    
    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if processor supports the given query type."""
        return query_type == QueryType.FUZZY_SEARCH
    
    async def validate_query(self, request: QueryRequest) -> List[str]:
        """Validate fuzzy search query request."""
        errors = []
        
        if not request.search_text:
            errors.append("Fuzzy search queries require search text")
        
        if request.fuzzy_threshold < 0 or request.fuzzy_threshold > 1:
            errors.append("Fuzzy threshold must be between 0 and 1")
        
        # Validate filters
        filter_errors = self.filter_validator.validate_filters(request.filters)
        errors.extend(filter_errors)
        
        return errors
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process fuzzy search query."""
        start_time = datetime.now(UTC)
        errors = await self.validate_query(request)
        
        if errors:
            return QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=0,
                results=[],
                errors=errors
            )
        
        try:
            # Normalize filters
            normalized_filters = [
                self.filter_validator.normalize_filter(f) for f in request.filters
            ]
            
            # Get broader search results
            broad_results = await self.repository.search_entities(
                query=request.search_text,
                entity_types=request.target_types,
                filters=normalized_filters,
                limit=request.limit * 3,  # Get more results for fuzzy matching
                offset=request.offset
            )
            
            # Calculate fuzzy scores and filter
            fuzzy_results = []
            for result in broad_results:
                fuzzy_score = self._calculate_fuzzy_score(request.search_text, result)
                if fuzzy_score >= request.fuzzy_threshold:
                    result.score = fuzzy_score
                    result.explanation = f"Fuzzy match (score: {fuzzy_score:.3f})"
                    fuzzy_results.append(result)
            
            # Sort by fuzzy score
            fuzzy_results.sort(key=lambda r: r.score, reverse=True)
            
            # Apply limit
            final_results = fuzzy_results[:request.limit]
            
            # Calculate execution time
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            return QueryResponse(
                query_id=request.query_id,
                total_results=len(fuzzy_results),
                returned_results=len(final_results),
                execution_time_ms=execution_time,
                results=final_results
            )
            
        except Exception as e:
            logger.error(f"Error processing fuzzy search query {request.query_id}: {e}")
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            return QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=execution_time,
                results=[],
                errors=[f"Query processing error: {str(e)}"]
            )
    
    def _calculate_fuzzy_score(self, query_text: str, result: QueryResult) -> float:
        """Calculate fuzzy match score using Levenshtein distance."""
        query_lower = query_text.lower()
        best_score = 0.0
        
        # Check against all string fields in result data
        for field_value in result.data.values():
            if isinstance(field_value, str):
                score = self._levenshtein_similarity(query_lower, field_value.lower())
                best_score = max(best_score, score)
        
        return best_score
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity using Levenshtein distance."""
        if len(s1) == 0:
            return 0.0 if len(s2) > 0 else 1.0
        if len(s2) == 0:
            return 0.0
        
        # Create distance matrix
        rows = len(s1) + 1
        cols = len(s2) + 1
        distance = [[0] * cols for _ in range(rows)]
        
        # Initialize first row and column
        for i in range(1, rows):
            distance[i][0] = i
        for j in range(1, cols):
            distance[0][j] = j
        
        # Fill the matrix
        for i in range(1, rows):
            for j in range(1, cols):
                if s1[i-1] == s2[j-1]:
                    cost = 0
                else:
                    cost = 1
                
                distance[i][j] = min(
                    distance[i-1][j] + 1,      # deletion
                    distance[i][j-1] + 1,      # insertion
                    distance[i-1][j-1] + cost   # substitution
                )
        
        # Convert distance to similarity
        max_len = max(len(s1), len(s2))
        similarity = (max_len - distance[rows-1][cols-1]) / max_len
        return similarity


class SemanticSearchProcessor(QueryProcessor):
    """Processor for semantic search queries using embeddings."""
    
    def __init__(self, repository: RetrievalRepository, semantic_engine=None):
        self.repository = repository
        self.semantic_engine = semantic_engine
        self.filter_validator = FilterValidator()
    
    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if processor supports the given query type."""
        return query_type == QueryType.SEMANTIC_SEARCH
    
    async def validate_query(self, request: QueryRequest) -> List[str]:
        """Validate semantic search query request."""
        errors = []
        
        if not request.search_text:
            errors.append("Semantic search queries require search text")
        
        if not self.semantic_engine:
            errors.append("Semantic engine not available for semantic search")
        
        if request.similarity_threshold < 0 or request.similarity_threshold > 1:
            errors.append("Similarity threshold must be between 0 and 1")
        
        # Validate filters
        filter_errors = self.filter_validator.validate_filters(request.filters)
        errors.extend(filter_errors)
        
        return errors
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process semantic search query."""
        start_time = datetime.now(UTC)
        errors = await self.validate_query(request)
        
        if errors:
            return QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=0,
                results=[],
                errors=errors
            )
        
        try:
            # Generate query embedding
            query_embedding = await self.semantic_engine.generate_embedding(request.search_text)
            
            # Normalize filters
            normalized_filters = [
                self.filter_validator.normalize_filter(f) for f in request.filters
            ]
            
            # Get candidate results
            candidates = await self.repository.search_entities(
                query=request.search_text,
                entity_types=request.target_types,
                filters=normalized_filters,
                limit=request.limit * 5,  # Get more candidates for semantic ranking
                offset=request.offset
            )
            
            # Calculate semantic similarity scores
            semantic_results = []
            for result in candidates:
                # Get or generate embedding for result
                result_embedding = await self._get_result_embedding(result)
                if result_embedding:
                    similarity = self._cosine_similarity(query_embedding, result_embedding)
                    if similarity >= request.similarity_threshold:
                        result.score = similarity
                        result.explanation = f"Semantic similarity: {similarity:.3f}"
                        semantic_results.append(result)
            
            # Sort by similarity score
            semantic_results.sort(key=lambda r: r.score, reverse=True)
            
            # Apply limit
            final_results = semantic_results[:request.limit]
            
            # Calculate execution time
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            return QueryResponse(
                query_id=request.query_id,
                total_results=len(semantic_results),
                returned_results=len(final_results),
                execution_time_ms=execution_time,
                results=final_results
            )
            
        except Exception as e:
            logger.error(f"Error processing semantic search query {request.query_id}: {e}")
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            return QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=execution_time,
                results=[],
                errors=[f"Query processing error: {str(e)}"]
            )
    
    async def _get_result_embedding(self, result: QueryResult) -> Optional[List[float]]:
        """Get or generate embedding for a result."""
        # Check if embedding is already included
        if result.embeddings and 'text' in result.embeddings:
            return result.embeddings['text']
        
        # Generate embedding from result data
        if self.semantic_engine:
            # Combine relevant text fields
            text_content = []
            for key, value in result.data.items():
                if isinstance(value, str) and value.strip():
                    text_content.append(f"{key}: {value}")
            
            if text_content:
                combined_text = " ".join(text_content)
                return await self.semantic_engine.generate_embedding(combined_text)
        
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class QueryEngine:
    """Main query engine that orchestrates different processors."""
    
    def __init__(self, repository: RetrievalRepository):
        self.repository = repository
        self.processors: Dict[QueryType, QueryProcessor] = {}
        self.query_optimizer = QueryOptimizer()
        self.query_history: List[QueryHistory] = []
        
        # Register default processors
        self.register_processor(QueryType.EXACT_MATCH, ExactMatchProcessor(repository))
        self.register_processor(QueryType.FUZZY_SEARCH, FuzzySearchProcessor(repository))
        
        # Semantic processor will be registered when semantic engine is available
        self._semantic_engine = None
    
    def register_processor(self, query_type: QueryType, processor: QueryProcessor):
        """Register a query processor for a specific query type."""
        self.processors[query_type] = processor
        logger.info(f"Registered processor for query type: {query_type}")
    
    def set_semantic_engine(self, semantic_engine):
        """Set semantic engine and register semantic processor."""
        self._semantic_engine = semantic_engine
        self.register_processor(
            QueryType.SEMANTIC_SEARCH, 
            SemanticSearchProcessor(self.repository, semantic_engine)
        )
        logger.info("Semantic search processor registered")
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process a query request using the appropriate processor."""
        start_time = datetime.now(UTC)
        
        try:
            # Optimize query
            optimized_request = self.query_optimizer.optimize_query(request)
            
            # Get appropriate processor
            processor = self.processors.get(optimized_request.query_type)
            if not processor:
                return QueryResponse(
                    query_id=request.query_id,
                    total_results=0,
                    returned_results=0,
                    execution_time_ms=0,
                    results=[],
                    errors=[f"No processor available for query type: {optimized_request.query_type}"]
                )
            
            # Process query
            response = await processor.process_query(optimized_request)
            
            # Add to history
            history = QueryHistory(
                query_id=request.query_id,
                request=request,
                response=response,
                success=len(response.errors) == 0,
                error_message="; ".join(response.errors) if response.errors else None
            )
            self.query_history.append(history)
            
            # Limit history size
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-1000:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error in query engine for request {request.query_id}: {e}")
            execution_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            error_response = QueryResponse(
                query_id=request.query_id,
                total_results=0,
                returned_results=0,
                execution_time_ms=execution_time,
                results=[],
                errors=[f"Query engine error: {str(e)}"]
            )
            
            # Add error to history
            history = QueryHistory(
                query_id=request.query_id,
                request=request,
                response=error_response,
                success=False,
                error_message=str(e)
            )
            self.query_history.append(history)
            
            return error_response
    
    async def validate_query(self, request: QueryRequest) -> List[str]:
        """Validate a query request."""
        errors = []
        
        # Basic validation
        if not request.query_id:
            errors.append("Query ID is required")
        
        if not request.query_type:
            errors.append("Query type is required")
        
        # Get processor for validation
        processor = self.processors.get(request.query_type)
        if not processor:
            errors.append(f"No processor available for query type: {request.query_type}")
        else:
            # Use processor-specific validation
            processor_errors = await processor.validate_query(request)
            errors.extend(processor_errors)
        
        return errors
    
    def get_supported_query_types(self) -> List[QueryType]:
        """Get list of supported query types."""
        return list(self.processors.keys())
    
    def get_query_history(
        self, 
        limit: int = 100, 
        query_type: Optional[QueryType] = None,
        success_only: bool = False
    ) -> List[QueryHistory]:
        """Get query execution history."""
        filtered_history = self.query_history
        
        # Filter by query type
        if query_type:
            filtered_history = [h for h in filtered_history if h.request.query_type == query_type]
        
        # Filter by success
        if success_only:
            filtered_history = [h for h in filtered_history if h.success]
        
        # Apply limit
        return filtered_history[-limit:]
    
    async def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on partial input."""
        suggestions = []
        
        # Get suggestions from recent successful queries
        recent_queries = [
            h.request.search_text for h in self.query_history[-100:]
            if h.success and h.request.search_text and partial_query.lower() in h.request.search_text.lower()
        ]
        
        # Remove duplicates and sort by relevance
        unique_suggestions = list(set(recent_queries))
        unique_suggestions.sort(key=lambda s: s.lower().index(partial_query.lower()) if partial_query.lower() in s.lower() else 999)
        
        return unique_suggestions[:limit]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get query performance metrics."""
        if not self.query_history:
            return {}
        
        # Calculate metrics
        total_queries = len(self.query_history)
        successful_queries = sum(1 for h in self.query_history if h.success)
        
        execution_times = [h.response.execution_time_ms for h in self.query_history if h.response.execution_time_ms]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Query type distribution
        query_type_counts = defaultdict(int)
        for history in self.query_history:
            query_type_counts[history.request.query_type] += 1
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "average_execution_time_ms": avg_execution_time,
            "query_type_distribution": dict(query_type_counts),
            "supported_query_types": [qt.value for qt in self.get_supported_query_types()]
        }
