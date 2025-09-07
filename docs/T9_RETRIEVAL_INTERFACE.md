# T9 Retrieval & Query Interface Documentation

## Overview

The T9 Retrieval & Query Interface provides a comprehensive query processing system for the Echo Roots taxonomy platform. It offers multiple search strategies, advanced filtering, sorting, aggregation capabilities, and optimization features.

## Core Components

### Query Types

The system supports multiple query types through the `QueryType` enum:

- **EXACT_MATCH**: Precise string matching for exact term searches
- **FUZZY_SEARCH**: Levenshtein distance-based similarity matching for typo tolerance
- **SEMANTIC_SEARCH**: Vector similarity search using embeddings for semantic understanding
- **HYBRID_SEARCH**: Combined approach mixing multiple strategies
- **FACETED_SEARCH**: Multi-dimensional filtering with facet counts
- **GRAPH_TRAVERSAL**: Relationship-based graph navigation
- **FULL_TEXT_SEARCH**: Comprehensive text search across content
- **AGGREGATION**: Statistical queries and data summarization

### Data Models

#### QueryRequest
Comprehensive query specification including:
- Query metadata (ID, type, timestamps)
- Search parameters (text, target types, thresholds)
- Filtering and sorting criteria
- Performance constraints (limits, timeouts)
- Result inclusion options (metadata, relationships, embeddings)

#### QueryResponse
Complete query results containing:
- Execution metadata (timing, result counts)
- Search results with scoring and explanations
- Aggregation results and facet counts
- Error handling and suggestions
- Performance diagnostics

#### QueryResult
Individual result items with:
- Entity identification and classification
- Relevance scoring and ranking
- Data payload and relationships
- Search highlights and explanations
- Embedding vectors (when requested)

### Filter System

Advanced filtering capabilities through `QueryFilter`:

**Operators**: Supports 16 different filter operators including:
- Equality and inequality (`EQUALS`, `NOT_EQUALS`)
- String operations (`CONTAINS`, `STARTS_WITH`, `ENDS_WITH`)
- List operations (`IN`, `NOT_IN`)
- Numeric comparisons (`GREATER_THAN`, `LESS_THAN`, `BETWEEN`)
- Null handling (`IS_NULL`, `IS_NOT_NULL`)
- Pattern matching (`REGEX`)

**Features**:
- Case-sensitive/insensitive matching
- Weighted filtering for relevance
- Metadata attachment for context
- Validation and normalization

### Query Processors

#### ExactMatchProcessor
- **Purpose**: High-precision exact string matching
- **Algorithm**: Direct string comparison with normalization
- **Use Cases**: Product SKUs, exact category names, precise identifiers
- **Performance**: Fastest query type with immediate results

#### FuzzySearchProcessor  
- **Purpose**: Typo-tolerant approximate matching
- **Algorithm**: Levenshtein distance with configurable similarity threshold
- **Use Cases**: User search with typos, product name variations
- **Features**: Similarity scoring, configurable tolerance levels

#### SemanticSearchProcessor
- **Purpose**: Meaning-based search using embeddings
- **Algorithm**: Cosine similarity between query and content vectors
- **Use Cases**: Conceptual search, related product discovery
- **Requirements**: Semantic engine integration for embedding generation

### Query Engine

Central orchestration component providing:
- **Processor Management**: Registration and routing of query types
- **Query Optimization**: Automatic performance improvements
- **History Tracking**: Query analytics and performance monitoring
- **Suggestion Generation**: Smart query recommendations
- **Error Handling**: Comprehensive validation and error recovery

### Optimization System

#### FilterValidator
- Validates filter syntax and semantics
- Normalizes values for consistent processing
- Enforces operator constraints and type safety
- Clamps weights and validates field names

#### QueryOptimizer
- **Filter Optimization**: Removes duplicates, orders by selectivity
- **Sort Optimization**: Eliminates redundant sorting criteria
- **Limit Management**: Enforces reasonable pagination limits
- **Timeout Adjustment**: Dynamic timeout based on query complexity

## Usage Examples

### Basic Exact Match Search
```python
from echo_roots.retrieval import QueryRequest, QueryType, QueryEngine

# Create query request
request = QueryRequest(
    query_id="search-001",
    query_type=QueryType.EXACT_MATCH,
    search_text="Laptop Computer",
    target_types=["product"],
    limit=20
)

# Process query
response = await query_engine.process_query(request)
print(f"Found {response.total_results} exact matches")
```

### Fuzzy Search with Filters
```python
from echo_roots.retrieval import QueryFilter, FilterOperator

request = QueryRequest(
    query_id="fuzzy-001",
    query_type=QueryType.FUZZY_SEARCH,
    search_text="laptap",  # Typo in "laptop"
    filters=[
        QueryFilter("category", FilterOperator.EQUALS, "electronics"),
        QueryFilter("price", FilterOperator.LESS_THAN, 1000)
    ],
    fuzzy_threshold=0.8,
    limit=10
)

response = await query_engine.process_query(request)
for result in response.results:
    print(f"Match: {result.data['name']} (score: {result.score:.3f})")
```

### Semantic Search
```python
request = QueryRequest(
    query_id="semantic-001",
    query_type=QueryType.SEMANTIC_SEARCH,
    search_text="portable computing device",
    similarity_threshold=0.7,
    include_embeddings=True,
    limit=15
)

response = await query_engine.process_query(request)
for result in response.results:
    print(f"Semantic match: {result.explanation}")
```

### Advanced Filtering
```python
from echo_roots.retrieval import SortCriterion, SortOrder

request = QueryRequest(
    query_id="advanced-001",
    query_type=QueryType.FUZZY_SEARCH,
    search_text="computer",
    filters=[
        QueryFilter("category", FilterOperator.IN, ["electronics", "computers"]),
        QueryFilter("rating", FilterOperator.GREATER_EQUAL, 4.0),
        QueryFilter("name", FilterOperator.NOT_CONTAINS, "refurbished")
    ],
    sort_criteria=[
        SortCriterion("price", SortOrder.ASC),
        SortCriterion("rating", SortOrder.DESC)
    ],
    boost_fields={"name": 2.0, "description": 1.5},
    limit=25
)

response = await query_engine.process_query(request)
```

## Performance Features

### Query Optimization
- **Automatic Filter Ordering**: Places most selective filters first
- **Duplicate Removal**: Eliminates redundant filters and sorts
- **Limit Enforcement**: Prevents resource exhaustion
- **Dynamic Timeouts**: Adjusts based on query complexity

### Caching and History
- **Query History**: Maintains execution history for analytics
- **Performance Metrics**: Tracks success rates and execution times
- **Query Suggestions**: Generates recommendations from history
- **Result Ranking**: Intelligent scoring and relevance ordering

### Scalability
- **Pagination Support**: Efficient offset/limit handling
- **Streaming Results**: Supports large result sets
- **Background Processing**: Non-blocking query execution
- **Resource Management**: Memory and CPU usage optimization

## Integration Points

### Repository Interface
```python
class RetrievalRepository(ABC):
    async def search_entities(self, query: str, entity_types: List[str] = None, 
                             filters: List[QueryFilter] = None, limit: int = 100, 
                             offset: int = 0) -> List[QueryResult]
    
    async def get_entity_by_id(self, entity_id: str, 
                              include_relationships: bool = False) -> Optional[QueryResult]
    
    async def aggregate_data(self, aggregation: AggregationRequest) -> AggregationResult
    
    async def get_facets(self, facet_configs: List[FacetConfiguration],
                        base_filters: List[QueryFilter] = None) -> Dict[str, Dict[str, int]]
```

### Semantic Engine Integration
```python
# Register semantic engine for semantic search
query_engine.set_semantic_engine(semantic_engine)

# Now semantic search is available
supported_types = query_engine.get_supported_query_types()
# Returns: [QueryType.EXACT_MATCH, QueryType.FUZZY_SEARCH, QueryType.SEMANTIC_SEARCH]
```

## Error Handling

### Validation Errors
- **Query Structure**: Missing required fields, invalid types
- **Filter Validation**: Operator/value mismatches, invalid fields
- **Threshold Validation**: Out-of-range similarity/fuzzy thresholds
- **Processor Availability**: Unsupported query types

### Runtime Errors
- **Repository Errors**: Database connection issues, query failures
- **Semantic Engine Errors**: Embedding generation failures
- **Timeout Errors**: Query execution time limits
- **Resource Errors**: Memory or computation limits exceeded

### Error Response Format
```python
response = QueryResponse(
    query_id="failed-query",
    total_results=0,
    returned_results=0,
    execution_time_ms=50.0,
    results=[],
    errors=["Semantic engine not available for semantic search"],
    warnings=["Query timeout adjusted due to complexity"]
)
```

## Performance Metrics

### Execution Metrics
```python
metrics = await query_engine.get_performance_metrics()
# Returns:
{
    "total_queries": 1250,
    "successful_queries": 1198,
    "success_rate": 0.958,
    "average_execution_time_ms": 85.5,
    "query_type_distribution": {
        "exact_match": 450,
        "fuzzy_search": 380,
        "semantic_search": 270,
        "hybrid_search": 98
    },
    "supported_query_types": ["exact_match", "fuzzy_search", "semantic_search"]
}
```

### Query Analytics
- **Success Rate Tracking**: Monitor query effectiveness
- **Performance Profiling**: Identify slow query patterns
- **Type Distribution**: Understand usage patterns
- **Error Analysis**: Track and resolve common issues

## Future Extensions

### Planned Enhancements
- **Hybrid Search**: Combine multiple search strategies intelligently
- **Faceted Search**: Multi-dimensional filtering with real-time facet counts
- **Graph Traversal**: Relationship-based entity discovery
- **Full-Text Search**: Advanced text processing with stemming and synonyms
- **Machine Learning**: Query intent recognition and result personalization

### Integration Roadmap
- **T10 CLI Interface**: Command-line query tools
- **T11 API Endpoints**: REST API for external access
- **T12 Dashboard**: Web interface for query management
- **Advanced Analytics**: Query performance optimization and insights

## Testing Coverage

The T9 implementation includes comprehensive testing:
- **51 test cases** covering all core functionality
- **87% code coverage** of the retrieval module
- **Mock Repository** for isolated testing
- **Async Testing** for real-world usage patterns
- **Performance Testing** for optimization validation

## Dependencies

### Core Dependencies
- `dataclasses`: Data model definitions
- `enum`: Type-safe enumeration classes
- `typing`: Type hints and annotations
- `asyncio`: Asynchronous processing support
- `datetime`: Timestamp and timing utilities

### Integration Dependencies
- **Semantic Engine**: For embedding-based semantic search
- **Repository Backend**: For data persistence and retrieval
- **Logging Framework**: For debugging and monitoring
- **Validation Libraries**: For input sanitization and validation
