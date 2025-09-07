# T9 Retrieval & Query Interface Tests

import asyncio
import pytest
from datetime import datetime
from typing import List, Optional, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock

from echo_roots.retrieval import (
    # Enums
    QueryType, SortOrder, FilterOperator, AggregationType,
    
    # Data models
    QueryFilter, SortCriterion, AggregationRequest, QueryRequest,
    QueryResult, AggregationResult, QueryResponse, QueryHistory,
    FacetConfiguration,
    
    # Abstract interfaces
    QueryProcessor, RetrievalRepository,
    
    # Components
    FilterValidator, QueryOptimizer, ExactMatchProcessor,
    FuzzySearchProcessor, SemanticSearchProcessor, QueryEngine
)


class TestQueryEnums:
    """Test query enumeration types."""
    
    def test_query_type_enum_values(self):
        """Test QueryType enum has expected values."""
        expected_types = {
            "exact_match", "fuzzy_search", "semantic_search", "hybrid_search",
            "faceted_search", "graph_traversal", "full_text_search", "aggregation"
        }
        actual_types = {qt.value for qt in QueryType}
        assert actual_types == expected_types
    
    def test_sort_order_enum_values(self):
        """Test SortOrder enum has expected values."""
        assert SortOrder.ASC.value == "ascending"
        assert SortOrder.DESC.value == "descending"
    
    def test_filter_operator_enum_values(self):
        """Test FilterOperator enum has comprehensive operators."""
        operators = {op.value for op in FilterOperator}
        expected_operators = {
            "eq", "ne", "contains", "not_contains", "starts_with", "ends_with",
            "in", "not_in", "gt", "gte", "lt", "lte", "between", 
            "is_null", "is_not_null", "regex"
        }
        assert operators == expected_operators
    
    def test_aggregation_type_enum_values(self):
        """Test AggregationType enum has expected values."""
        agg_types = {at.value for at in AggregationType}
        expected_types = {
            "count", "sum", "avg", "min", "max", "distinct_count", 
            "group_by", "percentile"
        }
        assert agg_types == expected_types


class TestQueryDataModels:
    """Test query data model classes."""
    
    def test_query_filter_creation(self):
        """Test QueryFilter creation and defaults."""
        filter_obj = QueryFilter(
            field="name",
            operator=FilterOperator.CONTAINS,
            value="test"
        )
        
        assert filter_obj.field == "name"
        assert filter_obj.operator == FilterOperator.CONTAINS
        assert filter_obj.value == "test"
        assert filter_obj.case_sensitive is False
        assert filter_obj.weight == 1.0
        assert filter_obj.metadata == {}
    
    def test_query_filter_with_metadata(self):
        """Test QueryFilter with custom metadata."""
        metadata = {"source": "user_input", "priority": "high"}
        filter_obj = QueryFilter(
            field="category",
            operator=FilterOperator.IN,
            value=["electronics", "books"],
            case_sensitive=True,
            weight=2.5,
            metadata=metadata
        )
        
        assert filter_obj.case_sensitive is True
        assert filter_obj.weight == 2.5
        assert filter_obj.metadata == metadata
    
    def test_sort_criterion_creation(self):
        """Test SortCriterion creation and defaults."""
        sort_obj = SortCriterion(field="price")
        
        assert sort_obj.field == "price"
        assert sort_obj.order == SortOrder.ASC
        assert sort_obj.weight == 1.0
        assert sort_obj.null_handling == "last"
    
    def test_query_request_comprehensive(self):
        """Test comprehensive QueryRequest creation."""
        filters = [
            QueryFilter("category", FilterOperator.EQUALS, "electronics"),
            QueryFilter("price", FilterOperator.LESS_THAN, 100)
        ]
        sorts = [SortCriterion("price", SortOrder.DESC)]
        aggregations = [
            AggregationRequest(AggregationType.COUNT, "id"),
            AggregationRequest(AggregationType.AVG, "price", group_by_fields=["category"])
        ]
        
        request = QueryRequest(
            query_id="test-query-001",
            query_type=QueryType.FUZZY_SEARCH,
            search_text="laptop computer",
            target_types=["product"],
            filters=filters,
            sort_criteria=sorts,
            limit=50,
            offset=20,
            include_metadata=True,
            include_relationships=True,
            similarity_threshold=0.7,
            fuzzy_threshold=0.8,
            boost_fields={"title": 2.0, "description": 1.5},
            aggregations=aggregations,
            timeout_seconds=45
        )
        
        assert request.query_id == "test-query-001"
        assert request.query_type == QueryType.FUZZY_SEARCH
        assert request.search_text == "laptop computer"
        assert request.target_types == ["product"]
        assert len(request.filters) == 2
        assert len(request.sort_criteria) == 1
        assert request.limit == 50
        assert request.offset == 20
        assert request.include_metadata is True
        assert request.include_relationships is True
        assert request.similarity_threshold == 0.7
        assert request.fuzzy_threshold == 0.8
        assert request.boost_fields == {"title": 2.0, "description": 1.5}
        assert len(request.aggregations) == 2
        assert request.timeout_seconds == 45
        assert isinstance(request.created_at, datetime)
    
    def test_query_result_creation(self):
        """Test QueryResult creation."""
        result = QueryResult(
            entity_id="prod-123",
            entity_type="product",
            score=0.95,
            data={"name": "Laptop", "price": 899.99},
            relationships=[{"type": "category", "target": "electronics"}],
            embeddings={"text": [0.1, 0.2, 0.3]},
            highlights={"name": ["<em>Laptop</em>"]},
            explanation="Exact match on product name"
        )
        
        assert result.entity_id == "prod-123"
        assert result.entity_type == "product"
        assert result.score == 0.95
        assert result.data["name"] == "Laptop"
        assert len(result.relationships) == 1
        assert result.embeddings["text"] == [0.1, 0.2, 0.3]
        assert result.explanation == "Exact match on product name"
    
    def test_query_response_comprehensive(self):
        """Test comprehensive QueryResponse creation."""
        results = [
            QueryResult("1", "product", 0.9, {"name": "Product 1"}),
            QueryResult("2", "product", 0.8, {"name": "Product 2"})
        ]
        aggregations = [
            AggregationResult(AggregationType.COUNT, "id", 100)
        ]
        facets = {"category": {"electronics": 45, "books": 30}}
        
        response = QueryResponse(
            query_id="test-query-001",
            total_results=100,
            returned_results=2,
            execution_time_ms=150.5,
            results=results,
            aggregations=aggregations,
            facets=facets,
            suggestions=["laptop", "computer", "notebook"],
            errors=["Warning: fuzzy threshold very low"],
            warnings=["Large result set truncated"]
        )
        
        assert response.query_id == "test-query-001"
        assert response.total_results == 100
        assert response.returned_results == 2
        assert response.execution_time_ms == 150.5
        assert len(response.results) == 2
        assert len(response.aggregations) == 1
        assert response.facets["category"]["electronics"] == 45
        assert "laptop" in response.suggestions
        assert len(response.errors) == 1
        assert len(response.warnings) == 1


class TestFilterValidator:
    """Test filter validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FilterValidator()
    
    def test_validate_valid_filters(self):
        """Test validation of valid filters."""
        filters = [
            QueryFilter("name", FilterOperator.CONTAINS, "test"),
            QueryFilter("price", FilterOperator.GREATER_THAN, 100),
            QueryFilter("category", FilterOperator.IN, ["electronics", "books"]),
            QueryFilter("active", FilterOperator.EQUALS, True)
        ]
        
        errors = self.validator.validate_filters(filters)
        assert len(errors) == 0
    
    def test_validate_invalid_field_name(self):
        """Test validation with invalid field name."""
        filters = [
            QueryFilter("", FilterOperator.EQUALS, "test"),
            QueryFilter(None, FilterOperator.CONTAINS, "value")
        ]
        
        errors = self.validator.validate_filters(filters)
        assert len(errors) == 2
        assert "Field name must be a non-empty string" in errors[0]
        assert "Field name must be a non-empty string" in errors[1]
    
    def test_validate_invalid_operators(self):
        """Test validation with invalid operators."""
        # This would be caught at the enum level, but let's test our validation
        filters = [
            QueryFilter("field", FilterOperator.IN, "not_a_list"),  # Wrong value type
            QueryFilter("field", FilterOperator.BETWEEN, "not_a_range"),  # Wrong value type
            QueryFilter("field", FilterOperator.REGEX, 123)  # Wrong value type
        ]
        
        errors = self.validator.validate_filters(filters)
        assert len(errors) == 3
        assert "requires a list value" in errors[0]
        assert "requires a list/tuple with exactly 2 values" in errors[1]
        assert "requires a string value" in errors[2]
    
    def test_validate_null_operators(self):
        """Test validation of null operators that don't need values."""
        filters = [
            QueryFilter("field", FilterOperator.IS_NULL, None),
            QueryFilter("field", FilterOperator.IS_NOT_NULL, "ignored_value")
        ]
        
        errors = self.validator.validate_filters(filters)
        assert len(errors) == 0
    
    def test_normalize_filter_case_insensitive(self):
        """Test filter normalization for case insensitive operations."""
        filter_obj = QueryFilter(
            field="NAME",
            operator=FilterOperator.CONTAINS,
            value="TEST_VALUE",
            case_sensitive=False
        )
        
        normalized = self.validator.normalize_filter(filter_obj)
        assert normalized.field == "name"
        assert normalized.value == "test_value"
    
    def test_normalize_filter_case_sensitive(self):
        """Test filter normalization for case sensitive operations."""
        filter_obj = QueryFilter(
            field="NAME",
            operator=FilterOperator.CONTAINS,
            value="TEST_Value",
            case_sensitive=True
        )
        
        normalized = self.validator.normalize_filter(filter_obj)
        assert normalized.field == "name"  # Field is always normalized
        assert normalized.value == "TEST_Value"  # Value preserves case
    
    def test_normalize_filter_weight_clamping(self):
        """Test filter weight clamping during normalization."""
        filter_obj = QueryFilter(
            field="test",
            operator=FilterOperator.EQUALS,
            value="value",
            weight=15.0  # Over the max of 10
        )
        
        normalized = self.validator.normalize_filter(filter_obj)
        assert normalized.weight == 10.0
        
        filter_obj.weight = -5.0  # Under the min of 0
        normalized = self.validator.normalize_filter(filter_obj)
        assert normalized.weight == 0.0


class TestQueryOptimizer:
    """Test query optimization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = QueryOptimizer()
    
    def test_optimize_duplicate_filters(self):
        """Test removal of duplicate filters."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.EXACT_MATCH,
            search_text="test",
            filters=[
                QueryFilter("name", FilterOperator.EQUALS, "test"),
                QueryFilter("price", FilterOperator.GREATER_THAN, 100),
                QueryFilter("name", FilterOperator.EQUALS, "test"),  # Duplicate
            ]
        )
        
        optimized = self.optimizer.optimize_query(request)
        assert len(optimized.filters) == 2
        
        # Check that we kept the unique filters
        field_operators = [(f.field, f.operator, str(f.value)) for f in optimized.filters]
        assert ("name", FilterOperator.EQUALS, "test") in field_operators
        assert ("price", FilterOperator.GREATER_THAN, "100") in field_operators
    
    def test_optimize_filter_ordering(self):
        """Test filter ordering by selectivity."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.FUZZY_SEARCH,
            search_text="test",
            filters=[
                QueryFilter("description", FilterOperator.CONTAINS, "laptop"),  # Less selective
                QueryFilter("id", FilterOperator.EQUALS, "123"),  # More selective
                QueryFilter("category", FilterOperator.IN, ["electronics"]),  # More selective
            ]
        )
        
        optimized = self.optimizer.optimize_query(request)
        
        # More selective filters should come first
        assert optimized.filters[0].operator in [FilterOperator.EQUALS, FilterOperator.IN]
        assert optimized.filters[-1].operator == FilterOperator.CONTAINS
    
    def test_optimize_duplicate_sorts(self):
        """Test removal of duplicate sort criteria."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.SEMANTIC_SEARCH,
            search_text="test",
            sort_criteria=[
                SortCriterion("price", SortOrder.DESC),
                SortCriterion("name", SortOrder.ASC),
                SortCriterion("price", SortOrder.ASC),  # Duplicate field (different order)
            ]
        )
        
        optimized = self.optimizer.optimize_query(request)
        assert len(optimized.sort_criteria) == 2
        
        # Should keep first occurrence of each field
        fields = [s.field for s in optimized.sort_criteria]
        assert fields == ["price", "name"]
        assert optimized.sort_criteria[0].order == SortOrder.DESC  # Original order preserved
    
    def test_optimize_limit_constraints(self):
        """Test limit and offset optimization."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.EXACT_MATCH,
            search_text="test",
            limit=5000,  # Over max
            offset=-10,  # Negative
            timeout_seconds=120  # Long timeout
        )
        
        optimized = self.optimizer.optimize_query(request)
        assert optimized.limit == 1000  # Clamped to max
        assert optimized.offset == 0  # Clamped to non-negative
        assert optimized.timeout_seconds <= 120  # Should be adjusted based on complexity
    
    def test_optimize_timeout_based_on_complexity(self):
        """Test timeout adjustment based on query complexity."""
        # Simple query
        simple_request = QueryRequest(
            query_id="simple",
            query_type=QueryType.EXACT_MATCH,
            search_text="test",
            timeout_seconds=60
        )
        optimized_simple = self.optimizer.optimize_query(simple_request)
        simple_timeout = optimized_simple.timeout_seconds
        
        # Complex query
        complex_request = QueryRequest(
            query_id="complex",
            query_type=QueryType.SEMANTIC_SEARCH,
            search_text="test",
            filters=[QueryFilter(f"field_{i}", FilterOperator.EQUALS, f"value_{i}") for i in range(15)],
            aggregations=[AggregationRequest(AggregationType.COUNT, f"field_{i}") for i in range(10)],
            include_embeddings=True,
            timeout_seconds=60
        )
        optimized_complex = self.optimizer.optimize_query(complex_request)
        complex_timeout = optimized_complex.timeout_seconds
        
        # Complex query should have adjusted timeout
        assert complex_timeout <= 60  # Should be reduced due to complexity


class MockRetrievalRepository(RetrievalRepository):
    """Mock repository for testing."""
    
    def __init__(self):
        self.entities = [
            {"id": "1", "type": "product", "name": "Laptop Computer", "price": 899.99, "category": "electronics"},
            {"id": "2", "type": "product", "name": "Desktop Computer", "price": 699.99, "category": "electronics"},
            {"id": "3", "type": "product", "name": "Tablet", "price": 299.99, "category": "electronics"},
            {"id": "4", "type": "book", "name": "Python Programming", "price": 49.99, "category": "books"},
            {"id": "5", "type": "book", "name": "Computer Science", "price": 79.99, "category": "books"},
        ]
    
    async def search_entities(
        self, 
        query: str, 
        entity_types: List[str] = None,
        filters: List[QueryFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[QueryResult]:
        """Mock entity search."""
        results = []
        
        for entity in self.entities:
            # Apply type filter
            if entity_types and entity["type"] not in entity_types:
                continue
            
            # Apply filters
            if filters:
                matches_filters = True
                for filter_obj in filters:
                    entity_value = entity.get(filter_obj.field)
                    if not self._apply_filter(entity_value, filter_obj):
                        matches_filters = False
                        break
                if not matches_filters:
                    continue
            
            # Simple text matching
            if query and query.lower() not in entity["name"].lower():
                continue
            
            # Create result
            result = QueryResult(
                entity_id=entity["id"],
                entity_type=entity["type"],
                score=1.0,
                data=entity.copy()
            )
            results.append(result)
        
        # Apply pagination
        return results[offset:offset + limit]
    
    def _apply_filter(self, entity_value: Any, filter_obj: QueryFilter) -> bool:
        """Apply a single filter to entity value."""
        if filter_obj.operator == FilterOperator.EQUALS:
            return entity_value == filter_obj.value
        elif filter_obj.operator == FilterOperator.GREATER_THAN:
            return entity_value > filter_obj.value if entity_value is not None else False
        elif filter_obj.operator == FilterOperator.LESS_THAN:
            return entity_value < filter_obj.value if entity_value is not None else False
        elif filter_obj.operator == FilterOperator.CONTAINS:
            return str(filter_obj.value).lower() in str(entity_value).lower() if entity_value else False
        elif filter_obj.operator == FilterOperator.IN:
            return entity_value in filter_obj.value if entity_value is not None else False
        else:
            return True  # Default to true for unsupported operators in mock
    
    async def get_entity_by_id(self, entity_id: str, include_relationships: bool = False) -> Optional[QueryResult]:
        """Mock get entity by ID."""
        for entity in self.entities:
            if entity["id"] == entity_id:
                return QueryResult(
                    entity_id=entity["id"],
                    entity_type=entity["type"],
                    score=1.0,
                    data=entity.copy()
                )
        return None
    
    async def get_entities_by_ids(self, entity_ids: List[str]) -> List[QueryResult]:
        """Mock get multiple entities by IDs."""
        results = []
        for entity_id in entity_ids:
            entity = await self.get_entity_by_id(entity_id)
            if entity:
                results.append(entity)
        return results
    
    async def aggregate_data(self, aggregation: AggregationRequest) -> AggregationResult:
        """Mock data aggregation."""
        return AggregationResult(
            aggregation_type=aggregation.aggregation_type,
            field=aggregation.field,
            value=len(self.entities)
        )
    
    async def get_facets(
        self, 
        facet_configs: List[FacetConfiguration],
        base_filters: List[QueryFilter] = None
    ) -> Dict[str, Dict[str, int]]:
        """Mock facet calculation."""
        return {
            "category": {"electronics": 3, "books": 2},
            "type": {"product": 3, "book": 2}
        }


class TestExactMatchProcessor:
    """Test exact match query processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = MockRetrievalRepository()
        self.processor = ExactMatchProcessor(self.repository)
    
    def test_supports_query_type(self):
        """Test query type support."""
        assert self.processor.supports_query_type(QueryType.EXACT_MATCH)
        assert not self.processor.supports_query_type(QueryType.FUZZY_SEARCH)
        assert not self.processor.supports_query_type(QueryType.SEMANTIC_SEARCH)
    
    @pytest.mark.asyncio
    async def test_validate_query_success(self):
        """Test successful query validation."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.EXACT_MATCH,
            search_text="laptop",
            filters=[QueryFilter("category", FilterOperator.EQUALS, "electronics")]
        )
        
        errors = await self.processor.validate_query(request)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_query_missing_text(self):
        """Test validation failure for missing search text."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.EXACT_MATCH
        )
        
        errors = await self.processor.validate_query(request)
        assert len(errors) == 1
        assert "require search text" in errors[0]
    
    @pytest.mark.asyncio
    async def test_process_exact_match_query(self):
        """Test processing exact match query."""
        request = QueryRequest(
            query_id="test-exact-001",
            query_type=QueryType.EXACT_MATCH,
            search_text="Laptop Computer",
            target_types=["product"],
            limit=10
        )
        
        response = await self.processor.process_query(request)
        
        assert response.query_id == "test-exact-001"
        assert response.total_results >= 0
        assert response.execution_time_ms > 0
        assert len(response.errors) == 0
        
        # Should find exact matches
        for result in response.results:
            assert "Laptop Computer" in result.data.get("name", "")
    
    @pytest.mark.asyncio
    async def test_process_query_with_filters(self):
        """Test processing query with filters."""
        request = QueryRequest(
            query_id="test-filtered",
            query_type=QueryType.EXACT_MATCH,
            search_text="Computer",
            filters=[QueryFilter("category", FilterOperator.EQUALS, "electronics")],
            sort_criteria=[SortCriterion("price", SortOrder.DESC)]
        )
        
        response = await self.processor.process_query(request)
        
        assert len(response.errors) == 0
        
        # Results should be filtered and sorted
        for result in response.results:
            assert result.data.get("category") == "electronics"


class TestFuzzySearchProcessor:
    """Test fuzzy search query processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = MockRetrievalRepository()
        self.processor = FuzzySearchProcessor(self.repository)
    
    def test_supports_query_type(self):
        """Test query type support."""
        assert self.processor.supports_query_type(QueryType.FUZZY_SEARCH)
        assert not self.processor.supports_query_type(QueryType.EXACT_MATCH)
        assert not self.processor.supports_query_type(QueryType.SEMANTIC_SEARCH)
    
    @pytest.mark.asyncio
    async def test_validate_query_success(self):
        """Test successful query validation."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.FUZZY_SEARCH,
            search_text="laptap",  # Typo
            fuzzy_threshold=0.7
        )
        
        errors = await self.processor.validate_query(request)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_query_invalid_threshold(self):
        """Test validation failure for invalid threshold."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.FUZZY_SEARCH,
            search_text="test",
            fuzzy_threshold=1.5  # Invalid
        )
        
        errors = await self.processor.validate_query(request)
        assert len(errors) == 1
        assert "threshold must be between 0 and 1" in errors[0]
    
    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity calculation."""
        # Exact match
        assert self.processor._levenshtein_similarity("test", "test") == 1.0
        
        # No match
        assert self.processor._levenshtein_similarity("test", "xyz") < 0.5
        
        # Partial match
        sim = self.processor._levenshtein_similarity("laptop", "laptap")
        assert 0.8 < sim < 1.0
        
        # Empty strings
        assert self.processor._levenshtein_similarity("", "") == 1.0
        assert self.processor._levenshtein_similarity("test", "") == 0.0
        assert self.processor._levenshtein_similarity("", "test") == 0.0
    
    @pytest.mark.asyncio
    async def test_process_fuzzy_search_query(self):
        """Test processing fuzzy search query."""
        request = QueryRequest(
            query_id="test-fuzzy-001",
            query_type=QueryType.FUZZY_SEARCH,
            search_text="laptap",  # Typo in "laptop"
            target_types=["product"],
            fuzzy_threshold=0.7,
            limit=10
        )
        
        response = await self.processor.process_query(request)
        
        assert response.query_id == "test-fuzzy-001"
        assert response.execution_time_ms > 0
        assert len(response.errors) == 0
        
        # Should find fuzzy matches with scores
        for result in response.results:
            assert result.score >= 0.7
            assert "Fuzzy match" in result.explanation


class TestSemanticSearchProcessor:
    """Test semantic search query processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = MockRetrievalRepository()
        self.semantic_engine = Mock()
        self.semantic_engine.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        self.processor = SemanticSearchProcessor(self.repository, self.semantic_engine)
    
    def test_supports_query_type(self):
        """Test query type support."""
        assert self.processor.supports_query_type(QueryType.SEMANTIC_SEARCH)
        assert not self.processor.supports_query_type(QueryType.EXACT_MATCH)
        assert not self.processor.supports_query_type(QueryType.FUZZY_SEARCH)
    
    @pytest.mark.asyncio
    async def test_validate_query_success(self):
        """Test successful query validation."""
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.SEMANTIC_SEARCH,
            search_text="portable computer",
            similarity_threshold=0.6
        )
        
        errors = await self.processor.validate_query(request)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_query_no_semantic_engine(self):
        """Test validation failure when no semantic engine available."""
        processor = SemanticSearchProcessor(self.repository, None)
        request = QueryRequest(
            query_id="test",
            query_type=QueryType.SEMANTIC_SEARCH,
            search_text="test"
        )
        
        errors = await processor.validate_query(request)
        assert len(errors) == 1
        assert "Semantic engine not available" in errors[0]
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1, 0, 0]
        vec2 = [1, 0, 0]
        assert self.processor._cosine_similarity(vec1, vec2) == 1.0
        
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        assert self.processor._cosine_similarity(vec1, vec2) == 0.0
        
        vec1 = [1, 1, 0]
        vec2 = [1, 0, 0]
        sim = self.processor._cosine_similarity(vec1, vec2)
        assert 0.7 < sim < 0.8
        
        # Different lengths
        vec1 = [1, 0]
        vec2 = [1, 0, 0]
        assert self.processor._cosine_similarity(vec1, vec2) == 0.0
        
        # Zero vectors
        vec1 = [0, 0, 0]
        vec2 = [1, 2, 3]
        assert self.processor._cosine_similarity(vec1, vec2) == 0.0
    
    @pytest.mark.asyncio
    async def test_process_semantic_search_query(self):
        """Test processing semantic search query."""
        request = QueryRequest(
            query_id="test-semantic-001",
            query_type=QueryType.SEMANTIC_SEARCH,
            search_text="portable computing device",
            target_types=["product"],
            similarity_threshold=0.3,
            limit=10
        )
        
        response = await self.processor.process_query(request)
        
        assert response.query_id == "test-semantic-001"
        assert response.execution_time_ms > 0
        assert len(response.errors) == 0
        
        # Should have called semantic engine
        self.semantic_engine.generate_embedding.assert_called()
        
        # Results should have semantic similarity scores
        for result in response.results:
            assert result.score >= 0.3
            assert "Semantic similarity" in result.explanation


class TestQueryEngine:
    """Test main query engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.repository = MockRetrievalRepository()
        self.engine = QueryEngine(self.repository)
        
        # Mock semantic engine
        self.semantic_engine = Mock()
        self.semantic_engine.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        self.engine.set_semantic_engine(self.semantic_engine)
    
    def test_initialization(self):
        """Test query engine initialization."""
        assert isinstance(self.engine.repository, MockRetrievalRepository)
        assert isinstance(self.engine.query_optimizer, QueryOptimizer)
        assert len(self.engine.processors) >= 2  # At least exact and fuzzy processors
        assert QueryType.EXACT_MATCH in self.engine.processors
        assert QueryType.FUZZY_SEARCH in self.engine.processors
    
    def test_register_processor(self):
        """Test processor registration."""
        mock_processor = Mock(spec=QueryProcessor)
        self.engine.register_processor(QueryType.HYBRID_SEARCH, mock_processor)
        
        assert QueryType.HYBRID_SEARCH in self.engine.processors
        assert self.engine.processors[QueryType.HYBRID_SEARCH] == mock_processor
    
    def test_set_semantic_engine(self):
        """Test semantic engine registration."""
        # Should have registered semantic processor
        assert QueryType.SEMANTIC_SEARCH in self.engine.processors
        assert isinstance(self.engine.processors[QueryType.SEMANTIC_SEARCH], SemanticSearchProcessor)
    
    def test_get_supported_query_types(self):
        """Test getting supported query types."""
        supported_types = self.engine.get_supported_query_types()
        assert QueryType.EXACT_MATCH in supported_types
        assert QueryType.FUZZY_SEARCH in supported_types
        assert QueryType.SEMANTIC_SEARCH in supported_types
    
    @pytest.mark.asyncio
    async def test_validate_query_success(self):
        """Test successful query validation."""
        request = QueryRequest(
            query_id="test-001",
            query_type=QueryType.EXACT_MATCH,
            search_text="laptop"
        )
        
        errors = await self.engine.validate_query(request)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_query_missing_id(self):
        """Test validation failure for missing query ID."""
        request = QueryRequest(
            query_id="",
            query_type=QueryType.EXACT_MATCH,
            search_text="laptop"
        )
        
        errors = await self.engine.validate_query(request)
        assert len(errors) == 1
        assert "Query ID is required" in errors[0]
    
    @pytest.mark.asyncio
    async def test_validate_query_unsupported_type(self):
        """Test validation failure for unsupported query type."""
        request = QueryRequest(
            query_id="test-001",
            query_type=QueryType.GRAPH_TRAVERSAL  # Not registered
        )
        
        errors = await self.engine.validate_query(request)
        assert len(errors) == 1
        assert "No processor available" in errors[0]
    
    @pytest.mark.asyncio
    async def test_process_exact_match_query(self):
        """Test processing exact match query through engine."""
        request = QueryRequest(
            query_id="engine-test-001",
            query_type=QueryType.EXACT_MATCH,
            search_text="Laptop Computer",
            target_types=["product"]
        )
        
        response = await self.engine.process_query(request)
        
        assert response.query_id == "engine-test-001"
        assert response.execution_time_ms > 0
        assert len(response.errors) == 0
        
        # Should be added to history
        assert len(self.engine.query_history) == 1
        assert self.engine.query_history[0].query_id == "engine-test-001"
        assert self.engine.query_history[0].success is True
    
    @pytest.mark.asyncio
    async def test_process_fuzzy_search_query(self):
        """Test processing fuzzy search query through engine."""
        request = QueryRequest(
            query_id="engine-test-002",
            query_type=QueryType.FUZZY_SEARCH,
            search_text="laptap",  # Typo
            fuzzy_threshold=0.7
        )
        
        response = await self.engine.process_query(request)
        
        assert response.query_id == "engine-test-002"
        assert len(response.errors) == 0
        
        # Check history
        history = self.engine.get_query_history(limit=1)
        assert len(history) == 1
        assert history[0].request.query_type == QueryType.FUZZY_SEARCH
    
    @pytest.mark.asyncio
    async def test_process_semantic_search_query(self):
        """Test processing semantic search query through engine."""
        request = QueryRequest(
            query_id="engine-test-003",
            query_type=QueryType.SEMANTIC_SEARCH,
            search_text="portable computing device",
            similarity_threshold=0.5
        )
        
        response = await self.engine.process_query(request)
        
        assert response.query_id == "engine-test-003"
        assert len(response.errors) == 0
        
        # Should have used semantic engine
        self.semantic_engine.generate_embedding.assert_called()
    
    @pytest.mark.asyncio
    async def test_process_query_unsupported_type(self):
        """Test processing query with unsupported type."""
        request = QueryRequest(
            query_id="engine-test-004",
            query_type=QueryType.GRAPH_TRAVERSAL,  # Not registered
            search_text="test"
        )
        
        response = await self.engine.process_query(request)
        
        assert response.query_id == "engine-test-004"
        assert len(response.errors) == 1
        assert "No processor available" in response.errors[0]
        assert response.total_results == 0
    
    def test_get_query_history_filtering(self):
        """Test query history filtering."""
        # Add some mock history
        self.engine.query_history = [
            QueryHistory("1", 
                QueryRequest("1", QueryType.EXACT_MATCH, search_text="test1"), 
                QueryResponse("1", 1, 1, 100, []), success=True),
            QueryHistory("2", 
                QueryRequest("2", QueryType.FUZZY_SEARCH, search_text="test2"), 
                QueryResponse("2", 0, 0, 50, [], errors=["error"]), success=False),
            QueryHistory("3", 
                QueryRequest("3", QueryType.EXACT_MATCH, search_text="test3"), 
                QueryResponse("3", 2, 2, 200, []), success=True),
        ]
        
        # Test filtering by query type
        exact_history = self.engine.get_query_history(query_type=QueryType.EXACT_MATCH)
        assert len(exact_history) == 2
        
        # Test filtering by success
        success_history = self.engine.get_query_history(success_only=True)
        assert len(success_history) == 2
        
        # Test limit
        limited_history = self.engine.get_query_history(limit=1)
        assert len(limited_history) == 1
        assert limited_history[0].query_id == "3"  # Most recent
    
    @pytest.mark.asyncio
    async def test_get_query_suggestions(self):
        """Test query suggestion generation."""
        # Add some mock history
        self.engine.query_history = [
            QueryHistory("1", 
                QueryRequest("1", QueryType.EXACT_MATCH, search_text="laptop computer"), 
                QueryResponse("1", 1, 1, 100, []), success=True),
            QueryHistory("2", 
                QueryRequest("2", QueryType.FUZZY_SEARCH, search_text="desktop computer"), 
                QueryResponse("2", 1, 1, 50, []), success=True),
            QueryHistory("3", 
                QueryRequest("3", QueryType.EXACT_MATCH, search_text="tablet device"), 
                QueryResponse("3", 0, 0, 200, [], errors=["error"]), success=False),
        ]
        
        suggestions = await self.engine.get_query_suggestions("comp", limit=3)
        
        # Should include successful queries containing "comp"
        assert len(suggestions) == 2
        assert "laptop computer" in suggestions
        assert "desktop computer" in suggestions
        assert "tablet device" not in suggestions  # Failed query
    
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some mock history
        self.engine.query_history = [
            QueryHistory("1", 
                QueryRequest("1", QueryType.EXACT_MATCH, search_text="test1"), 
                QueryResponse("1", 1, 1, 100, []), success=True),
            QueryHistory("2", 
                QueryRequest("2", QueryType.FUZZY_SEARCH, search_text="test2"), 
                QueryResponse("2", 0, 0, 50, [], errors=["error"]), success=False),
            QueryHistory("3", 
                QueryRequest("3", QueryType.EXACT_MATCH, search_text="test3"), 
                QueryResponse("3", 2, 2, 200, []), success=True),
        ]
        
        metrics = await self.engine.get_performance_metrics()
        
        assert metrics["total_queries"] == 3
        assert metrics["successful_queries"] == 2
        assert metrics["success_rate"] == 2/3
        assert metrics["average_execution_time_ms"] == (100 + 50 + 200) / 3
        assert metrics["query_type_distribution"][QueryType.EXACT_MATCH] == 2
        assert metrics["query_type_distribution"][QueryType.FUZZY_SEARCH] == 1
        assert len(metrics["supported_query_types"]) >= 3
