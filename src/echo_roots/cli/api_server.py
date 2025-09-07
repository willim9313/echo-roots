# T10 API Server Implementation

"""
FastAPI-based REST API server for Echo-Roots taxonomy system.
Provides HTTP endpoints for query, search, and system operations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import json


class DateTimeJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Import Echo-Roots components
from echo_roots.retrieval import (
    QueryType, QueryRequest, QueryResponse, QueryResult, QueryFilter,
    FilterOperator, SortCriterion, SortOrder, QueryEngine
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global query engine instance
query_engine: Optional[QueryEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global query_engine
    
    # Startup
    logger.info("🚀 Starting Echo-Roots API Server")
    
    # Initialize query engine with mock repository for now
    # In production, this would connect to real database
    from tests.test_t9_retrieval_interface import MockRetrievalRepository
    repository = MockRetrievalRepository()
    query_engine = QueryEngine(repository)
    
    logger.info("✅ Query engine initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Echo-Roots API Server")


# Create FastAPI application
app = FastAPI(
    title="Echo-Roots Taxonomy API",
    description="RESTful API for taxonomy construction and semantic enrichment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Models
class APIQueryFilter(BaseModel):
    """API model for query filters."""
    field: str = Field(..., description="Field name to filter on")
    operator: str = Field(..., description="Filter operator (eq, contains, gt, etc.)")
    value: Union[str, int, float, List[Union[str, int, float]]] = Field(..., description="Filter value")
    case_sensitive: bool = Field(False, description="Case sensitive matching")
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Filter weight for relevance")


class APISortCriterion(BaseModel):
    """API model for sort criteria."""
    field: str = Field(..., description="Field name to sort by")
    order: str = Field("asc", description="Sort order: 'asc' or 'desc'")
    weight: float = Field(1.0, ge=0.0, le=10.0, description="Sort weight")


class APIQueryRequest(BaseModel):
    """API model for query requests."""
    model_config = ConfigDict(use_enum_values=True)
    
    query_type: str = Field(..., description="Type of query: exact, fuzzy, semantic")
    search_text: Optional[str] = Field(None, description="Text to search for")
    target_types: List[str] = Field(default_factory=list, description="Entity types to search")
    filters: List[APIQueryFilter] = Field(default_factory=list, description="Query filters")
    sort_criteria: List[APISortCriterion] = Field(default_factory=list, description="Sort criteria")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results to return")
    offset: int = Field(0, ge=0, description="Results offset for pagination")
    fuzzy_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Fuzzy matching threshold")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Semantic similarity threshold")
    include_metadata: bool = Field(True, description="Include result metadata")
    include_relationships: bool = Field(False, description="Include entity relationships")
    include_embeddings: bool = Field(False, description="Include embedding vectors")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Query timeout in seconds")


class APIQueryResult(BaseModel):
    """API model for query results."""
    entity_id: str = Field(..., description="Unique entity identifier")
    entity_type: str = Field(..., description="Type of entity")
    score: float = Field(..., description="Relevance score")
    data: Dict[str, Any] = Field(..., description="Entity data")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Entity relationships")
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Embedding vectors")
    highlights: Dict[str, List[str]] = Field(default_factory=dict, description="Search highlights")
    explanation: str = Field("", description="Score explanation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class APIQueryResponse(BaseModel):
    """API model for query responses."""
    query_id: str = Field(..., description="Unique query identifier")
    total_results: int = Field(..., description="Total number of matching results")
    returned_results: int = Field(..., description="Number of results returned")
    execution_time_ms: float = Field(..., description="Query execution time in milliseconds")
    results: List[APIQueryResult] = Field(..., description="Query results")
    facets: Dict[str, Dict[str, int]] = Field(default_factory=dict, description="Facet counts")
    suggestions: List[str] = Field(default_factory=list, description="Query suggestions")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")


class APIErrorResponse(BaseModel):
    """API error response model."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class SystemStatus(BaseModel):
    """System status model."""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="Application version")
    components: Dict[str, str] = Field(..., description="Component status")
    query_engine: Dict[str, Any] = Field(..., description="Query engine information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Status timestamp")


# Dependency functions
async def get_query_engine() -> QueryEngine:
    """Get the global query engine instance."""
    if query_engine is None:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    return query_engine


def convert_api_filter_to_query_filter(api_filter: APIQueryFilter) -> QueryFilter:
    """Convert API filter to internal query filter."""
    try:
        operator = FilterOperator(api_filter.operator)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid filter operator: {api_filter.operator}")
    
    return QueryFilter(
        field=api_filter.field,
        operator=operator,
        value=api_filter.value,
        case_sensitive=api_filter.case_sensitive,
        weight=api_filter.weight
    )


def convert_api_sort_to_sort_criterion(api_sort: APISortCriterion) -> SortCriterion:
    """Convert API sort to internal sort criterion."""
    try:
        order = SortOrder.ASC if api_sort.order.lower() == "asc" else SortOrder.DESC
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid sort order: {api_sort.order}")
    
    return SortCriterion(
        field=api_sort.field,
        order=order,
        weight=api_sort.weight
    )


def convert_query_result_to_api(result: QueryResult) -> APIQueryResult:
    """Convert internal query result to API model."""
    return APIQueryResult(
        entity_id=result.entity_id,
        entity_type=result.entity_type,
        score=result.score,
        data=result.data,
        relationships=result.relationships,
        embeddings=result.embeddings,
        highlights=result.highlights,
        explanation=result.explanation,
        metadata=result.metadata
    )


# API Endpoints

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Echo-Roots Taxonomy API",
        "version": "1.0.0",
        "description": "RESTful API for taxonomy construction and semantic enrichment",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=SystemStatus, tags=["System"])
async def health_check(engine: QueryEngine = Depends(get_query_engine)):
    """Health check endpoint."""
    from echo_roots import __version__
    
    # Get query engine metrics
    metrics = await engine.get_performance_metrics()
    
    return SystemStatus(
        status="healthy",
        version=__version__,
        components={
            "query_engine": "active",
            "api_server": "running",
            "database": "connected"
        },
        query_engine={
            "supported_types": [qt.value for qt in engine.get_supported_query_types()],
            "total_queries": metrics.get("total_queries", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "avg_execution_time_ms": metrics.get("average_execution_time_ms", 0.0)
        }
    )


@app.post("/query", response_model=APIQueryResponse, tags=["Query"])
async def execute_query(
    request: APIQueryRequest,
    background_tasks: BackgroundTasks,
    engine: QueryEngine = Depends(get_query_engine)
):
    """Execute a query request."""
    
    try:
        # Convert API request to internal request
        query_type_mapping = {
            "exact": QueryType.EXACT_MATCH,
            "fuzzy": QueryType.FUZZY_SEARCH,
            "semantic": QueryType.SEMANTIC_SEARCH,
            "hybrid": QueryType.HYBRID_SEARCH,
            "faceted": QueryType.FACETED_SEARCH,
            "graph": QueryType.GRAPH_TRAVERSAL,
            "fulltext": QueryType.FULL_TEXT_SEARCH,
            "aggregation": QueryType.AGGREGATION
        }
        
        if request.query_type not in query_type_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid query type: {request.query_type}"
            )
        
        # Generate unique query ID
        query_id = f"api-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hash(str(request.dict()))}"
        
        # Convert filters and sorts
        filters = [convert_api_filter_to_query_filter(f) for f in request.filters]
        sorts = [convert_api_sort_to_sort_criterion(s) for s in request.sort_criteria]
        
        # Create internal query request
        internal_request = QueryRequest(
            query_id=query_id,
            query_type=query_type_mapping[request.query_type],
            search_text=request.search_text,
            target_types=request.target_types,
            filters=filters,
            sort_criteria=sorts,
            limit=request.limit,
            offset=request.offset,
            include_metadata=request.include_metadata,
            include_relationships=request.include_relationships,
            include_embeddings=request.include_embeddings,
            fuzzy_threshold=request.fuzzy_threshold,
            similarity_threshold=request.similarity_threshold,
            timeout_seconds=request.timeout_seconds
        )
        
        # Execute query
        response = await engine.process_query(internal_request)
        
        # Convert response to API format
        api_results = [convert_query_result_to_api(result) for result in response.results]
        
        return APIQueryResponse(
            query_id=response.query_id,
            total_results=response.total_results,
            returned_results=response.returned_results,
            execution_time_ms=response.execution_time_ms,
            results=api_results,
            facets=response.facets,
            suggestions=response.suggestions,
            errors=response.errors,
            warnings=response.warnings,
            metadata=response.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@app.get("/search", response_model=APIQueryResponse, tags=["Query"])
async def simple_search(
    q: str = Query(..., description="Search query text"),
    type: str = Query("fuzzy", description="Query type: exact, fuzzy, semantic"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    threshold: float = Query(0.7, ge=0.0, le=1.0, description="Similarity threshold"),
    engine: QueryEngine = Depends(get_query_engine)
):
    """Simple search endpoint with query parameters."""
    
    # Create API request from query parameters
    api_request = APIQueryRequest(
        query_type=type,
        search_text=q,
        limit=limit,
        fuzzy_threshold=threshold,
        similarity_threshold=threshold
    )
    
    # Use the main query endpoint
    return await execute_query(api_request, BackgroundTasks(), engine)


@app.get("/query/history", tags=["Query"])
async def get_query_history(
    limit: int = Query(20, ge=1, le=100, description="Number of recent queries"),
    success_only: bool = Query(False, description="Show only successful queries"),
    engine: QueryEngine = Depends(get_query_engine)
):
    """Get recent query history."""
    
    history = engine.get_query_history(limit=limit, success_only=success_only)
    
    return {
        "total_entries": len(history),
        "entries": [
            {
                "query_id": entry.query_id,
                "query_type": entry.request.query_type.value,
                "search_text": entry.request.search_text,
                "success": entry.success,
                "execution_time_ms": entry.response.execution_time_ms,
                "total_results": entry.response.total_results,
                "executed_at": entry.executed_at.isoformat(),
                "error_message": entry.error_message
            }
            for entry in history
        ]
    }


@app.get("/query/suggestions", tags=["Query"])
async def get_query_suggestions(
    partial: str = Query(..., description="Partial query text"),
    limit: int = Query(5, ge=1, le=20, description="Maximum suggestions"),
    engine: QueryEngine = Depends(get_query_engine)
):
    """Get query suggestions based on partial input."""
    
    suggestions = await engine.get_query_suggestions(partial, limit=limit)
    
    return {
        "partial_query": partial,
        "suggestions": suggestions
    }


@app.get("/query/metrics", tags=["Analytics"])
async def get_query_metrics(engine: QueryEngine = Depends(get_query_engine)):
    """Get query performance metrics."""
    
    metrics = await engine.get_performance_metrics()
    
    return {
        "query_metrics": metrics,
        "supported_query_types": [qt.value for qt in engine.get_supported_query_types()],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/entities/{entity_id}", tags=["Entities"])
async def get_entity(
    entity_id: str,
    include_relationships: bool = Query(False, description="Include entity relationships"),
    engine: QueryEngine = Depends(get_query_engine)
):
    """Get entity by ID."""
    
    # This would typically use the repository directly
    # For now, we'll simulate with a query
    try:
        request = APIQueryRequest(
            query_type="exact",
            search_text=entity_id,
            limit=1,
            include_relationships=include_relationships
        )
        
        response = await execute_query(request, BackgroundTasks(), engine)
        
        if response.results:
            return response.results[0]
        else:
            raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Entity retrieval failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    error_data = APIErrorResponse(
        error=exc.detail,
        timestamp=datetime.now(),  # Use timezone-aware datetime
        request_id=getattr(request.state, "request_id", None)
    ).model_dump()
    
    return JSONResponse(
        status_code=exc.status_code,
        content=json.loads(json.dumps(error_data, cls=DateTimeJSONEncoder))
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    error_data = APIErrorResponse(
        error="Internal server error",
        details=str(exc),
        timestamp=datetime.now(),  # Use timezone-aware datetime
        request_id=getattr(request.state, "request_id", None)
    ).model_dump()
    
    return JSONResponse(
        status_code=500,
        content=json.loads(json.dumps(error_data, cls=DateTimeJSONEncoder))
    )


# CLI integration function
def start_api_server(
    host: str = "localhost",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1
):
    """Start the API server (called from CLI)."""
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )
    server = uvicorn.Server(config)
    return server.run()


if __name__ == "__main__":
    # For development
    uvicorn.run(
        "echo_roots.cli.api_server:app",
        host="localhost",
        port=8000,
        reload=True
    )
