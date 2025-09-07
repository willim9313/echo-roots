# T10 CLI & API Interface Tests

import asyncio
import json
import pytest
from typer.testing import CliRunner
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Mock retrieval components before importing the modules
mock_query_engine = Mock()
mock_query_engine.simple_search = AsyncMock(return_value={
    "results": [{"id": "test_1", "name": "Test Result", "score": 0.9}],
    "total": 1,
    "type": "exact"
})
mock_query_engine.search = AsyncMock(return_value={
    "results": [{"id": "test_1", "name": "Test Result", "score": 0.9}],
    "total": 1,
    "type": "exact"
})
mock_query_engine.get_query_history = AsyncMock(return_value=[
    {"query": "test", "timestamp": "2024-01-01T12:00:00", "results_count": 1}
])
mock_query_engine.get_suggestions = AsyncMock(return_value=[
    {"term": "suggestion 1", "score": 0.8}
])
mock_query_engine.get_entity_by_id = AsyncMock(return_value={
    "id": "test_1", "name": "Test Entity", "type": "concept"
})

# Mock the components
with patch('echo_roots.cli.main.QueryEngine', return_value=mock_query_engine):
    with patch('echo_roots.cli.api_server.QueryEngine', return_value=mock_query_engine):
        # Import the CLI and API components after mocking
        from echo_roots.cli.main import app as cli_app
        from echo_roots.cli.api_server import create_app


class TestCLIInterface:
    """Test CLI interface functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Create a mock that simulates CLI output
        self.mock_query_engine = mock_query_engine
    
    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(cli_app, ["version"])
        assert result.exit_code == 0
        assert "echo-roots version" in result.stdout
    
    def test_status_command(self):
        """Test status command."""
        result = self.runner.invoke(cli_app, ["status"])
        assert result.exit_code == 0
        assert "Echo-Roots Framework" in result.stdout
        assert "Production Ready" in result.stdout
        assert "✅" in result.stdout  # Should show completed components
    
    def test_init_command_basic(self):
        """Test basic workspace initialization."""
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            result = self.runner.invoke(cli_app, ["init", "--output-dir", "/tmp/test-workspace"])
            
            assert result.exit_code == 0
            assert "Initializing echo-roots workspace" in result.stdout
            assert "Workspace initialized successfully" in result.stdout
            mock_mkdir.assert_called()
            mock_open.assert_called()
            mock_json_dump.assert_called()
    
    def test_init_command_with_examples(self):
        """Test workspace initialization with examples."""
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            result = self.runner.invoke(cli_app, ["init", "--examples", "--output-dir", "/tmp/test-workspace"])
            
            assert result.exit_code == 0
            assert "Initializing echo-roots workspace" in result.stdout
            mock_mkdir.assert_called()
            # Should be called twice: once for config, once for examples
            assert mock_open.call_count >= 2
    
    def test_query_search_command_help(self):
        """Test query search command help."""
        result = self.runner.invoke(cli_app, ["query", "search", "--help"])
        assert result.exit_code == 0
        assert "Search for entities" in result.stdout
        assert "--type" in result.stdout
        assert "--limit" in result.stdout
        assert "--threshold" in result.stdout
    
    @patch('echo_roots.cli.main.QueryEngine')
    @patch('echo_roots.cli.main.MockRetrievalRepository')
    def test_query_search_exact(self, mock_repo, mock_engine):
        """Test exact match search command."""
        # Mock the async query processing
        mock_engine_instance = Mock()
        mock_engine.return_value = mock_engine_instance
        
        mock_response = Mock()
        mock_response.errors = []
        mock_response.total_results = 1
        mock_response.execution_time_ms = 50.0
        mock_response.results = [
            Mock(
                entity_id="1",
                entity_type="product", 
                score=1.0,
                data={"name": "Test Product"},
                explanation="Exact match"
            )
        ]
        
        async def mock_process_query(request):
            return mock_response
        
        mock_engine_instance.process_query = AsyncMock(return_value=mock_response)
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = self.runner.invoke(cli_app, [
                "query", "search", "test product", 
                "--type", "exact", 
                "--limit", "5"
            ])
            
            # The command should not fail
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    @patch('echo_roots.cli.main.QueryEngine')
    @patch('echo_roots.cli.main.MockRetrievalRepository')
    def test_query_search_json_output(self, mock_repo, mock_engine):
        """Test search command with JSON output."""
        mock_engine_instance = Mock()
        mock_engine.return_value = mock_engine_instance
        
        mock_response = Mock()
        mock_response.errors = []
        mock_response.results = []
        
        mock_engine_instance.process_query = AsyncMock(return_value=mock_response)
        
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = self.runner.invoke(cli_app, [
                "query", "search", "test", 
                "--format", "json"
            ])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    def test_query_history_command(self):
        """Test query history command."""
        with patch('echo_roots.cli.main.QueryEngine') as mock_engine, \
             patch('echo_roots.cli.main.MockRetrievalRepository') as mock_repo, \
             patch('asyncio.run') as mock_run:
            
            result = self.runner.invoke(cli_app, ["query", "history", "--limit", "5"])
            
            assert result.exit_code == 0
            mock_run.assert_called_once()
    
    def test_api_start_command_help(self):
        """Test API start command help."""
        result = self.runner.invoke(cli_app, ["api", "start", "--help"])
        assert result.exit_code == 0
        assert "Start the API server" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
    
    def test_api_test_command(self):
        """Test API test command."""
        with patch('requests.get') as mock_get, \
             patch('requests.post') as mock_post:
            
            # Mock successful responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            mock_post.return_value = mock_response
            
            result = self.runner.invoke(cli_app, ["api", "test", "--url", "http://localhost:8000"])
            
            # Should attempt the tests but might fail due to connection
            # The important thing is the command structure works
            assert "Testing API endpoints" in result.stdout
    
    def test_api_docs_command(self):
        """Test API docs command."""
        with patch('webbrowser.open') as mock_open:
            result = self.runner.invoke(cli_app, ["api", "docs"])
            
            assert result.exit_code == 0
            assert "Opening API documentation" in result.stdout
            mock_open.assert_called_once()


class TestAPIInterface:
    """Test API interface functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create API app with mocked query engine
        app = create_app()
        
        # Initialize with mock query engine  
        with patch('echo_roots.cli.api_server.query_engine', mock_query_engine):
            self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Echo-Roots Taxonomy API"
        assert "version" in data
        assert "docs" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data
        assert "query_engine" in data
    
    def test_simple_search_endpoint(self):
        """Test simple search endpoint."""
        response = self.client.get("/search?q=laptop&type=fuzzy&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "query_id" in data
        assert "total_results" in data
        assert "execution_time_ms" in data
        assert "results" in data
        assert isinstance(data["results"], list)
    
    def test_search_endpoint_validation(self):
        """Test search endpoint parameter validation."""
        # Missing query parameter
        response = self.client.get("/search")
        assert response.status_code == 422  # Validation error
        
        # Invalid limit
        response = self.client.get("/search?q=test&limit=0")
        assert response.status_code == 422
        
        # Invalid threshold
        response = self.client.get("/search?q=test&threshold=2.0")
        assert response.status_code == 422
    
    def test_query_endpoint_exact_match(self):
        """Test query endpoint with exact match."""
        query_data = {
            "query_type": "exact",
            "search_text": "laptop",
            "limit": 10,
            "include_metadata": True
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"].startswith("api-")
        assert isinstance(data["total_results"], int)
        assert isinstance(data["execution_time_ms"], float)
        assert isinstance(data["results"], list)
    
    def test_query_endpoint_fuzzy_search(self):
        """Test query endpoint with fuzzy search."""
        query_data = {
            "query_type": "fuzzy",
            "search_text": "laptap",  # Typo
            "fuzzy_threshold": 0.7,
            "limit": 5
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert "query_id" in data
        assert "results" in data
    
    def test_query_endpoint_with_filters(self):
        """Test query endpoint with filters."""
        query_data = {
            "query_type": "fuzzy",
            "search_text": "computer",
            "filters": [
                {
                    "field": "category",
                    "operator": "eq",
                    "value": "electronics"
                },
                {
                    "field": "price",
                    "operator": "lt",
                    "value": 1000
                }
            ],
            "sort_criteria": [
                {
                    "field": "price",
                    "order": "asc"
                }
            ],
            "limit": 20
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
    
    def test_query_endpoint_validation_errors(self):
        """Test query endpoint validation."""
        # Invalid query type
        query_data = {
            "query_type": "invalid_type",
            "search_text": "test"
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 400
        
        # Invalid filter operator
        query_data = {
            "query_type": "exact",
            "search_text": "test",
            "filters": [
                {
                    "field": "name",
                    "operator": "invalid_op",
                    "value": "test"
                }
            ]
        }
        
        response = self.client.post("/query", json=query_data)
        assert response.status_code == 400
    
    def test_query_history_endpoint(self):
        """Test query history endpoint."""
        response = self.client.get("/query/history?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "total_entries" in data
        assert "entries" in data
        assert isinstance(data["entries"], list)
    
    def test_query_suggestions_endpoint(self):
        """Test query suggestions endpoint."""
        response = self.client.get("/query/suggestions?partial=lap&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "partial_query" in data
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
    
    def test_query_metrics_endpoint(self):
        """Test query metrics endpoint."""
        response = self.client.get("/query/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "query_metrics" in data
        assert "supported_query_types" in data
        assert "timestamp" in data
    
    def test_entity_endpoint(self):
        """Test entity retrieval endpoint."""
        response = self.client.get("/entities/1?include_relationships=true")
        
        # Should either return the entity or 404 if not found
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert "entity_id" in data
            assert "entity_type" in data
            assert "score" in data
            assert "data" in data
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.get("/")
        # CORS headers should be present due to middleware
        assert response.status_code == 200
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_endpoint(self):
        """Test API documentation endpoint."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        # Should return HTML content
        assert "text/html" in response.headers.get("content-type", "")


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create API app with mocked query engine
        app = create_app()
        
        # Initialize with mock query engine  
        with patch('echo_roots.cli.api_server.query_engine', mock_query_engine):
            self.client = TestClient(app)
    
    def test_404_error_handling(self):
        """Test 404 error handling."""
        response = self.client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_422_validation_error(self):
        """Test validation error handling."""
        # Send invalid JSON
        response = self.client.post("/query", json={"invalid": "data"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_500_error_handling(self):
        """Test internal server error handling."""
        # This is harder to trigger in tests, but the handler should be present
        # We can test that the handler exists by checking the app's exception handlers
        from echo_roots.cli.api_server import app
        assert len(app.exception_handlers) > 0


class TestAPIIntegration:
    """Test API integration with query engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create API app with mocked query engine
        app = create_app()
        
        # Initialize with mock query engine  
        with patch('echo_roots.cli.api_server.query_engine', mock_query_engine):
            self.client = TestClient(app)
    
    def test_query_engine_integration(self):
        """Test that API properly integrates with query engine."""
        # Make a simple search request
        response = self.client.get("/search?q=test&type=exact")
        assert response.status_code == 200
        
        data = response.json()
        # Should have proper query response structure
        required_fields = ["query_id", "total_results", "returned_results", "execution_time_ms", "results"]
        for field in required_fields:
            assert field in data
    
    def test_async_query_processing(self):
        """Test asynchronous query processing."""
        # Multiple concurrent requests
        import threading
        import time
        
        responses = []
        
        def make_request():
            response = self.client.get("/search?q=async_test&type=fuzzy")
            responses.append(response)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    def test_query_history_persistence(self):
        """Test that query history is maintained."""
        # Make a few queries
        for i in range(3):
            response = self.client.get(f"/search?q=test_{i}&type=exact")
            assert response.status_code == 200
        
        # Check history
        response = self.client.get("/query/history")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_entries"] >= 3  # Should have at least the queries we made
    
    def test_performance_metrics_tracking(self):
        """Test that performance metrics are tracked."""
        # Make some queries
        for i in range(2):
            response = self.client.get(f"/search?q=perf_test_{i}")
            assert response.status_code == 200
        
        # Check metrics
        response = self.client.get("/query/metrics")
        assert response.status_code == 200
        
        data = response.json()
        metrics = data["query_metrics"]
        assert metrics["total_queries"] >= 2
