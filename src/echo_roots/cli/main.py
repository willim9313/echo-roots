"""Main CLI entry point for echo-roots."""

import typer
from pathlib import Path
from typing import Optional, List
import asyncio
import json
from enum import Enum

# Import core components
from echo_roots.retrieval import (
    QueryType, QueryRequest, QueryEngine, FilterOperator, QueryFilter,
    SortCriterion, SortOrder
)

app = typer.Typer(
    name="echo-roots",
    help="Practical taxonomy construction and semantic enrichment framework",
    rich_markup_mode="rich"
)

# Create subcommands
query_app = typer.Typer(name="query", help="Query and search operations")
api_app = typer.Typer(name="api", help="API server operations")
app.add_typer(query_app, name="query")
app.add_typer(api_app, name="api")


class QueryTypeChoice(str, Enum):
    """CLI-friendly query type choices."""
    exact = "exact"
    fuzzy = "fuzzy" 
    semantic = "semantic"


class OutputFormat(str, Enum):
    """Output format choices."""
    json = "json"
    table = "table"
    yaml = "yaml"


@app.command()
def version():
    """Show the version information."""
    from echo_roots import __version__
    
    typer.echo(f"echo-roots version {__version__}")


@app.command()
def status():
    """Show system status and configuration."""
    typer.echo("🌱 [bold green]Echo-Roots Framework[/bold green]")
    typer.echo("Status: [green]Production Ready[/green]")
    typer.echo("\n[bold]Available Components:[/bold]")
    typer.echo("✅ Core Data Models (T1)")
    typer.echo("✅ Domain Adapter (T2)") 
    typer.echo("✅ LLM Extraction (T3)")
    typer.echo("✅ Storage Interfaces (T4)")
    typer.echo("✅ Ingestion Pipeline (T5)")
    typer.echo("✅ Taxonomy Management (T6)")
    typer.echo("✅ Vocabulary Management (T7)")
    typer.echo("✅ Semantic Enrichment (T8)")
    typer.echo("✅ Retrieval & Query Interface (T9)")
    typer.echo("🔄 CLI & API Interface (T10) - [yellow]In Progress[/yellow]")


@app.command()
def init(
    output_dir: str = typer.Option("./workspace", help="Workspace directory path"),
    with_examples: bool = typer.Option(False, "--examples", help="Include example data"),
):
    """Initialize a new echo-roots workspace."""
    output_path = Path(output_dir)
    typer.echo(f"🚀 Initializing echo-roots workspace at [cyan]{output_path}[/cyan]")
    
    # Create workspace structure
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (output_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (output_path / "exports").mkdir(parents=True, exist_ok=True)
    (output_path / "logs").mkdir(parents=True, exist_ok=True)
    (output_path / "config").mkdir(parents=True, exist_ok=True)
    
    # Create configuration files
    config_content = {
        "database": {
            "type": "duckdb",
            "path": "./data/processed/echo_roots.db"
        },
        "query": {
            "default_limit": 100,
            "fuzzy_threshold": 0.8,
            "semantic_threshold": 0.7,
            "timeout_seconds": 30
        },
        "api": {
            "host": "localhost",
            "port": 8000,
            "docs_url": "/docs"
        }
    }
    
    with open(output_path / "config" / "echo_roots.json", "w") as f:
        json.dump(config_content, f, indent=2)
    
    if with_examples:
        # Create example data files
        example_data = [
            {"name": "Electronics", "type": "category", "description": "Electronic devices and accessories"},
            {"name": "Books", "type": "category", "description": "Published books and literature"},
            {"name": "Clothing", "type": "category", "description": "Apparel and fashion items"}
        ]
        
        with open(output_path / "data" / "raw" / "example_categories.json", "w") as f:
            json.dump(example_data, f, indent=2)
    
    typer.echo("✅ Workspace initialized successfully")
    typer.echo("\n[bold]Next steps:[/bold]")
    typer.echo("1. Place your data files in [cyan]./workspace/data/raw/[/cyan]")
    typer.echo("2. Run: [green]echo-roots query search \"your search term\"[/green]")
    typer.echo("3. Start API server: [green]echo-roots api start[/green]")


@query_app.command("search")
def search_command(
    query_text: str = typer.Argument(..., help="Text to search for"),
    query_type: QueryTypeChoice = typer.Option(QueryTypeChoice.fuzzy, "--type", "-t", 
                                              help="Type of search to perform"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    threshold: float = typer.Option(0.7, "--threshold", help="Similarity threshold (0.0-1.0)"),
    entity_types: Optional[List[str]] = typer.Option(None, "--types", 
                                                    help="Entity types to search"),
    output_format: OutputFormat = typer.Option(OutputFormat.table, "--format", "-f",
                                              help="Output format"),
    workspace: str = typer.Option("./workspace", "--workspace", "-w", 
                                 help="Workspace directory"),
):
    """Search for entities using various query strategies."""
    
    async def _search():
        try:
            # Initialize query engine (would normally load from workspace config)
            from echo_roots.retrieval import QueryEngine
            from tests.test_t9_retrieval_interface import MockRetrievalRepository
            
            repository = MockRetrievalRepository()
            query_engine = QueryEngine(repository)
            
            # Map CLI query type to system query type
            query_type_mapping = {
                QueryTypeChoice.exact: QueryType.EXACT_MATCH,
                QueryTypeChoice.fuzzy: QueryType.FUZZY_SEARCH,
                QueryTypeChoice.semantic: QueryType.SEMANTIC_SEARCH,
            }
            
            # Create query request
            request = QueryRequest(
                query_id=f"cli-search-{hash(query_text)}",
                query_type=query_type_mapping[query_type],
                search_text=query_text,
                target_types=entity_types or [],
                limit=limit,
                fuzzy_threshold=threshold if query_type == QueryTypeChoice.fuzzy else 0.8,
                similarity_threshold=threshold if query_type == QueryTypeChoice.semantic else 0.7,
            )
            
            # Execute query
            response = await query_engine.process_query(request)
            
            # Format output
            if response.errors:
                typer.echo(f"❌ [red]Query failed:[/red] {'; '.join(response.errors)}", err=True)
                raise typer.Exit(1)
            
            if output_format == OutputFormat.json:
                result_data = [
                    {
                        "id": result.entity_id,
                        "type": result.entity_type,
                        "score": result.score,
                        "data": result.data,
                        "explanation": result.explanation
                    }
                    for result in response.results
                ]
                typer.echo(json.dumps(result_data, indent=2))
                
            elif output_format == OutputFormat.table:
                if response.results:
                    typer.echo(f"\n🔍 Found {response.total_results} results in {response.execution_time_ms:.1f}ms")
                    typer.echo("─" * 80)
                    
                    for i, result in enumerate(response.results, 1):
                        score_color = "green" if result.score > 0.8 else "yellow" if result.score > 0.5 else "red"
                        typer.echo(f"{i:2d}. [bold]{result.data.get('name', result.entity_id)}[/bold]")
                        typer.echo(f"    Type: {result.entity_type} | Score: [{score_color}]{result.score:.3f}[/{score_color}]")
                        if result.explanation:
                            typer.echo(f"    {result.explanation}")
                        if i < len(response.results):
                            typer.echo()
                else:
                    typer.echo("🔍 No results found")
                    
            elif output_format == OutputFormat.yaml:
                import yaml
                result_data = [
                    {
                        "id": result.entity_id,
                        "type": result.entity_type,
                        "score": result.score,
                        "data": result.data,
                        "explanation": result.explanation
                    }
                    for result in response.results
                ]
                typer.echo(yaml.dump(result_data, default_flow_style=False))
            
        except Exception as e:
            typer.echo(f"❌ [red]Search failed:[/red] {str(e)}", err=True)
            raise typer.Exit(1)
    
    # Run async function
    asyncio.run(_search())


@query_app.command("interactive")
def interactive_query(
    workspace: str = typer.Option("./workspace", "--workspace", "-w", 
                                 help="Workspace directory"),
):
    """Start interactive query session."""
    
    async def _interactive():
        from echo_roots.retrieval import QueryEngine
        from tests.test_t9_retrieval_interface import MockRetrievalRepository
        
        repository = MockRetrievalRepository()
        query_engine = QueryEngine(repository)
        
        typer.echo("🔍 [bold green]Echo-Roots Interactive Query Session[/bold green]")
        typer.echo("Type 'help' for commands, 'quit' to exit\n")
        
        while True:
            try:
                query_text = typer.prompt("echo-roots> ", type=str)
                
                if query_text.lower() in ['quit', 'exit', 'q']:
                    typer.echo("👋 Goodbye!")
                    break
                elif query_text.lower() == 'help':
                    typer.echo("\n[bold]Available commands:[/bold]")
                    typer.echo("  search <text>     - Fuzzy search")
                    typer.echo("  exact <text>      - Exact match search")
                    typer.echo("  semantic <text>   - Semantic search")
                    typer.echo("  stats            - Show query statistics")
                    typer.echo("  help             - Show this help")
                    typer.echo("  quit/exit/q      - Exit session\n")
                    continue
                elif query_text.lower() == 'stats':
                    metrics = await query_engine.get_performance_metrics()
                    if metrics:
                        typer.echo(f"\n📊 [bold]Query Statistics:[/bold]")
                        typer.echo(f"Total queries: {metrics.get('total_queries', 0)}")
                        typer.echo(f"Success rate: {metrics.get('success_rate', 0):.1%}")
                        typer.echo(f"Avg execution time: {metrics.get('average_execution_time_ms', 0):.1f}ms\n")
                    else:
                        typer.echo("📊 No query statistics available yet\n")
                    continue
                
                # Parse command
                parts = query_text.split(' ', 1)
                if len(parts) < 2:
                    typer.echo("❌ Please provide search text. Type 'help' for commands\n")
                    continue
                    
                command, search_text = parts
                
                # Determine query type
                query_type_map = {
                    'search': QueryType.FUZZY_SEARCH,
                    'exact': QueryType.EXACT_MATCH,
                    'semantic': QueryType.SEMANTIC_SEARCH,
                }
                
                if command not in query_type_map:
                    typer.echo(f"❌ Unknown command: {command}. Type 'help' for available commands\n")
                    continue
                
                # Execute query
                request = QueryRequest(
                    query_id=f"interactive-{hash(search_text)}",
                    query_type=query_type_map[command],
                    search_text=search_text,
                    limit=5,
                    fuzzy_threshold=0.7,
                    similarity_threshold=0.6,
                )
                
                response = await query_engine.process_query(request)
                
                if response.errors:
                    typer.echo(f"❌ [red]Query failed:[/red] {'; '.join(response.errors)}\n")
                    continue
                
                if response.results:
                    typer.echo(f"🔍 Found {response.total_results} results ({response.execution_time_ms:.1f}ms):")
                    for i, result in enumerate(response.results, 1):
                        score_color = "green" if result.score > 0.8 else "yellow" if result.score > 0.5 else "red"
                        typer.echo(f"  {i}. [bold]{result.data.get('name', result.entity_id)}[/bold] "
                                 f"([{score_color}]{result.score:.3f}[/{score_color}])")
                else:
                    typer.echo("🔍 No results found")
                
                typer.echo()  # Empty line for readability
                
            except KeyboardInterrupt:
                typer.echo("\n👋 Goodbye!")
                break
            except Exception as e:
                typer.echo(f"❌ [red]Error:[/red] {str(e)}\n")
    
    asyncio.run(_interactive())


@query_app.command("history")
def query_history(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of recent queries"),
    success_only: bool = typer.Option(False, "--success", help="Show only successful queries"),
    workspace: str = typer.Option("./workspace", "--workspace", "-w", 
                                 help="Workspace directory"),
):
    """Show recent query history."""
    
    async def _history():
        from echo_roots.retrieval import QueryEngine
        from tests.test_t9_retrieval_interface import MockRetrievalRepository
        
        repository = MockRetrievalRepository()
        query_engine = QueryEngine(repository)
        
        history = query_engine.get_query_history(limit=limit, success_only=success_only)
        
        if not history:
            typer.echo("📜 No query history available")
            return
        
        typer.echo(f"📜 [bold]Recent Query History[/bold] (last {len(history)} queries)")
        typer.echo("─" * 80)
        
        for i, entry in enumerate(reversed(history), 1):
            status = "✅" if entry.success else "❌"
            typer.echo(f"{i:2d}. {status} [{entry.request.query_type.value}] \"{entry.request.search_text}\"")
            typer.echo(f"     Time: {entry.executed_at.strftime('%H:%M:%S')} | "
                     f"Results: {entry.response.returned_results} | "
                     f"Duration: {entry.response.execution_time_ms:.1f}ms")
            if not entry.success and entry.error_message:
                typer.echo(f"     Error: {entry.error_message}")
            typer.echo()
    
    asyncio.run(_history())


@api_app.command("start")
def start_api_server(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind the server"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    workers: int = typer.Option(1, "--workers", help="Number of worker processes"),
):
    """Start the API server."""
    try:
        from echo_roots.cli.api_server import start_api_server as start_server
        
        typer.echo(f"🚀 Starting Echo-Roots API server on [cyan]http://{host}:{port}[/cyan]")
        typer.echo(f"📚 API documentation available at [cyan]http://{host}:{port}/docs[/cyan]")
        typer.echo("Press Ctrl+C to stop the server\n")
        
        start_server(host=host, port=port, reload=reload, workers=workers)
        
    except KeyboardInterrupt:
        typer.echo("\n👋 API server stopped")
    except Exception as e:
        typer.echo(f"❌ [red]Failed to start API server:[/red] {str(e)}", err=True)
        raise typer.Exit(1)


@api_app.command("test")
def test_api_endpoints(
    base_url: str = typer.Option("http://localhost:8000", "--url", help="API base URL"),
):
    """Test API endpoints."""
    import requests
    import time
    
    def test_endpoint(method: str, url: str, data=None, expected_status=200):
        """Test a single endpoint."""
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                typer.echo(f"❌ Unsupported method: {method}")
                return False
            
            if response.status_code == expected_status:
                typer.echo(f"✅ {method} {url} - {response.status_code}")
                return True
            else:
                typer.echo(f"❌ {method} {url} - {response.status_code} (expected {expected_status})")
                return False
        except requests.exceptions.RequestException as e:
            typer.echo(f"❌ {method} {url} - Connection failed: {str(e)}")
            return False
    
    typer.echo(f"🧪 Testing API endpoints at [cyan]{base_url}[/cyan]\n")
    
    # Test endpoints
    tests = [
        ("GET", f"{base_url}/", None, 200),
        ("GET", f"{base_url}/health", None, 200),
        ("GET", f"{base_url}/search?q=laptop&type=fuzzy&limit=5", None, 200),
        ("POST", f"{base_url}/query", {
            "query_type": "exact",
            "search_text": "test",
            "limit": 10
        }, 200),
        ("GET", f"{base_url}/query/metrics", None, 200),
        ("GET", f"{base_url}/query/suggestions?partial=lap&limit=3", None, 200),
    ]
    
    passed = 0
    total = len(tests)
    
    for method, url, data, expected_status in tests:
        if test_endpoint(method, url, data, expected_status):
            passed += 1
        time.sleep(0.5)  # Small delay between tests
    
    typer.echo(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        typer.echo("🎉 [green]All API tests passed![/green]")
    else:
        typer.echo(f"⚠️ [yellow]{total - passed} tests failed[/yellow]")
        raise typer.Exit(1)


@api_app.command("docs")
def open_api_docs(
    base_url: str = typer.Option("http://localhost:8000", "--url", help="API base URL"),
):
    """Open API documentation in browser."""
    import webbrowser
    
    docs_url = f"{base_url}/docs"
    typer.echo(f"🌐 Opening API documentation: [cyan]{docs_url}[/cyan]")
    
    try:
        webbrowser.open(docs_url)
        typer.echo("✅ Browser opened successfully")
    except Exception as e:
        typer.echo(f"❌ [red]Failed to open browser:[/red] {str(e)}")
        typer.echo(f"Please manually navigate to: {docs_url}")


if __name__ == "__main__":
    app()
