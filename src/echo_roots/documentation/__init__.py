# T12 Documentation & Knowledge Management Implementation

"""
Comprehensive documentation system with automated generation, knowledge management,
interactive help, and integrated learning resources for the Echo-Roots taxonomy framework.
"""

import asyncio
import json
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ast
import inspect
import importlib
import markdown
from jinja2 import Template, Environment, FileSystemLoader
import subprocess

# Rich components for beautiful output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.tree import Tree


class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"
    TROUBLESHOOTING = "troubleshooting"


class ContentFormat(Enum):
    """Supported content formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    YAML = "yaml"


@dataclass
class DocumentSection:
    """Represents a section within a document."""
    title: str
    content: str
    level: int = 1
    subsections: List['DocumentSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert section to markdown format."""
        header = "#" * self.level
        md_content = f"{header} {self.title}\n\n{self.content}\n\n"
        
        for subsection in self.subsections:
            md_content += subsection.to_markdown()
        
        return md_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [sub.to_dict() for sub in self.subsections],
            "metadata": self.metadata
        }


@dataclass
class Document:
    """Represents a complete document."""
    title: str
    doc_type: DocumentationType
    sections: List[DocumentSection]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    authors: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: DocumentSection):
        """Add a section to the document."""
        self.sections.append(section)
        self.updated_at = datetime.now()
    
    def to_markdown(self) -> str:
        """Convert entire document to markdown."""
        header = f"# {self.title}\n\n"
        header += f"**Version:** {self.version}  \n"
        header += f"**Updated:** {self.updated_at.strftime('%Y-%m-%d')}  \n"
        
        if self.authors:
            header += f"**Authors:** {', '.join(self.authors)}  \n"
        
        if self.tags:
            header += f"**Tags:** {', '.join(self.tags)}  \n"
        
        header += "\n---\n\n"
        
        content = ""
        for section in self.sections:
            content += section.to_markdown()
        
        return header + content
    
    def to_html(self) -> str:
        """Convert document to HTML."""
        md_content = self.to_markdown()
        html = markdown.markdown(md_content, extensions=['codehilite', 'tables', 'toc'])
        
        # Wrap in basic HTML structure
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; }}
        pre {{ background: #f6f8fa; padding: 16px; border-radius: 6px; overflow-x: auto; }}
        code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html}
</body>
</html>
        """
        return html_template


class CodeAnalyzer:
    """Analyzes Python code to extract documentation information."""
    
    def __init__(self):
        self.console = Console()
    
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a Python module and extract documentation info."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            module_info = {
                "path": str(module_path),
                "docstring": ast.get_docstring(tree),
                "classes": [],
                "functions": [],
                "imports": [],
                "constants": []
            }
            
            # Process top-level nodes only for module-level functions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node)
                    module_info["classes"].append(class_info)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):  # Skip private functions
                        func_info = self._analyze_function(node)
                        module_info["functions"].append(func_info)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        module_info["imports"].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            module_info["imports"].append(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.Assign):
                    # Look for module-level constants
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            module_info["constants"].append(target.id)
            
            return module_info
            
        except Exception as e:
            self.console.print(f"Error analyzing {module_path}: {e}")
            return {}
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition."""
        class_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "attributes": [],
            "inheritance": [base.id for base in node.bases if isinstance(base, ast.Name)]
        }
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._analyze_function(item)
                class_info["methods"].append(method_info)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info["attributes"].append(target.id)
        
        return class_info
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze a function definition."""
        func_info = {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [],
            "returns": None,
            "decorators": []
        }
        
        # Analyze arguments
        for arg in node.args.args:
            func_info["args"].append(arg.arg)
        
        # Check for return type annotation
        if node.returns:
            if isinstance(node.returns, ast.Name):
                func_info["returns"] = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                func_info["returns"] = str(node.returns.value)
        
        # Analyze decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                func_info["decorators"].append(decorator.id)
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                func_info["decorators"].append(decorator.func.id)
        
        return func_info


class DocumentGenerator:
    """Generates documentation from various sources."""
    
    def __init__(self):
        self.console = Console()
        self.code_analyzer = CodeAnalyzer()
        self.template_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))
    
    def generate_api_reference(self, module_paths: List[Path]) -> Document:
        """Generate API reference documentation from Python modules."""
        doc = Document(
            title="Echo-Roots API Reference",
            doc_type=DocumentationType.API_REFERENCE,
            sections=[],
            authors=["Echo-Roots Team"],
            tags=["api", "reference", "python"]
        )
        
        # Introduction section
        intro_section = DocumentSection(
            title="Overview",
            content="This document provides a comprehensive reference for the Echo-Roots taxonomy framework API.",
            level=2
        )
        doc.add_section(intro_section)
        
        # Analyze each module
        for module_path in module_paths:
            module_info = self.code_analyzer.analyze_module(module_path)
            if module_info:
                module_section = self._create_module_section(module_info)
                doc.add_section(module_section)
        
        return doc
    
    def _create_module_section(self, module_info: Dict[str, Any]) -> DocumentSection:
        """Create a documentation section for a module."""
        module_name = Path(module_info["path"]).stem
        
        content = ""
        if module_info.get("docstring"):
            content += f"{module_info['docstring']}\n\n"
        
        content += f"**Module Path:** `{module_info['path']}`\n\n"
        
        section = DocumentSection(
            title=f"Module: {module_name}",
            content=content,
            level=2
        )
        
        # Add classes subsection
        if module_info.get("classes"):
            classes_content = ""
            for class_info in module_info["classes"]:
                classes_content += self._format_class_docs(class_info)
            
            classes_section = DocumentSection(
                title="Classes",
                content=classes_content,
                level=3
            )
            section.subsections.append(classes_section)
        
        # Add functions subsection
        if module_info.get("functions"):
            functions_content = ""
            for func_info in module_info["functions"]:
                functions_content += self._format_function_docs(func_info)
            
            functions_section = DocumentSection(
                title="Functions",
                content=functions_content,
                level=3
            )
            section.subsections.append(functions_section)
        
        return section
    
    def _format_class_docs(self, class_info: Dict[str, Any]) -> str:
        """Format class documentation."""
        content = f"#### `{class_info['name']}`\n\n"
        
        if class_info.get("docstring"):
            content += f"{class_info['docstring']}\n\n"
        
        if class_info.get("inheritance"):
            content += f"**Inherits from:** {', '.join(class_info['inheritance'])}\n\n"
        
        # Methods
        if class_info.get("methods"):
            content += "**Methods:**\n\n"
            for method in class_info["methods"]:
                content += f"- `{method['name']}({', '.join(method['args'])})`"
                if method.get("docstring"):
                    content += f": {method['docstring'].split('.')[0]}"
                content += "\n"
            content += "\n"
        
        return content
    
    def _format_function_docs(self, func_info: Dict[str, Any]) -> str:
        """Format function documentation."""
        args_str = ', '.join(func_info['args'])
        content = f"#### `{func_info['name']}({args_str})`\n\n"
        
        if func_info.get("docstring"):
            content += f"{func_info['docstring']}\n\n"
        
        if func_info.get("decorators"):
            content += f"**Decorators:** {', '.join(func_info['decorators'])}\n\n"
        
        if func_info.get("returns"):
            content += f"**Returns:** `{func_info['returns']}`\n\n"
        
        return content
    
    def generate_user_guide(self) -> Document:
        """Generate comprehensive user guide."""
        doc = Document(
            title="Echo-Roots User Guide",
            doc_type=DocumentationType.USER_GUIDE,
            sections=[],
            authors=["Echo-Roots Team"],
            tags=["user-guide", "tutorial", "getting-started"]
        )
        
        # Getting Started
        getting_started = DocumentSection(
            title="Getting Started",
            content="""
Welcome to Echo-Roots! This guide will help you get up and running with the taxonomy framework.

## What is Echo-Roots?

Echo-Roots is a comprehensive taxonomy construction and semantic enrichment framework designed for building, managing, and querying complex knowledge structures.

## Key Features

- **Modular Architecture**: Flexible component system for custom workflows
- **Semantic Search**: Advanced query capabilities with multiple search strategies
- **Domain Integration**: Support for domain-specific taxonomies
- **API Access**: Full REST API and CLI interface
- **Governance**: Built-in monitoring, access control, and audit logging
- **Extensible**: Plugin architecture for custom functionality
            """,
            level=2
        )
        
        # Installation
        installation = DocumentSection(
            title="Installation",
            content="""
## Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

## Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/echo-roots.git
cd echo-roots

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install echo-roots
pip install -e .
```

## Verify Installation

```bash
echo-roots version
echo-roots status
```
            """,
            level=2
        )
        
        # Basic Usage
        basic_usage = DocumentSection(
            title="Basic Usage",
            content="""
## Initialize Workspace

```bash
echo-roots init --output-dir ./my-taxonomy
cd my-taxonomy
```

## Search and Query

```bash
# Simple search
echo-roots query search "machine learning"

# Advanced search with filters
echo-roots query search "AI" --type fuzzy --limit 10 --format json

# Interactive mode
echo-roots query interactive
```

## API Server

```bash
# Start API server
echo-roots api start --port 8000

# Test API
curl http://localhost:8000/health
curl "http://localhost:8000/search?q=artificial+intelligence"
```

## System Administration

```bash
# Check system status
echo-roots governance status

# Monitor performance
echo-roots governance metrics

# View audit logs
echo-roots governance audit --limit 20
```
            """,
            level=2
        )
        
        doc.add_section(getting_started)
        doc.add_section(installation)
        doc.add_section(basic_usage)
        
        return doc
    
    def generate_developer_guide(self) -> Document:
        """Generate developer guide with architecture and extension information."""
        doc = Document(
            title="Echo-Roots Developer Guide",
            doc_type=DocumentationType.DEVELOPER_GUIDE,
            sections=[],
            authors=["Echo-Roots Team"],
            tags=["developer", "architecture", "api", "extension"]
        )
        
        # Architecture Overview
        architecture = DocumentSection(
            title="Architecture Overview",
            content="""
Echo-Roots follows a modular, layered architecture designed for scalability and extensibility.

## Core Components

### T0-T2: Foundation Layer
- **Storage**: DuckDB-based storage with pluggable backends
- **Models**: Core data structures and validation
- **Configuration**: YAML-based configuration management

### T3-T5: Processing Layer  
- **Pipelines**: Data ingestion, validation, and extraction
- **Domain Integration**: Domain-specific adapters and mergers
- **Semantic Processing**: Graph analytics and enrichment

### T6-T8: Intelligence Layer
- **Taxonomy Management**: Hierarchical structure management
- **Vocabulary**: Term and concept management
- **Semantic Search**: Advanced search and similarity

### T9: Query Layer
- **Retrieval Interface**: Unified query processing
- **Multiple Search Types**: Exact, fuzzy, semantic matching
- **Performance Optimization**: Caching and optimization

### T10-T11: Interface Layer
- **CLI**: Rich command-line interface
- **REST API**: Full HTTP API with OpenAPI docs
- **Governance**: Monitoring, access control, audit logging

### T12: Knowledge Layer
- **Documentation**: Automated doc generation
- **Knowledge Management**: Learning resources and guides
            """,
            level=2
        )
        
        # Extension Development
        extension_dev = DocumentSection(
            title="Extension Development",
            content="""
## Creating Custom Components

### Custom Search Strategies

```python
from echo_roots.retrieval import SearchStrategy

class CustomSearchStrategy(SearchStrategy):
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        # Implement custom search logic
        pass
```

### Custom Domain Adapters

```python
from echo_roots.domain import DomainAdapter

class MyDomainAdapter(DomainAdapter):
    def load_domain_data(self, source: Path) -> Dict[str, Any]:
        # Implement domain-specific loading
        pass
```

### Custom Pipeline Components

```python
from echo_roots.pipelines import PipelineComponent

class CustomProcessor(PipelineComponent):
    async def process(self, data: Any) -> Any:
        # Implement custom processing
        pass
```

## API Integration

### Authentication

```python
from echo_roots.governance import governance_manager

# Authenticate requests
authorized, user = await governance_manager.authorize_request(api_key, "read_access")
```

### Custom Endpoints

```python
from fastapi import FastAPI
from echo_roots.cli.api_server import app

@app.get("/custom/endpoint")
async def custom_endpoint():
    return {"message": "Custom functionality"}
```
            """,
            level=2
        )
        
        doc.add_section(architecture)
        doc.add_section(extension_dev)
        
        return doc
    
    def generate_changelog(self, version_history: List[Dict[str, Any]]) -> Document:
        """Generate changelog from version history."""
        doc = Document(
            title="Echo-Roots Changelog",
            doc_type=DocumentationType.CHANGELOG,
            sections=[],
            authors=["Echo-Roots Team"],
            tags=["changelog", "releases", "history"]
        )
        
        for version_info in version_history:
            version_section = DocumentSection(
                title=f"Version {version_info['version']}",
                content=f"**Release Date:** {version_info['date']}\n\n{version_info['description']}\n\n",
                level=2
            )
            
            if version_info.get('features'):
                features_content = "### ✨ New Features\n\n"
                for feature in version_info['features']:
                    features_content += f"- {feature}\n"
                features_content += "\n"
                
                features_subsection = DocumentSection(
                    title="New Features",
                    content=features_content,
                    level=3
                )
                version_section.subsections.append(features_subsection)
            
            if version_info.get('fixes'):
                fixes_content = "### 🐛 Bug Fixes\n\n"
                for fix in version_info['fixes']:
                    fixes_content += f"- {fix}\n"
                fixes_content += "\n"
                
                fixes_subsection = DocumentSection(
                    title="Bug Fixes", 
                    content=fixes_content,
                    level=3
                )
                version_section.subsections.append(fixes_subsection)
            
            doc.add_section(version_section)
        
        return doc


class KnowledgeBase:
    """Manages knowledge base with searchable documentation and learning resources."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.documents: Dict[str, Document] = {}
        self.index: Dict[str, List[str]] = {}  # Simple keyword index
        self.console = Console()
        
        # Create base directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "generated").mkdir(exist_ok=True)
        (self.base_path / "static").mkdir(exist_ok=True)
        (self.base_path / "templates").mkdir(exist_ok=True)
    
    def add_document(self, doc_id: str, document: Document):
        """Add a document to the knowledge base."""
        self.documents[doc_id] = document
        self._update_index(doc_id, document)
        self._save_document(doc_id, document)
    
    def _update_index(self, doc_id: str, document: Document):
        """Update search index with document content."""
        # Simple keyword extraction
        content = document.to_markdown().lower()
        words = re.findall(r'\b\w+\b', content)
        
        for word in words:
            if len(word) > 3:  # Skip short words
                if word not in self.index:
                    self.index[word] = []
                if doc_id not in self.index[word]:
                    self.index[word].append(doc_id)
    
    def _save_document(self, doc_id: str, document: Document):
        """Save document to disk in multiple formats."""
        # Save as markdown
        md_path = self.base_path / "generated" / f"{doc_id}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(document.to_markdown())
        
        # Save as HTML
        html_path = self.base_path / "generated" / f"{doc_id}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(document.to_html())
        
        # Save metadata as JSON
        metadata_path = self.base_path / "generated" / f"{doc_id}.json"
        metadata = {
            "title": document.title,
            "type": document.doc_type.value,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "version": document.version,
            "authors": document.authors,
            "tags": document.tags,
            "metadata": document.metadata
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Document, float]]:
        """Search the knowledge base."""
        query_words = re.findall(r'\b\w+\b', query.lower())
        doc_scores: Dict[str, float] = {}
        
        for word in query_words:
            # Search in index (already lowercase)
            if word in self.index:
                for doc_id in self.index[word]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
            
            # Also search document content directly for case-insensitive matching
            for doc_id, doc in self.documents.items():
                doc_content = doc.to_markdown().lower()
                if word in doc_content:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 0.5  # Lower weight for content match
        
        # Sort by score and return top results
        results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            if doc_id in self.documents:
                results.append((doc_id, self.documents[doc_id], score))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[Tuple[str, Document]]:
        """List all documents."""
        return list(self.documents.items())
    
    def generate_index_page(self) -> str:
        """Generate HTML index page for the knowledge base."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Echo-Roots Knowledge Base</title>
    <meta charset="utf-8">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; }
        .doc-card { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 6px; }
        .doc-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
        .doc-meta { color: #666; font-size: 0.9em; }
        .doc-tags { margin-top: 10px; }
        .tag { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; margin-right: 5px; font-size: 0.8em; }
    </style>
</head>
<body>
    <h1>📚 Echo-Roots Knowledge Base</h1>
    <p>Comprehensive documentation and learning resources for the Echo-Roots taxonomy framework.</p>
    
    <h2>📖 Available Documents</h2>
        """
        
        for doc_id, doc in self.documents.items():
            html += f"""
    <div class="doc-card">
        <div class="doc-title">
            <a href="{doc_id}.html">{doc.title}</a>
        </div>
        <div class="doc-meta">
            Type: {doc.doc_type.value} | 
            Updated: {doc.updated_at.strftime('%Y-%m-%d')} |
            Version: {doc.version}
        </div>
        <div class="doc-tags">
            {''.join(f'<span class="tag">{tag}</span>' for tag in doc.tags)}
        </div>
    </div>
            """
        
        html += """
</body>
</html>
        """
        
        # Save index page
        index_path = self.base_path / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return html


class DocumentationManager:
    """Main documentation management system."""
    
    def __init__(self, docs_path: Path = None):
        self.docs_path = docs_path or Path("docs")
        self.knowledge_base = KnowledgeBase(self.docs_path / "knowledge_base")
        self.generator = DocumentGenerator()
        self.console = Console()
    
    def initialize(self):
        """Initialize the documentation system."""
        self.docs_path.mkdir(parents=True, exist_ok=True)
        
        # Create default structure
        (self.docs_path / "api").mkdir(exist_ok=True)
        (self.docs_path / "guides").mkdir(exist_ok=True)
        (self.docs_path / "tutorials").mkdir(exist_ok=True)
        (self.docs_path / "architecture").mkdir(exist_ok=True)
        
        self.console.print("✅ [green]Documentation system initialized[/green]")
    
    def generate_all_docs(self):
        """Generate comprehensive documentation set."""
        self.console.print("🔄 [blue]Generating documentation...[/blue]")
        
        # Generate API reference
        src_path = Path("src/echo_roots")
        if src_path.exists():
            python_files = list(src_path.rglob("*.py"))
            api_doc = self.generator.generate_api_reference(python_files)
            self.knowledge_base.add_document("api_reference", api_doc)
            self.console.print("✅ API Reference generated")
        
        # Generate user guide
        user_guide = self.generator.generate_user_guide()
        self.knowledge_base.add_document("user_guide", user_guide)
        self.console.print("✅ User Guide generated")
        
        # Generate developer guide
        dev_guide = self.generator.generate_developer_guide()
        self.knowledge_base.add_document("developer_guide", dev_guide)
        self.console.print("✅ Developer Guide generated")
        
        # Generate changelog
        version_history = [
            {
                "version": "1.0.0",
                "date": "2025-09-07",
                "description": "Initial release of Echo-Roots taxonomy framework",
                "features": [
                    "Complete T0-T12 implementation",
                    "CLI and API interfaces",
                    "Governance and monitoring",
                    "Documentation system"
                ]
            }
        ]
        changelog = self.generator.generate_changelog(version_history)
        self.knowledge_base.add_document("changelog", changelog)
        self.console.print("✅ Changelog generated")
        
        # Generate index page
        self.knowledge_base.generate_index_page()
        self.console.print("✅ Index page generated")
        
        self.console.print(f"\n📚 [green]Documentation generated in:[/green] {self.docs_path}")
    
    def search_docs(self, query: str, limit: int = 5) -> List[Tuple[str, Document, float]]:
        """Search documentation."""
        return self.knowledge_base.search(query, limit)
    
    def get_doc_stats(self) -> Dict[str, Any]:
        """Get documentation statistics."""
        docs = self.knowledge_base.list_documents()
        
        stats = {
            "total_documents": len(docs),
            "by_type": {},
            "total_sections": 0,
            "last_updated": None
        }
        
        for doc_id, doc in docs:
            doc_type = doc.doc_type.value
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            stats["total_sections"] += len(doc.sections)
            
            if stats["last_updated"] is None or doc.updated_at > stats["last_updated"]:
                stats["last_updated"] = doc.updated_at
        
        return stats


# Global documentation manager instance
documentation_manager = DocumentationManager()


class InteractiveHelp:
    """Interactive help system for CLI and API."""
    
    def __init__(self):
        self.console = Console()
    
    def show_command_help(self, command: str):
        """Show detailed help for a specific command."""
        help_content = {
            "query": """
# Query Commands

The query commands provide access to the Echo-Roots search and retrieval system.

## Available Commands

### `search`
Search the taxonomy with various strategies.

**Usage:**
```bash
echo-roots query search "machine learning" --type exact --limit 10
```

**Options:**
- `--type`: Search type (exact, fuzzy, semantic)
- `--limit`: Maximum results to return
- `--format`: Output format (table, json, yaml)
- `--include-relationships`: Include entity relationships

### `interactive`
Start an interactive query session.

**Usage:**
```bash
echo-roots query interactive
```

### `history`
View recent query history.

**Usage:**
```bash
echo-roots query history --limit 20
```
            """,
            "api": """
# API Commands

Commands for managing the Echo-Roots API server.

## Available Commands

### `start`
Start the API server.

**Usage:**
```bash
echo-roots api start --port 8000 --host localhost
```

**Options:**
- `--port`: Server port (default: 8000)
- `--host`: Server host (default: localhost)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes

### `test`
Test API connectivity.

**Usage:**
```bash
echo-roots api test --url http://localhost:8000
```

### `docs`
Open API documentation in browser.

**Usage:**
```bash
echo-roots api docs
```
            """,
            "governance": """
# Governance Commands

System administration and monitoring commands.

## Available Commands

### `status`
Show system governance dashboard.

**Usage:**
```bash
echo-roots governance status
```

### `metrics`
Display current system metrics.

**Usage:**
```bash
echo-roots governance metrics
```

### `alerts`
Manage system alerts.

**Usage:**
```bash
echo-roots governance alerts --severity critical
```

### `users`
View user accounts and access control.

**Usage:**
```bash
echo-roots governance users
```

### `audit`
View audit logs.

**Usage:**
```bash
echo-roots governance audit --user admin --limit 50
```
            """
        }
        
        if command in help_content:
            self.console.print(Panel(
                Markdown(help_content[command]),
                title=f"Help: {command}",
                border_style="blue"
            ))
        else:
            self.console.print(f"❌ [red]No help available for command:[/red] {command}")
    
    def show_topic_help(self, topic: str):
        """Show help for a specific topic."""
        topics = {
            "getting-started": """
# Getting Started with Echo-Roots

## Quick Start

1. **Initialize a workspace:**
   ```bash
   echo-roots init --output-dir ./my-taxonomy
   ```

2. **Search the taxonomy:**
   ```bash
   echo-roots query search "artificial intelligence"
   ```

3. **Start the API server:**
   ```bash
   echo-roots api start
   ```

4. **Check system status:**
   ```bash
   echo-roots governance status
   ```

## Next Steps

- Explore the interactive query mode: `echo-roots query interactive`
- Read the full documentation: `echo-roots docs generate`
- Check the API documentation: `echo-roots api docs`
            """,
            "configuration": """
# Configuration

Echo-Roots uses YAML configuration files for customization.

## Default Configuration Locations

- Workspace: `./workspace/config.yaml`
- User: `~/.echo-roots/config.yaml`
- System: `/etc/echo-roots/config.yaml`

## Configuration Structure

```yaml
# Database configuration
storage:
  backend: "duckdb"
  connection_string: "./data/echo_roots.db"

# Query configuration
query:
  default_limit: 10
  cache_enabled: true
  timeout_seconds: 30

# API configuration
api:
  host: "localhost"
  port: 8000
  cors_enabled: true

# Governance configuration
governance:
  audit_logging: true
  metrics_collection: true
  session_timeout_hours: 24
```

## Environment Variables

- `ECHO_ROOTS_CONFIG`: Path to configuration file
- `ECHO_ROOTS_DATA_DIR`: Data directory path
- `ECHO_ROOTS_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
            """
        }
        
        if topic in topics:
            self.console.print(Panel(
                Markdown(topics[topic]),
                title=f"Help: {topic}",
                border_style="green"
            ))
        else:
            available_topics = ", ".join(topics.keys())
            self.console.print(f"❌ [red]Unknown topic:[/red] {topic}")
            self.console.print(f"Available topics: {available_topics}")


# Global interactive help instance
interactive_help = InteractiveHelp()


if __name__ == "__main__":
    # Demo usage
    print("Echo-Roots Documentation & Knowledge Management System")
    print("=====================================================")
    
    # Initialize documentation system
    doc_manager = DocumentationManager()
    doc_manager.initialize()
    
    # Generate all documentation
    doc_manager.generate_all_docs()
    
    # Show statistics
    stats = doc_manager.get_doc_stats()
    print(f"\nDocumentation Statistics:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Sections: {stats['total_sections']}")
    print(f"Last Updated: {stats['last_updated']}")
    
    print("\nDocumentation system ready!")
