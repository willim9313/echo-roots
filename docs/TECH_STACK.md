# Technology Stack

## Core Environment
- **Python 3.13+**: Modern Python with enhanced error messages and performance
- **UV Package Manager**: Fast, modern dependency management
- **Pydantic v2**: Type-safe data validation and serialization
- **FastAPI**: High-performance REST API framework
- **Typer**: Modern CLI framework with rich output

## Principles
- Support Chinese + English (simplified & traditional).
- Avoid rigid ontology; prefer property graphs + controlled vocab.
- Python-first for pipelines and APIs.
- Maintain raw + normalized data in parallel.

## Storage & Graph
- **Neo4j**: core graph (A & C).
- **Qdrant**: semantic candidate retrieval (D).
- **NetworkX**: prototyping & in-memory analysis.
- **DuckDB**: central structured store (core DB) for raw and normalized data.  
  - Stores ingestion inputs, preprocessed attributes, and evaluation tables.  
  - Optimized for Parquet snapshots and analytical queries.  
  - Acts as the staging layer before data flows into graph/vector stores.
- Optional: OpenSearch/ES for keyword search .

## Models & AI
- **Embeddings**: sentence-transformers (multilingual support)
- **LLM Integration**: OpenAI API (GPT-4/3.5-turbo)
- **Framework**: Custom semantic enrichment engine
- **Utilities**: tiktoken (token budgeting), DeepEval (evaluation)

## Development & Quality
- **Testing**: pytest with comprehensive coverage
- **Linting**: Ruff (fast Python linter and formatter)
- **Type Checking**: MyPy with strict configuration
- **CI/CD**: GitHub Actions with automated testing
- **Pre-commit**: Quality gates before commits

## Services & Interfaces
- **REST API**: FastAPI with comprehensive endpoints (`/query`, `/search`, `/governance`)
- **CLI Interface**: Typer-based commands with rich formatting
- **Governance System**: User management, audit logging, system monitoring
- **Documentation**: Auto-generated API docs and interactive knowledge base
- **Export**: Multiple formats (JSON, CSV, Parquet) with datetime handling
