"""Echo-Roots: Practical taxonomy construction and semantic enrichment framework.

This package provides tools for building, managing, and evolving taxonomies
with controlled vocabularies and semantic layers across domains like e-commerce,
media, and knowledge graphs.

Requirements:
- Python 3.13+
- FastAPI for REST API endpoints
- DuckDB for core storage
- OpenAI for LLM integration (optional)

Key components:
- Domain adapters for flexible input mapping
- A/C/D taxonomy framework (Classification/Controlled/Dynamic)
- Multi-storage backend support (DuckDB core + optional Neo4j/Qdrant)
- LLM-powered attribute extraction and normalization
- Governance workflows for taxonomy evolution
- CLI and REST API interfaces
"""

__version__ = "1.0.0"
__python_requires__ = ">=3.13"
__author__ = "Echo-Roots Contributors"

# Core exports for public API
from echo_roots.models.core import (
    AttributeExtraction,
    ElevationProposal,
    ExtractionResult,
    IngestionItem,
    Mapping,
    SemanticTerm,
)
from echo_roots.models.domain import DomainPack
from echo_roots.models.taxonomy import (
    Attribute,
    Category,
    SemanticCandidate,
)

__all__ = [
    "__version__",
    "__python_requires__",
    "__author__",
    # Core models
    "IngestionItem",
    "ExtractionResult",
    "AttributeExtraction",
    "SemanticTerm",
    "ElevationProposal",
    "Mapping",
    # Taxonomy models
    "Category",
    "Attribute",
    "SemanticCandidate",
    # Domain models
    "DomainPack",
]
