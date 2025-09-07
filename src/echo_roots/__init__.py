"""Echo-Roots: Practical taxonomy construction and semantic enrichment framework.

This package provides tools for building, managing, and evolving taxonomies
with controlled vocabularies and semantic layers across domains like e-commerce,
media, and knowledge graphs.

Key components:
- Domain adapters for flexible input mapping
- A/C/D taxonomy framework (Classification/Controlled/Dynamic)
- Multi-storage backend support (DuckDB core + optional Neo4j/Qdrant)
- LLM-powered attribute extraction and normalization
- Governance workflows for taxonomy evolution
"""

__version__ = "0.1.0"
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
