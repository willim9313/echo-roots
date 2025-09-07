"""
Storage interface protocols for the echo-roots taxonomy system.

This module defines the abstract storage interfaces that enable
pluggable storage backends while maintaining type safety and
consistency across the application.

Based on ADR-0001: Hybrid Storage Model
- DuckDB: Core ingestion, normalization, analytics
- Neo4j: Graph operations, hierarchy navigation  
- Qdrant: Vector search, semantic similarity

The interfaces provide a unified API while allowing specialized
backends to optimize for their strengths.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator, Protocol
from datetime import datetime
from uuid import UUID

from ..models.core import IngestionItem, ExtractionResult, ElevationProposal, Mapping
from ..models.taxonomy import Category, Attribute, SemanticCandidate
from ..models.domain import DomainPack


class StorageBackend(Protocol):
    """
    Protocol defining the core storage interface.
    
    All storage backends must implement this protocol to ensure
    consistent behavior across different storage technologies.
    """
    
    async def initialize(self) -> None:
        """Initialize the storage backend and create necessary structures."""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status and basic metrics."""
        ...
    
    async def close(self) -> None:
        """Clean up connections and resources."""
        ...


class IngestionRepository(Protocol):
    """Repository for managing raw ingestion data."""
    
    async def store_item(self, item: IngestionItem) -> str:
        """Store an ingestion item and return its ID."""
        ...
    
    async def get_item(self, item_id: str) -> Optional[IngestionItem]:
        """Retrieve an ingestion item by ID."""
        ...
    
    async def list_items(
        self, 
        source: Optional[str] = None,
        language: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[IngestionItem]:
        """List ingestion items with optional filtering."""
        ...
    
    async def update_status(self, item_id: str, status: str) -> bool:
        """Update the processing status of an item."""
        ...
    
    async def delete_item(self, item_id: str) -> bool:
        """Delete an ingestion item."""
        ...


class ExtractionRepository(Protocol):
    """Repository for managing LLM extraction results."""
    
    async def store_result(self, result: ExtractionResult) -> str:
        """Store an extraction result and return its ID."""
        ...
    
    async def get_result(self, result_id: str) -> Optional[ExtractionResult]:
        """Retrieve an extraction result by ID."""
        ...
    
    async def get_results_for_item(self, item_id: str) -> List[ExtractionResult]:
        """Get all extraction results for a specific ingestion item."""
        ...
    
    async def list_results(
        self,
        source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ExtractionResult]:
        """List extraction results with optional filtering."""
        ...
    
    async def delete_result(self, result_id: str) -> bool:
        """Delete an extraction result."""
        ...


class TaxonomyRepository(Protocol):
    """Repository for managing the canonical taxonomy."""
    
    async def store_category(self, category: Category) -> str:
        """Store a category and return its ID."""
        ...
    
    async def get_category(self, category_id: str) -> Optional[Category]:
        """Retrieve a category by ID."""
        ...
    
    async def get_category_by_name(self, name: str, domain: str) -> Optional[Category]:
        """Retrieve a category by name within a domain."""
        ...
    
    async def list_categories(
        self,
        domain: Optional[str] = None,
        parent_id: Optional[str] = None,
        level: Optional[int] = None
    ) -> List[Category]:
        """List categories with optional filtering."""
        ...
    
    async def store_attribute(self, attribute: Attribute) -> str:
        """Store an attribute and return its ID."""
        ...
    
    async def get_attributes_for_category(self, category_id: str) -> List[Attribute]:
        """Get all attributes associated with a category."""
        ...
    
    async def store_semantic_candidate(self, candidate: SemanticCandidate) -> str:
        """Store a semantic candidate and return its ID."""
        ...
    
    async def search_semantic_candidates(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10
    ) -> List[SemanticCandidate]:
        """Search semantic candidates by text similarity."""
        ...


class MappingRepository(Protocol):
    """Repository for managing domain mappings and transformations."""
    
    async def store_mapping(self, mapping: Mapping) -> str:
        """Store a mapping and return its ID."""
        ...
    
    async def get_mapping(self, mapping_id: str) -> Optional[Mapping]:
        """Retrieve a mapping by ID."""
        ...
    
    async def get_mappings_for_domain(self, domain: str) -> List[Mapping]:
        """Get all mappings for a specific domain."""
        ...
    
    async def apply_mapping(
        self, 
        mapping_id: str, 
        source_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a mapping transformation to source data."""
        ...


class ElevationRepository(Protocol):
    """Repository for managing elevation proposals and feedback."""
    
    async def store_proposal(self, proposal: ElevationProposal) -> str:
        """Store an elevation proposal and return its ID."""
        ...
    
    async def get_proposal(self, proposal_id: str) -> Optional[ElevationProposal]:
        """Retrieve an elevation proposal by ID."""
        ...
    
    async def list_proposals(
        self,
        status: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ElevationProposal]:
        """List elevation proposals with optional filtering."""
        ...
    
    async def update_proposal_status(
        self, 
        proposal_id: str, 
        status: str,
        reviewer_notes: Optional[str] = None
    ) -> bool:
        """Update the status of an elevation proposal."""
        ...


class AnalyticsRepository(Protocol):
    """Repository for analytics and reporting queries."""
    
    async def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a specific domain."""
        ...
    
    async def get_extraction_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get extraction pipeline metrics."""
        ...
    
    async def get_category_distribution(self, domain: str) -> Dict[str, int]:
        """Get category distribution within a domain."""
        ...
    
    async def get_quality_trends(
        self,
        domain: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get quality score trends over time."""
        ...


class StorageManager(ABC):
    """
    Abstract base class for storage management.
    
    Coordinates multiple storage backends and provides
    high-level repository interfaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize all configured storage backends."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all storage backends."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close all storage connections."""
        pass
    
    # Repository access methods
    @property
    @abstractmethod
    def ingestion(self) -> IngestionRepository:
        """Access to ingestion data repository."""
        pass
    
    @property
    @abstractmethod
    def extraction(self) -> ExtractionRepository:
        """Access to extraction results repository."""
        pass
    
    @property
    @abstractmethod
    def taxonomy(self) -> TaxonomyRepository:
        """Access to taxonomy repository."""
        pass
    
    @property
    @abstractmethod
    def mappings(self) -> MappingRepository:
        """Access to mappings repository."""
        pass
    
    @property
    @abstractmethod
    def elevation(self) -> ElevationRepository:
        """Access to elevation repository."""
        pass
    
    @property
    @abstractmethod
    def analytics(self) -> AnalyticsRepository:
        """Access to analytics repository."""
        pass


class TransactionContext(Protocol):
    """Protocol for transaction management across repositories."""
    
    async def __aenter__(self):
        """Begin transaction."""
        ...
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Commit or rollback transaction."""
        ...
    
    async def commit(self) -> None:
        """Explicitly commit the transaction."""
        ...
    
    async def rollback(self) -> None:
        """Explicitly rollback the transaction."""
        ...


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class ConnectionError(StorageError):
    """Raised when storage backend connection fails."""
    pass


class IntegrityError(StorageError):
    """Raised when data integrity constraints are violated."""
    pass


class NotFoundError(StorageError):
    """Raised when requested entity is not found."""
    pass


class ConflictError(StorageError):
    """Raised when operation conflicts with existing data."""
    pass
