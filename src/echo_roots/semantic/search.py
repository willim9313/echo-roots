# T8 Semantic Enrichment Engine - Search and Ranking Components

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable, TYPE_CHECKING
import numpy as np
import logging
from collections import defaultdict
import asyncio
import json
import heapq

from . import (
    SemanticEmbedding, SemanticRelationship, SemanticConcept,
    SemanticQuery, SemanticSearchResult, SemanticRelationType,
    ConfidenceLevel, EmbeddingProvider, SemanticRepository
)

if TYPE_CHECKING:
    from ..storage.interfaces import SemanticVectorRepository

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Different semantic search strategies."""
    VECTOR_SIMILARITY = "vector_similarity"        # Pure embedding similarity
    HYBRID_SEMANTIC = "hybrid_semantic"           # Combine embeddings + relationships
    GRAPH_TRAVERSAL = "graph_traversal"           # Relationship graph exploration
    CONTEXTUAL_EXPANSION = "contextual_expansion" # Query expansion with context
    MULTI_MODAL = "multi_modal"                   # Multiple embedding models


class RankingStrategy(str, Enum):
    """Ranking strategies for search results."""
    SIMILARITY_SCORE = "similarity_score"         # Pure similarity ranking
    POPULARITY_WEIGHTED = "popularity_weighted"   # Weight by entity popularity
    FRESHNESS_WEIGHTED = "freshness_weighted"     # Weight by recency
    DIVERSITY_OPTIMIZED = "diversity_optimized"   # Maximize result diversity
    CONFIDENCE_WEIGHTED = "confidence_weighted"   # Weight by confidence scores


class SearchScope(str, Enum):
    """Scope of semantic search."""
    GLOBAL = "global"           # Search across all entities
    DOMAIN_SPECIFIC = "domain"  # Search within specific domain
    TYPE_SPECIFIC = "type"      # Search within entity types
    CONTEXTUAL = "contextual"   # Search within context/session


@dataclass
class SearchConfiguration:
    """Configuration for semantic search operations."""
    strategy: SearchStrategy = SearchStrategy.HYBRID_SEMANTIC
    ranking: RankingStrategy = RankingStrategy.CONFIDENCE_WEIGHTED
    scope: SearchScope = SearchScope.GLOBAL
    similarity_threshold: float = 0.5
    max_results: int = 20
    max_expansion_terms: int = 5
    diversity_threshold: float = 0.8
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    relationship_weights: Dict[SemanticRelationType, float] = field(default_factory=lambda: {
        SemanticRelationType.SYNONYM: 1.0,
        SemanticRelationType.HYPONYM: 0.9,
        SemanticRelationType.HYPERNYM: 0.9,
        SemanticRelationType.SIMILAR: 0.8,
        SemanticRelationType.RELATED: 0.6,
        SemanticRelationType.MERONYM: 0.7,
        SemanticRelationType.HOLONYM: 0.7,
        SemanticRelationType.ASSOCIATION: 0.5
    })


@dataclass
class RankingFactors:
    """Factors used in result ranking."""
    similarity_score: float
    popularity_score: float = 0.0
    freshness_score: float = 0.0
    confidence_score: float = 0.0
    diversity_penalty: float = 0.0
    relationship_boost: float = 0.0
    domain_relevance: float = 0.0
    final_score: float = 0.0


@dataclass
class SearchContext:
    """Context for semantic search session."""
    session_id: str
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    search_history: List[str] = field(default_factory=list)
    domain_focus: Optional[str] = None
    entity_type_preferences: List[str] = field(default_factory=list)
    previous_results: List[str] = field(default_factory=list)
    feedback_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchMetrics:
    """Metrics for search performance tracking."""
    query_time_ms: float
    total_candidates: int
    filtered_candidates: int
    final_results: int
    cache_hit: bool = False
    strategy_used: SearchStrategy = SearchStrategy.VECTOR_SIMILARITY
    ranking_used: RankingStrategy = RankingStrategy.SIMILARITY_SCORE
    expansion_terms: List[str] = field(default_factory=list)
    relationship_hops: int = 0


class QueryExpander:
    """Expands search queries with related terms."""
    
    def __init__(self, repository: SemanticRepository, embedding_provider: EmbeddingProvider):
        self.repository = repository
        self.embedding_provider = embedding_provider
    
    async def expand_query(
        self, 
        query: str, 
        max_expansions: int = 5,
        expansion_threshold: float = 0.7
    ) -> Tuple[str, List[str]]:
        """Expand query with semantically related terms."""
        # Generate embedding for original query
        query_embedding = await self.embedding_provider.generate_embedding(query)
        
        # Find similar entities
        similar_embeddings = await self.repository.find_similar_embeddings(
            query_embedding, limit=max_expansions * 2, threshold=expansion_threshold
        )
        
        expansion_terms = []
        for embedding, similarity in similar_embeddings:
            if similarity > expansion_threshold:
                # Extract key terms from similar entities
                source_text = embedding.metadata.get("source_text", "")
                if source_text and source_text.lower() not in query.lower():
                    expansion_terms.append(source_text)
        
        # Limit expansion terms
        expansion_terms = expansion_terms[:max_expansions]
        
        # Create expanded query
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"
        else:
            expanded_query = query
        
        return expanded_query, expansion_terms
    
    async def expand_with_relationships(
        self, 
        query_entities: List[str],
        max_hops: int = 2,
        max_expansions: int = 10
    ) -> List[str]:
        """Expand query using relationship graph traversal."""
        expanded_entities = set(query_entities)
        current_entities = set(query_entities)
        
        for hop in range(max_hops):
            next_entities = set()
            
            for entity_id in current_entities:
                # Get relationships for this entity
                relationships = await self.repository.get_relationships(entity_id)
                
                for rel in relationships:
                    # Add related entities based on relationship strength
                    weight = self._get_relationship_weight(rel.relationship_type)
                    if weight > 0.5 and rel.confidence_score > 0.6:
                        if rel.target_entity_id not in expanded_entities:
                            next_entities.add(rel.target_entity_id)
                            expanded_entities.add(rel.target_entity_id)
                        
                        if len(expanded_entities) >= max_expansions:
                            break
                
                if len(expanded_entities) >= max_expansions:
                    break
            
            current_entities = next_entities
            if not current_entities:
                break
        
        return list(expanded_entities)
    
    def _get_relationship_weight(self, rel_type: SemanticRelationType) -> float:
        """Get weight for relationship type in expansion."""
        weights = {
            SemanticRelationType.SYNONYM: 1.0,
            SemanticRelationType.HYPONYM: 0.8,
            SemanticRelationType.HYPERNYM: 0.8,
            SemanticRelationType.SIMILAR: 0.7,
            SemanticRelationType.RELATED: 0.5,
            SemanticRelationType.MERONYM: 0.6,
            SemanticRelationType.HOLONYM: 0.6,
            SemanticRelationType.ASSOCIATION: 0.4,
            SemanticRelationType.ANTONYM: 0.0
        }
        return weights.get(rel_type, 0.3)


class ResultRanker:
    """Ranks search results using multiple factors."""
    
    def __init__(self, repository: SemanticRepository):
        self.repository = repository
    
    async def rank_results(
        self, 
        results: List[SemanticSearchResult], 
        config: SearchConfiguration,
        context: Optional[SearchContext] = None
    ) -> List[Tuple[SemanticSearchResult, RankingFactors]]:
        """Rank search results using configured strategy."""
        ranked_results = []
        
        for result in results:
            factors = await self._calculate_ranking_factors(result, config, context)
            ranked_results.append((result, factors))
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x[1].final_score, reverse=True)
        
        # Apply diversity optimization if enabled
        if config.ranking == RankingStrategy.DIVERSITY_OPTIMIZED:
            ranked_results = await self._optimize_diversity(ranked_results, config)
        
        return ranked_results
    
    async def _calculate_ranking_factors(
        self, 
        result: SemanticSearchResult, 
        config: SearchConfiguration,
        context: Optional[SearchContext] = None
    ) -> RankingFactors:
        """Calculate ranking factors for a single result."""
        factors = RankingFactors(similarity_score=result.similarity_score)
        
        # Base similarity score
        factors.similarity_score = result.similarity_score
        
        # Popularity score (based on relationship count)
        relationships = await self.repository.get_relationships(result.entity_id)
        factors.popularity_score = min(1.0, len(relationships) / 10.0)
        
        # Freshness score (based on creation/update time)
        factors.freshness_score = self._calculate_freshness_score(result)
        
        # Confidence score (from metadata or relationships)
        factors.confidence_score = self._extract_confidence_score(result)
        
        # Relationship boost (if result came from relationships)
        if result.relationship_type:
            rel_weight = config.relationship_weights.get(result.relationship_type, 0.5)
            factors.relationship_boost = rel_weight * 0.2  # 20% boost maximum
        
        # Domain relevance (if context specifies domain focus)
        if context and context.domain_focus:
            factors.domain_relevance = self._calculate_domain_relevance(result, context)
        
        # User feedback integration
        if context and result.entity_id in context.feedback_scores:
            feedback_score = context.feedback_scores[result.entity_id]
            factors.final_score *= (1.0 + feedback_score * 0.1)  # 10% impact per feedback point
        
        # Calculate final score based on strategy
        factors.final_score = self._calculate_final_score(factors, config)
        
        return factors
    
    def _calculate_freshness_score(self, result: SemanticSearchResult) -> float:
        """Calculate freshness score based on recency."""
        # This would need timestamp information from metadata
        # Placeholder implementation
        return 0.5
    
    def _extract_confidence_score(self, result: SemanticSearchResult) -> float:
        """Extract confidence score from result metadata."""
        return result.metadata.get("confidence_score", result.similarity_score)
    
    def _calculate_domain_relevance(
        self, 
        result: SemanticSearchResult, 
        context: SearchContext
    ) -> float:
        """Calculate domain relevance score."""
        if not context.domain_focus:
            return 0.5
        
        # Simple domain matching - could be enhanced
        entity_text = result.entity_text.lower()
        domain_terms = context.domain_focus.lower().split()
        
        matches = sum(1 for term in domain_terms if term in entity_text)
        return min(1.0, matches / len(domain_terms))
    
    def _calculate_final_score(
        self, 
        factors: RankingFactors, 
        config: SearchConfiguration
    ) -> float:
        """Calculate final ranking score based on strategy."""
        if config.ranking == RankingStrategy.SIMILARITY_SCORE:
            return factors.similarity_score
        
        elif config.ranking == RankingStrategy.POPULARITY_WEIGHTED:
            return (factors.similarity_score * 0.7 + 
                   factors.popularity_score * 0.3 + 
                   factors.relationship_boost)
        
        elif config.ranking == RankingStrategy.FRESHNESS_WEIGHTED:
            return (factors.similarity_score * 0.6 + 
                   factors.freshness_score * 0.3 + 
                   factors.confidence_score * 0.1)
        
        elif config.ranking == RankingStrategy.CONFIDENCE_WEIGHTED:
            return (factors.similarity_score * 0.5 + 
                   factors.confidence_score * 0.3 + 
                   factors.popularity_score * 0.1 + 
                   factors.domain_relevance * 0.1 + 
                   factors.relationship_boost)
        
        else:  # Default to balanced scoring
            return (factors.similarity_score * 0.4 + 
                   factors.popularity_score * 0.2 + 
                   factors.confidence_score * 0.2 + 
                   factors.freshness_score * 0.1 + 
                   factors.domain_relevance * 0.1 + 
                   factors.relationship_boost)
    
    async def _optimize_diversity(
        self, 
        ranked_results: List[Tuple[SemanticSearchResult, RankingFactors]], 
        config: SearchConfiguration
    ) -> List[Tuple[SemanticSearchResult, RankingFactors]]:
        """Optimize result diversity by reducing similar results."""
        if not ranked_results or config.diversity_threshold >= 1.0:
            return ranked_results
        
        diverse_results = []
        
        for result, factors in ranked_results:
            # Check similarity with already selected results
            is_diverse = True
            
            for selected_result, _ in diverse_results:
                similarity = await self._calculate_result_similarity(
                    result, selected_result
                )
                
                if similarity > config.diversity_threshold:
                    # Apply diversity penalty
                    factors.diversity_penalty = similarity * 0.3
                    factors.final_score *= (1.0 - factors.diversity_penalty)
                    
                    # Skip if too similar to existing results
                    if similarity > 0.9:
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append((result, factors))
        
        # Re-sort after diversity optimization
        diverse_results.sort(key=lambda x: x[1].final_score, reverse=True)
        return diverse_results
    
    async def _calculate_result_similarity(
        self, 
        result1: SemanticSearchResult, 
        result2: SemanticSearchResult
    ) -> float:
        """Calculate similarity between two search results."""
        # Simple text-based similarity for now
        # Could be enhanced with embedding comparison
        text1_words = set(result1.entity_text.lower().split())
        text2_words = set(result2.entity_text.lower().split())
        
        if not text1_words or not text2_words:
            return 0.0
        
        intersection = len(text1_words & text2_words)
        union = len(text1_words | text2_words)
        
        return intersection / union if union > 0 else 0.0


class SemanticSearchEngine:
    """Advanced semantic search engine with multiple strategies."""
    
    def __init__(
        self, 
        repository: SemanticRepository,
        embedding_provider: EmbeddingProvider,
        vector_repository: Optional['SemanticVectorRepository'] = None
    ):
        self.repository = repository
        self.embedding_provider = embedding_provider
        self.vector_repository = vector_repository
        self.query_expander = QueryExpander(repository, embedding_provider)
        self.result_ranker = ResultRanker(repository)
        self.search_cache = {}
    
    async def search(
        self, 
        query: SemanticQuery, 
        config: Optional[SearchConfiguration] = None,
        context: Optional[SearchContext] = None
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Perform semantic search with specified strategy."""
        start_time = datetime.now(UTC)
        
        if not config:
            config = SearchConfiguration()
        
        # Check cache
        cache_key = self._generate_cache_key(query, config)
        if config.enable_caching and cache_key in self.search_cache:
            cached_result, cached_time = self.search_cache[cache_key]
            if (datetime.now(UTC) - cached_time).seconds < config.cache_ttl_seconds:
                metrics = SearchMetrics(
                    query_time_ms=0.0,
                    total_candidates=len(cached_result),
                    filtered_candidates=len(cached_result),
                    final_results=len(cached_result),
                    cache_hit=True,
                    strategy_used=config.strategy
                )
                return cached_result, metrics
        
        # Perform search based on strategy
        results, metrics = await self._execute_search_strategy(query, config, context)
        
        # Rank results
        ranked_results = await self.result_ranker.rank_results(results, config, context)
        final_results = [result for result, factors in ranked_results[:config.max_results]]
        
        # Update metrics
        end_time = datetime.now(UTC)
        metrics.query_time_ms = (end_time - start_time).total_seconds() * 1000
        metrics.final_results = len(final_results)
        metrics.ranking_used = config.ranking
        
        # Cache results
        if config.enable_caching:
            self.search_cache[cache_key] = (final_results, datetime.now(UTC))
        
        # Update search context
        if context:
            context.search_history.append(query.query_text)
            context.previous_results.extend([r.entity_id for r in final_results])
        
        return final_results, metrics
    
    async def _execute_search_strategy(
        self, 
        query: SemanticQuery, 
        config: SearchConfiguration,
        context: Optional[SearchContext] = None
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Execute search based on configured strategy."""
        metrics = SearchMetrics(
            query_time_ms=0.0,
            total_candidates=0,
            filtered_candidates=0,
            final_results=0,
            strategy_used=config.strategy
        )
        
        if config.strategy == SearchStrategy.VECTOR_SIMILARITY:
            return await self._vector_similarity_search(query, config, metrics)
        
        elif config.strategy == SearchStrategy.HYBRID_SEMANTIC:
            return await self._hybrid_semantic_search(query, config, metrics, context)
        
        elif config.strategy == SearchStrategy.GRAPH_TRAVERSAL:
            return await self._graph_traversal_search(query, config, metrics)
        
        elif config.strategy == SearchStrategy.CONTEXTUAL_EXPANSION:
            return await self._contextual_expansion_search(query, config, metrics, context)
        
        elif config.strategy == SearchStrategy.MULTI_MODAL:
            return await self._multi_modal_search(query, config, metrics)
        
        else:
            # Default to vector similarity
            return await self._vector_similarity_search(query, config, metrics)
    
    async def _vector_similarity_search(
        self, 
        query: SemanticQuery, 
        config: SearchConfiguration,
        metrics: SearchMetrics
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Pure vector similarity search using best available backend."""
        # Generate query embedding
        query_embedding = await self.embedding_provider.generate_embedding(query.query_text)
        
        similar_embeddings = []
        
        # Prefer Qdrant vector repository if available
        if self.vector_repository:
            try:
                # Use Qdrant for high-performance vector search
                similar_embeddings = await self.vector_repository.find_similar_embeddings(
                    query_vector=query_embedding,
                    limit=config.max_results * 2,
                    threshold=config.similarity_threshold,
                    entity_types=query.target_entity_types,
                    domains=query.filters.get("domains") if query.filters else None,
                    active_only=True
                )
                logger.debug(f"Used Qdrant vector search: {len(similar_embeddings)} results")
            except Exception as e:
                logger.warning(f"Qdrant vector search failed, falling back to repository: {e}")
                # Fallback to standard repository
                similar_embeddings = await self.repository.find_similar_embeddings(
                    query_embedding,
                    limit=config.max_results * 2,
                    threshold=config.similarity_threshold,
                    entity_types=query.target_entity_types
                )
        else:
            # Use standard repository
            similar_embeddings = await self.repository.find_similar_embeddings(
                query_embedding,
                limit=config.max_results * 2,
                threshold=config.similarity_threshold,
                entity_types=query.target_entity_types
            )
        
        metrics.total_candidates = len(similar_embeddings)
        metrics.filtered_candidates = len(similar_embeddings)
        
        # Convert to search results
        results = []
        for embedding, similarity in similar_embeddings:
            result = SemanticSearchResult(
                entity_id=embedding.entity_id,
                entity_type=embedding.entity_type,
                entity_text=embedding.metadata.get("source_text", ""),
                similarity_score=similarity,
                explanation=f"Vector similarity: {similarity:.3f}",
                metadata=embedding.metadata
            )
            results.append(result)
        
        return results, metrics
    
    async def _hybrid_semantic_search(
        self, 
        query: SemanticQuery, 
        config: SearchConfiguration,
        metrics: SearchMetrics,
        context: Optional[SearchContext] = None
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Combine vector similarity with relationship information."""
        # Start with vector similarity
        vector_results, vector_metrics = await self._vector_similarity_search(
            query, config, metrics
        )
        
        # Expand with relationship-based results
        relationship_results = []
        
        for result in vector_results[:10]:  # Top 10 vector results
            # Get relationships for this entity
            relationships = await self.repository.get_relationships(result.entity_id)
            
            for rel in relationships:
                if rel.confidence_score > 0.5:
                    # Get related entity embedding
                    related_embedding = await self.repository.get_embedding(rel.target_entity_id)
                    
                    if related_embedding:
                        # Calculate relationship-weighted similarity
                        rel_weight = config.relationship_weights.get(rel.relationship_type, 0.5)
                        weighted_similarity = result.similarity_score * rel_weight * rel.confidence_score
                        
                        if weighted_similarity > config.similarity_threshold:
                            rel_result = SemanticSearchResult(
                                entity_id=rel.target_entity_id,
                                entity_type=related_embedding.entity_type,
                                entity_text=related_embedding.metadata.get("source_text", ""),
                                similarity_score=weighted_similarity,
                                relationship_type=rel.relationship_type,
                                explanation=f"Related via {rel.relationship_type.value}: {weighted_similarity:.3f}",
                                metadata=related_embedding.metadata
                            )
                            relationship_results.append(rel_result)
        
        # Combine and deduplicate results
        all_results = vector_results + relationship_results
        seen_entities = set()
        unique_results = []
        
        for result in all_results:
            if result.entity_id not in seen_entities:
                unique_results.append(result)
                seen_entities.add(result.entity_id)
        
        metrics.total_candidates = len(all_results)
        metrics.filtered_candidates = len(unique_results)
        
        return unique_results, metrics
    
    async def _graph_traversal_search(
        self, 
        query: SemanticQuery, 
        config: SearchConfiguration,
        metrics: SearchMetrics
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Search using relationship graph traversal."""
        # Start with initial vector search to find seed entities
        seed_results, seed_metrics = await self._vector_similarity_search(
            query, SearchConfiguration(max_results=5), metrics
        )
        
        if not seed_results:
            return [], metrics
        
        # Expand through relationship graph
        expanded_entities = await self.query_expander.expand_with_relationships(
            [r.entity_id for r in seed_results],
            max_hops=2,
            max_expansions=config.max_results * 2
        )
        
        metrics.relationship_hops = 2
        
        # Get embeddings for expanded entities
        expanded_results = []
        for entity_id in expanded_entities:
            embedding = await self.repository.get_embedding(entity_id)
            if embedding:
                # Calculate propagated similarity
                base_similarity = 0.5  # Default for graph-traversed entities
                
                result = SemanticSearchResult(
                    entity_id=entity_id,
                    entity_type=embedding.entity_type,
                    entity_text=embedding.metadata.get("source_text", ""),
                    similarity_score=base_similarity,
                    explanation="Found via relationship graph traversal",
                    metadata=embedding.metadata
                )
                expanded_results.append(result)
        
        metrics.total_candidates = len(expanded_entities)
        metrics.filtered_candidates = len(expanded_results)
        
        return expanded_results, metrics
    
    async def _contextual_expansion_search(
        self, 
        query: SemanticQuery, 
        config: SearchConfiguration,
        metrics: SearchMetrics,
        context: Optional[SearchContext] = None
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Search with contextual query expansion."""
        # Expand query with related terms
        expanded_query, expansion_terms = await self.query_expander.expand_query(
            query.query_text, max_expansions=config.max_expansion_terms
        )
        
        metrics.expansion_terms = expansion_terms
        
        # Create expanded query object
        expanded_query_obj = SemanticQuery(
            query_text=expanded_query,
            query_type=query.query_type,
            target_entity_types=query.target_entity_types,
            filters=query.filters,
            limit=query.limit,
            threshold=query.threshold
        )
        
        # Perform search with expanded query
        return await self._vector_similarity_search(expanded_query_obj, config, metrics)
    
    async def _multi_modal_search(
        self, 
        query: SemanticQuery, 
        config: SearchConfiguration,
        metrics: SearchMetrics
    ) -> Tuple[List[SemanticSearchResult], SearchMetrics]:
        """Search using multiple embedding models and combine results."""
        # This would require multiple embedding providers
        # For now, fall back to hybrid search
        return await self._hybrid_semantic_search(query, config, metrics)
    
    def _generate_cache_key(self, query: SemanticQuery, config: SearchConfiguration) -> str:
        """Generate cache key for search query and config."""
        key_parts = [
            query.query_text,
            query.query_type,
            ','.join(sorted(query.target_entity_types)),
            str(query.limit),
            str(query.threshold),
            config.strategy.value,
            config.ranking.value,
            str(config.similarity_threshold)
        ]
        return '|'.join(key_parts)
    
    async def get_search_suggestions(
        self, 
        partial_query: str, 
        limit: int = 5
    ) -> List[str]:
        """Get search suggestions for partial query."""
        if len(partial_query) < 2:
            return []
        
        # Find entities that start with or contain the partial query
        # This would need full-text search capabilities in the repository
        # Placeholder implementation
        suggestions = []
        
        # Could implement with fuzzy matching on entity names
        # For now, return empty list
        return suggestions
    
    async def clear_cache(self):
        """Clear search result cache."""
        self.search_cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.search_cache),
            "cache_keys": list(self.search_cache.keys())
        }
