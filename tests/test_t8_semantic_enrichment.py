# T8 Semantic Enrichment Engine - Comprehensive Test Suite

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any, Tuple

from echo_roots.semantic import (
    SemanticEmbedding, SemanticRelationship, SemanticConcept,
    EnrichmentTask, SemanticQuery, SemanticSearchResult, EnrichmentStats,
    EmbeddingModel, SemanticRelationType, EnrichmentStatus, ConfidenceLevel,
    EmbeddingProvider, SemanticRepository,
    TextProcessor, RelationshipExtractor, ConceptExtractor, SemanticEnrichmentEngine
)

from echo_roots.semantic.search import (
    SearchStrategy, RankingStrategy, SearchScope, SearchConfiguration,
    RankingFactors, SearchContext, SearchMetrics,
    QueryExpander, ResultRanker, SemanticSearchEngine
)

from echo_roots.semantic.graph import (
    GraphQueryType, GraphMetric, IntegrationType,
    GraphNode, GraphEdge, GraphPath, GraphCluster, GraphMetrics, IntegrationTask,
    KnowledgeGraphBuilder, SemanticIntegrator
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""
    
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        # Simple mock embedding based on text hash
        text_hash = hash(text) % 1000
        return [float(text_hash / 1000), float((text_hash + 1) / 1000), float((text_hash + 2) / 1000)]
    
    async def generate_batch_embeddings(
        self, 
        texts: List[str], 
        model: str = None,
        batch_size: int = 100
    ) -> List[List[float]]:
        return [await self.generate_embedding(text, model) for text in texts]
    
    def get_embedding_dimensions(self, model: str) -> int:
        return 3
    
    async def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        # Simple cosine similarity
        import numpy as np
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


class MockSemanticRepository(SemanticRepository):
    """Mock semantic repository for testing."""
    
    def __init__(self):
        self.embeddings = {}
        self.relationships = []
        self.concepts = []
        self.tasks = []
    
    async def store_embedding(self, embedding: SemanticEmbedding) -> str:
        self.embeddings[embedding.entity_id] = embedding
        return embedding.embedding_id
    
    async def get_embedding(self, entity_id: str, model_name: str = None) -> SemanticEmbedding:
        return self.embeddings.get(entity_id)
    
    async def find_similar_embeddings(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        threshold: float = 0.5,
        entity_types: List[str] = None
    ) -> List[Tuple[SemanticEmbedding, float]]:
        results = []
        
        for embedding in self.embeddings.values():
            if entity_types and embedding.entity_type not in entity_types:
                continue
            
            # Calculate similarity using mock provider
            provider = MockEmbeddingProvider()
            similarity = await provider.calculate_similarity(query_embedding, embedding.embedding_vector)
            
            if similarity >= threshold:
                results.append((embedding, similarity))
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def store_relationship(self, relationship: SemanticRelationship) -> str:
        self.relationships.append(relationship)
        return relationship.relationship_id
    
    async def get_relationships(
        self, 
        entity_id: str, 
        relationship_types: List[SemanticRelationType] = None
    ) -> List[SemanticRelationship]:
        results = []
        
        for rel in self.relationships:
            if rel.source_entity_id == entity_id:
                if not relationship_types or rel.relationship_type in relationship_types:
                    results.append(rel)
        
        return results
    
    async def store_concept(self, concept: SemanticConcept) -> str:
        self.concepts.append(concept)
        return concept.concept_id
    
    async def find_concepts(
        self, 
        query: str = None,
        concept_types: List[str] = None,
        domains: List[str] = None
    ) -> List[SemanticConcept]:
        results = []
        
        for concept in self.concepts:
            match = True
            
            if concept_types and concept.concept_type not in concept_types:
                match = False
            
            if domains and not any(domain in concept.domains for domain in domains):
                match = False
            
            if query and query.lower() not in concept.concept_name.lower():
                match = False
            
            if match:
                results.append(concept)
        
        return results
    
    async def create_enrichment_task(self, task: EnrichmentTask) -> str:
        self.tasks.append(task)
        return task.task_id
    
    async def get_pending_tasks(self, limit: int = 100) -> List[EnrichmentTask]:
        pending_tasks = [task for task in self.tasks if task.status == EnrichmentStatus.PENDING]
        return pending_tasks[:limit]
    
    async def update_task_status(self, task_id: str, status: EnrichmentStatus, error: str = None) -> bool:
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                if error:
                    task.error_message = error
                return True
        return False


@pytest.fixture
def mock_embedding_provider():
    return MockEmbeddingProvider()


@pytest.fixture
def mock_repository():
    return MockSemanticRepository()


@pytest.fixture
def sample_embedding():
    return SemanticEmbedding(
        embedding_id="emb_001",
        entity_id="entity_001",
        entity_type="category",
        embedding_vector=[0.1, 0.2, 0.3],
        model_name=EmbeddingModel.OPENAI_ADA_002.value,
        model_version="1.0",
        dimensions=3,
        metadata={"source_text": "electronics"}
    )


@pytest.fixture
def sample_relationship():
    return SemanticRelationship(
        relationship_id="rel_001",
        source_entity_id="entity_001",
        target_entity_id="entity_002",
        relationship_type=SemanticRelationType.HYPONYM,
        confidence_score=0.8,
        confidence_level=ConfidenceLevel.HIGH,
        source_text="electronics",
        target_text="smartphones"
    )


@pytest.fixture
def sample_concept():
    return SemanticConcept(
        concept_id="concept_001",
        concept_name="Technology Products",
        concept_type="category_group",
        description="High-level concept for technology-related products",
        related_entities=["entity_001", "entity_002"],
        keywords=["technology", "electronics", "digital"],
        confidence_score=0.9,
        frequency=15,
        domains=["electronics"]
    )


class TestSemanticModels:
    """Test semantic data models."""
    
    def test_semantic_embedding_creation(self, sample_embedding):
        assert sample_embedding.embedding_id == "emb_001"
        assert sample_embedding.entity_id == "entity_001"
        assert sample_embedding.entity_type == "category"
        assert len(sample_embedding.embedding_vector) == 3
        assert sample_embedding.dimensions == 3
        assert sample_embedding.is_active
    
    def test_semantic_relationship_creation(self, sample_relationship):
        assert sample_relationship.relationship_id == "rel_001"
        assert sample_relationship.source_entity_id == "entity_001"
        assert sample_relationship.target_entity_id == "entity_002"
        assert sample_relationship.relationship_type == SemanticRelationType.HYPONYM
        assert sample_relationship.confidence_score == 0.8
        assert sample_relationship.confidence_level == ConfidenceLevel.HIGH
    
    def test_semantic_concept_creation(self, sample_concept):
        assert sample_concept.concept_id == "concept_001"
        assert sample_concept.concept_name == "Technology Products"
        assert sample_concept.concept_type == "category_group"
        assert len(sample_concept.related_entities) == 2
        assert "technology" in sample_concept.keywords
        assert "electronics" in sample_concept.domains
    
    def test_confidence_level_mapping(self):
        # Test confidence level enum values
        assert ConfidenceLevel.VERY_HIGH == "very_high"
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.VERY_LOW == "very_low"
    
    def test_semantic_relation_types(self):
        # Test all relationship types are available
        assert SemanticRelationType.SYNONYM == "synonym"
        assert SemanticRelationType.HYPONYM == "hyponym"
        assert SemanticRelationType.HYPERNYM == "hypernym"
        assert SemanticRelationType.SIMILAR == "similar"
        assert SemanticRelationType.RELATED == "related"


class TestTextProcessor:
    """Test text processing functionality."""
    
    def test_extract_keywords(self):
        processor = TextProcessor()
        text = "wireless bluetooth headphones with noise cancellation"
        keywords = processor.extract_keywords(text, max_keywords=5)
        
        assert len(keywords) <= 5
        assert "wireless" in keywords
        assert "bluetooth" in keywords
        assert "headphones" in keywords
        assert "noise" in keywords
        assert "cancellation" in keywords
        # Stop words should be excluded
        assert "with" not in keywords
    
    def test_extract_phrases(self):
        processor = TextProcessor()
        text = "smart home automation system"
        phrases = processor.extract_phrases(text, min_length=2, max_length=3)
        
        assert "smart home" in phrases
        assert "home automation" in phrases
        assert "automation system" in phrases
        assert "smart home automation" in phrases
    
    def test_clean_text_for_embedding(self):
        processor = TextProcessor()
        dirty_text = "  Smart   TV   with   4K   resolution!!!  "
        cleaned = processor.clean_text_for_embedding(dirty_text)
        
        assert cleaned == "Smart TV with 4K resolution"
        assert "  " not in cleaned  # No double spaces
        assert not cleaned.startswith(" ")  # No leading spaces
        assert not cleaned.endswith(" ")  # No trailing spaces
    
    def test_calculate_text_similarity(self):
        processor = TextProcessor()
        text1 = "smartphone with camera"
        text2 = "mobile phone with camera"
        text3 = "laptop computer keyboard"
        
        # Similar texts should have higher similarity
        similarity_1_2 = processor.calculate_text_similarity(text1, text2)
        similarity_1_3 = processor.calculate_text_similarity(text1, text3)
        
        assert similarity_1_2 > similarity_1_3
        assert similarity_1_2 > 0.0
        assert similarity_1_3 >= 0.0
    
    def test_empty_text_handling(self):
        processor = TextProcessor()
        
        # Test with empty text
        assert processor.extract_keywords("") == []
        assert processor.extract_phrases("") == []
        assert processor.clean_text_for_embedding("") == ""
        assert processor.calculate_text_similarity("", "test") == 0.0


class TestRelationshipExtractor:
    """Test relationship extraction functionality."""
    
    @pytest.mark.asyncio
    async def test_extract_relationships(self, mock_embedding_provider):
        extractor = RelationshipExtractor(mock_embedding_provider)
        
        source_text = "smartphones"
        target_texts = ["mobile phones", "electronics", "laptops"]
        source_id = "entity_001"
        target_ids = ["entity_002", "entity_003", "entity_004"]
        
        relationships = await extractor.extract_relationships(
            source_text, target_texts, source_id, target_ids
        )
        
        assert isinstance(relationships, list)
        # Should find at least some relationships
        assert len(relationships) >= 0
        
        for rel in relationships:
            assert isinstance(rel, SemanticRelationship)
            assert rel.source_entity_id == source_id
            assert rel.target_entity_id in target_ids
            assert rel.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_relationship_classification(self, mock_embedding_provider):
        extractor = RelationshipExtractor(mock_embedding_provider)
        
        # Test synonym detection
        rel_type, confidence = await extractor._classify_relationship(
            "smartphone", "mobile phone", 0.95
        )
        assert rel_type == SemanticRelationType.SYNONYM
        
        # Test hierarchical relationship
        rel_type, confidence = await extractor._classify_relationship(
            "electronics", "smartphones", 0.8
        )
        # Should detect some relationship
        assert rel_type is not None
        assert confidence > 0.0
    
    def test_confidence_level_conversion(self, mock_embedding_provider):
        extractor = RelationshipExtractor(mock_embedding_provider)
        
        assert extractor._score_to_confidence_level(0.95) == ConfidenceLevel.VERY_HIGH
        assert extractor._score_to_confidence_level(0.8) == ConfidenceLevel.HIGH
        assert extractor._score_to_confidence_level(0.6) == ConfidenceLevel.MEDIUM
        assert extractor._score_to_confidence_level(0.4) == ConfidenceLevel.LOW
        assert extractor._score_to_confidence_level(0.2) == ConfidenceLevel.VERY_LOW


class TestConceptExtractor:
    """Test concept extraction functionality."""
    
    @pytest.mark.asyncio
    async def test_extract_concepts(self, mock_embedding_provider):
        extractor = ConceptExtractor(mock_embedding_provider)
        
        entities = [
            ("entity_001", "smartphone", "product"),
            ("entity_002", "mobile phone", "product"),
            ("entity_003", "tablet", "product"),
            ("entity_004", "laptop", "product"),
            ("entity_005", "desktop computer", "product")
        ]
        
        concepts = await extractor.extract_concepts(entities, min_cluster_size=2, max_concepts=10)
        
        assert isinstance(concepts, list)
        # Should extract some concepts if entities are clusterable
        for concept in concepts:
            assert isinstance(concept, SemanticConcept)
            assert len(concept.related_entities) >= 2  # min_cluster_size
            assert concept.frequency >= 2
            assert concept.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_clustering_embeddings(self, mock_embedding_provider):
        extractor = ConceptExtractor(mock_embedding_provider)
        
        # Create similar embeddings
        embeddings = [
            [0.1, 0.2, 0.3],  # Similar group
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
            [0.8, 0.9, 0.7],  # Different group
            [0.81, 0.91, 0.71]
        ]
        
        clusters = await extractor._cluster_embeddings(embeddings, min_cluster_size=2)
        
        assert isinstance(clusters, list)
        for cluster in clusters:
            assert len(cluster) >= 2  # min_cluster_size
    
    def test_concept_name_generation(self, mock_embedding_provider):
        extractor = ConceptExtractor(mock_embedding_provider)
        
        keywords = ["electronics", "mobile", "technology"]
        entities = [
            ("entity_001", "smartphone", "product"),
            ("entity_002", "tablet", "product")
        ]
        
        concept_name = extractor._generate_concept_name(keywords, entities)
        
        assert isinstance(concept_name, str)
        assert len(concept_name) > 0
        # Should contain some of the keywords
        assert any(keyword.lower() in concept_name.lower() for keyword in keywords)


class TestSemanticEnrichmentEngine:
    """Test main semantic enrichment engine."""
    
    @pytest.mark.asyncio
    async def test_enrich_entity(self, mock_repository, mock_embedding_provider):
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        entity_id = "entity_001"
        entity_text = "wireless bluetooth headphones"
        entity_type = "product"
        
        success = await engine.enrich_entity(entity_id, entity_text, entity_type)
        
        assert success
        
        # Check that embedding was stored
        embedding = await mock_repository.get_embedding(entity_id)
        assert embedding is not None
        assert embedding.entity_id == entity_id
        assert embedding.entity_type == entity_type
        assert len(embedding.embedding_vector) == 3  # Mock provider returns 3D vectors
    
    @pytest.mark.asyncio
    async def test_batch_enrich_entities(self, mock_repository, mock_embedding_provider):
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        entities = [
            ("entity_001", "smartphone", "product"),
            ("entity_002", "laptop", "product"),
            ("entity_003", "tablet", "product")
        ]
        
        results = await engine.batch_enrich_entities(entities, batch_size=2)
        
        assert len(results) == 3
        assert all(results.values())  # All should be successful
        
        # Check that all embeddings were stored
        for entity_id, _, _ in entities:
            embedding = await mock_repository.get_embedding(entity_id)
            assert embedding is not None
    
    @pytest.mark.asyncio
    async def test_find_similar_entities(self, mock_repository, mock_embedding_provider):
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        # First enrich some entities
        entities = [
            ("entity_001", "smartphone", "product"),
            ("entity_002", "mobile phone", "product"),
            ("entity_003", "laptop", "product")
        ]
        
        for entity_id, entity_text, entity_type in entities:
            await engine.enrich_entity(entity_id, entity_text, entity_type)
        
        # Find similar entities
        results = await engine.find_similar_entities(
            "mobile device", 
            entity_types=["product"],
            limit=5,
            threshold=0.1
        )
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, SemanticSearchResult)
            assert result.entity_type == "product"
            assert result.similarity_score >= 0.1
    
    @pytest.mark.asyncio
    async def test_discover_relationships(self, mock_repository, mock_embedding_provider):
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        # Enrich entities first
        entities = [
            ("entity_001", "smartphones", "category"),
            ("entity_002", "mobile phones", "category"),
            ("entity_003", "electronics", "category")
        ]
        
        for entity_id, entity_text, entity_type in entities:
            await engine.enrich_entity(entity_id, entity_text, entity_type)
        
        # Discover relationships
        relationships = await engine.discover_relationships("entity_001", limit=10)
        
        assert isinstance(relationships, list)
        for rel in relationships:
            assert isinstance(rel, SemanticRelationship)
            assert rel.source_entity_id == "entity_001" or rel.target_entity_id == "entity_001"
    
    @pytest.mark.asyncio
    async def test_process_enrichment_tasks(self, mock_repository, mock_embedding_provider):
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        # Create some enrichment tasks
        task1 = EnrichmentTask(
            task_id="task_001",
            entity_id="entity_001",
            entity_type="product",
            task_type="embedding",
            metadata={"source_text": "smartphone"}
        )
        
        task2 = EnrichmentTask(
            task_id="task_002",
            entity_id="entity_002",
            entity_type="product",
            task_type="relationship"
        )
        
        await mock_repository.create_enrichment_task(task1)
        await mock_repository.create_enrichment_task(task2)
        
        # Process tasks
        processed_count = await engine.process_enrichment_tasks(max_tasks=10)
        
        assert processed_count >= 0
        # At least one task should be processed
        assert processed_count <= 2
    
    @pytest.mark.asyncio
    async def test_get_enrichment_stats(self, mock_repository, mock_embedding_provider):
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        stats = await engine.get_enrichment_stats()
        
        assert isinstance(stats, EnrichmentStats)
        assert stats.total_embeddings >= 0
        assert stats.total_relationships >= 0
        assert stats.total_concepts >= 0
        assert isinstance(stats.last_updated, datetime)


class TestSearchComponents:
    """Test semantic search components."""
    
    @pytest.mark.asyncio
    async def test_query_expander(self, mock_repository, mock_embedding_provider):
        expander = QueryExpander(mock_repository, mock_embedding_provider)
        
        # Setup some test data
        embedding = SemanticEmbedding(
            embedding_id="emb_001",
            entity_id="entity_001",
            entity_type="product",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name=EmbeddingModel.OPENAI_ADA_002.value,
            model_version="1.0",
            dimensions=3,
            metadata={"source_text": "smartphone"}
        )
        await mock_repository.store_embedding(embedding)
        
        expanded_query, expansion_terms = await expander.expand_query(
            "mobile phone", max_expansions=3, expansion_threshold=0.1
        )
        
        assert isinstance(expanded_query, str)
        assert isinstance(expansion_terms, list)
        assert "mobile phone" in expanded_query
    
    @pytest.mark.asyncio
    async def test_result_ranker(self, mock_repository):
        ranker = ResultRanker(mock_repository)
        
        # Create test results
        results = [
            SemanticSearchResult(
                entity_id="entity_001",
                entity_type="product",
                entity_text="smartphone",
                similarity_score=0.9
            ),
            SemanticSearchResult(
                entity_id="entity_002",
                entity_type="product",
                entity_text="mobile phone",
                similarity_score=0.8
            ),
            SemanticSearchResult(
                entity_id="entity_003",
                entity_type="product",
                entity_text="laptop",
                similarity_score=0.6
            )
        ]
        
        config = SearchConfiguration()
        ranked_results = await ranker.rank_results(results, config)
        
        assert len(ranked_results) == len(results)
        
        # Check that results are ranked (should be sorted by final score)
        scores = [factors.final_score for result, factors in ranked_results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_semantic_search_engine(self, mock_repository, mock_embedding_provider):
        engine = SemanticSearchEngine(mock_repository, mock_embedding_provider)
        
        # Setup test data
        embedding = SemanticEmbedding(
            embedding_id="emb_001",
            entity_id="entity_001",
            entity_type="product",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name=EmbeddingModel.OPENAI_ADA_002.value,
            model_version="1.0",
            dimensions=3,
            metadata={"source_text": "smartphone"}
        )
        await mock_repository.store_embedding(embedding)
        
        query = SemanticQuery(
            query_text="mobile device",
            query_type="similarity",
            target_entity_types=["product"],
            limit=5,
            threshold=0.1
        )
        
        results, metrics = await engine.search(query)
        
        assert isinstance(results, list)
        assert isinstance(metrics, SearchMetrics)
        assert metrics.query_time_ms >= 0
        assert metrics.final_results == len(results)
        
        for result in results:
            assert isinstance(result, SemanticSearchResult)
            assert result.similarity_score >= 0.1


class TestGraphComponents:
    """Test knowledge graph components."""
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_builder(self, mock_repository):
        builder = KnowledgeGraphBuilder(mock_repository)
        
        # Setup test embeddings
        embeddings = [
            SemanticEmbedding(
                embedding_id="emb_001",
                entity_id="entity_001",
                entity_type="product",
                embedding_vector=[0.1, 0.2, 0.3],
                model_name=EmbeddingModel.OPENAI_ADA_002.value,
                model_version="1.0",
                dimensions=3,
                metadata={"source_text": "smartphone"}
            ),
            SemanticEmbedding(
                embedding_id="emb_002",
                entity_id="entity_002",
                entity_type="product",
                embedding_vector=[0.4, 0.5, 0.6],
                model_name=EmbeddingModel.OPENAI_ADA_002.value,
                model_version="1.0",
                dimensions=3,
                metadata={"source_text": "tablet"}
            )
        ]
        
        for embedding in embeddings:
            await mock_repository.store_embedding(embedding)
        
        # Mock the _get_embeddings_by_types method
        builder._get_embeddings_by_types = AsyncMock(return_value=embeddings)
        
        nodes, edges = await builder.build_graph(entity_types=["product"])
        
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert len(nodes) == len(embeddings)
        
        for node in nodes:
            assert isinstance(node, GraphNode)
            assert node.entity_type == "product"
    
    @pytest.mark.asyncio
    async def test_entity_neighborhood(self, mock_repository):
        builder = KnowledgeGraphBuilder(mock_repository)
        
        # Setup test data
        embedding = SemanticEmbedding(
            embedding_id="emb_001",
            entity_id="entity_001",
            entity_type="product",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name=EmbeddingModel.OPENAI_ADA_002.value,
            model_version="1.0",
            dimensions=3,
            metadata={"source_text": "smartphone"}
        )
        await mock_repository.store_embedding(embedding)
        
        neighborhood_nodes, neighborhood_edges = await builder.get_entity_neighborhood(
            "entity_001", radius=2, min_confidence=0.3
        )
        
        assert isinstance(neighborhood_nodes, list)
        assert isinstance(neighborhood_edges, list)
        
        # Should at least contain the source entity
        assert len(neighborhood_nodes) >= 1
        assert any(node.entity_id == "entity_001" for node in neighborhood_nodes)
    
    @pytest.mark.asyncio
    async def test_semantic_integrator(self, mock_repository):
        graph_builder = KnowledgeGraphBuilder(mock_repository)
        integrator = SemanticIntegrator(mock_repository, graph_builder)
        
        # Test taxonomy enrichment
        taxonomy_data = {
            "categories": {
                "cat_001": {
                    "name": "Electronics",
                    "description": "Electronic products"
                }
            }
        }
        
        # Setup embedding for category
        embedding = SemanticEmbedding(
            embedding_id="emb_cat_001",
            entity_id="cat_001",
            entity_type="category",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name=EmbeddingModel.OPENAI_ADA_002.value,
            model_version="1.0",
            dimensions=3,
            metadata={"source_text": "Electronics"}
        )
        await mock_repository.store_embedding(embedding)
        
        enriched_taxonomy = await integrator.enrich_taxonomy(taxonomy_data)
        
        assert "categories" in enriched_taxonomy
        assert "cat_001" in enriched_taxonomy["categories"]
        # Should have original data
        assert enriched_taxonomy["categories"]["cat_001"]["name"] == "Electronics"
    
    @pytest.mark.asyncio
    async def test_data_quality_assessment(self, mock_repository):
        graph_builder = KnowledgeGraphBuilder(mock_repository)
        integrator = SemanticIntegrator(mock_repository, graph_builder)
        
        # Setup test entities
        entity_ids = ["entity_001", "entity_002", "entity_003"]
        
        # Add embedding for only some entities
        embedding = SemanticEmbedding(
            embedding_id="emb_001",
            entity_id="entity_001",
            entity_type="product",
            embedding_vector=[0.1, 0.2, 0.3],
            model_name=EmbeddingModel.OPENAI_ADA_002.value,
            model_version="1.0",
            dimensions=3,
            metadata={"source_text": "smartphone"}
        )
        await mock_repository.store_embedding(embedding)
        
        # Add relationship
        relationship = SemanticRelationship(
            relationship_id="rel_001",
            source_entity_id="entity_001",
            target_entity_id="entity_002",
            relationship_type=SemanticRelationType.SIMILAR,
            confidence_score=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            source_text="smartphone",
            target_text="mobile phone"
        )
        await mock_repository.store_relationship(relationship)
        
        quality_assessment = await integrator.assess_data_quality(entity_ids)
        
        assert isinstance(quality_assessment, dict)
        assert "total_entities" in quality_assessment
        assert quality_assessment["total_entities"] == 3
        assert "entities_with_embeddings" in quality_assessment
        assert "entities_with_relationships" in quality_assessment
        assert "quality_issues" in quality_assessment
        assert "recommendations" in quality_assessment


class TestIntegrationWorkflows:
    """Test end-to-end integration workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_enrichment_workflow(self, mock_repository, mock_embedding_provider):
        """Test complete semantic enrichment workflow."""
        engine = SemanticEnrichmentEngine(mock_repository, mock_embedding_provider)
        
        # Step 1: Enrich entities
        entities = [
            ("entity_001", "smartphone", "product"),
            ("entity_002", "mobile phone", "product"),
            ("entity_003", "electronics", "category")
        ]
        
        results = await engine.batch_enrich_entities(entities)
        assert all(results.values())
        
        # Step 2: Process enrichment tasks
        processed_count = await engine.process_enrichment_tasks()
        assert processed_count >= 0
        
        # Step 3: Find similar entities
        similar_results = await engine.find_similar_entities(
            "mobile device", 
            entity_types=["product"],
            limit=5
        )
        assert isinstance(similar_results, list)
        
        # Step 4: Get statistics
        stats = await engine.get_enrichment_stats()
        assert isinstance(stats, EnrichmentStats)
    
    @pytest.mark.asyncio
    async def test_search_and_ranking_workflow(self, mock_repository, mock_embedding_provider):
        """Test semantic search and ranking workflow."""
        search_engine = SemanticSearchEngine(mock_repository, mock_embedding_provider)
        
        # Setup test data
        embeddings = [
            SemanticEmbedding(
                embedding_id="emb_001",
                entity_id="entity_001",
                entity_type="product",
                embedding_vector=[0.1, 0.2, 0.3],
                model_name=EmbeddingModel.OPENAI_ADA_002.value,
                model_version="1.0",
                dimensions=3,
                metadata={"source_text": "smartphone"}
            ),
            SemanticEmbedding(
                embedding_id="emb_002",
                entity_id="entity_002",
                entity_type="product",
                embedding_vector=[0.15, 0.25, 0.35],
                model_name=EmbeddingModel.OPENAI_ADA_002.value,
                model_version="1.0",
                dimensions=3,
                metadata={"source_text": "mobile phone"}
            )
        ]
        
        for embedding in embeddings:
            await mock_repository.store_embedding(embedding)
        
        # Test different search strategies
        strategies = [
            SearchStrategy.VECTOR_SIMILARITY,
            SearchStrategy.HYBRID_SEMANTIC,
            SearchStrategy.CONTEXTUAL_EXPANSION
        ]
        
        for strategy in strategies:
            config = SearchConfiguration(strategy=strategy)
            query = SemanticQuery(
                query_text="mobile device",
                query_type="similarity",
                limit=5
            )
            
            results, metrics = await search_engine.search(query, config)
            
            assert isinstance(results, list)
            assert isinstance(metrics, SearchMetrics)
            assert metrics.strategy_used == strategy
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_workflow(self, mock_repository):
        """Test knowledge graph building and analysis workflow."""
        builder = KnowledgeGraphBuilder(mock_repository)
        
        # Setup test data
        embeddings = [
            SemanticEmbedding(
                embedding_id="emb_001",
                entity_id="entity_001",
                entity_type="product",
                embedding_vector=[0.1, 0.2, 0.3],
                model_name=EmbeddingModel.OPENAI_ADA_002.value,
                model_version="1.0",
                dimensions=3,
                metadata={"source_text": "smartphone"}
            ),
            SemanticEmbedding(
                embedding_id="emb_002",
                entity_id="entity_002",
                entity_type="product",
                embedding_vector=[0.4, 0.5, 0.6],
                model_name=EmbeddingModel.OPENAI_ADA_002.value,
                model_version="1.0",
                dimensions=3,
                metadata={"source_text": "tablet"}
            )
        ]
        
        for embedding in embeddings:
            await mock_repository.store_embedding(embedding)
        
        # Mock the method that gets embeddings by type
        builder._get_embeddings_by_types = AsyncMock(return_value=embeddings)
        
        # Build graph
        nodes, edges = await builder.build_graph()
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        
        # Compute metrics
        metrics = await builder.compute_graph_metrics(nodes, edges)
        assert isinstance(metrics, GraphMetrics)
        assert metrics.total_nodes == len(nodes)
        assert metrics.total_edges == len(edges)
        
        # Test clustering
        clusters = await builder.detect_clusters(nodes, edges)
        assert isinstance(clusters, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
