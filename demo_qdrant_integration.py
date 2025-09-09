#!/usr/bin/env python3
"""
Qdrant Integration Demo Script

This script demonstrates the complete Qdrant vector storage integration
with Echo Roots semantic processing capabilities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, UTC

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent / "src"))

from echo_roots.models.taxonomy import SemanticCandidate
from echo_roots.storage.qdrant_backend import QdrantBackend
from echo_roots.semantic.embedding_providers import EmbeddingProviderFactory
from echo_roots.semantic.pipeline import SemanticProcessingPipeline
from echo_roots.semantic.search import SemanticSearchEngine, SearchConfiguration
from echo_roots.semantic import SemanticQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantDemo:
    """Demonstrates Qdrant integration capabilities."""
    
    def __init__(self):
        self.qdrant_backend = None
        self.embedding_provider = None
        self.pipeline = None
        self.search_engine = None
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("🚀 Initializing Qdrant Integration Demo")
        
        # Qdrant configuration
        qdrant_config = {
            "host": "localhost",
            "port": 6333,
            "prefer_grpc": False,
            "timeout": 30.0,
            "collections": {
                "demo_candidates": {
                    "vector_size": 384,
                    "distance": "Cosine"
                }
            }
        }
        
        try:
            # Initialize Qdrant backend
            logger.info("📦 Initializing Qdrant backend...")
            self.qdrant_backend = QdrantBackend(qdrant_config)
            await self.qdrant_backend.initialize()
            
            # Test connection
            is_healthy = await self.qdrant_backend.health_check()
            if not is_healthy:
                raise ConnectionError("Qdrant health check failed")
            logger.info("✅ Qdrant connection established")
            
            # Create embedding provider
            logger.info("🧠 Initializing embedding provider...")
            embedding_config = {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32
            }
            
            self.embedding_provider = EmbeddingProviderFactory.create_provider(
                "sentence_transformers", embedding_config
            )
            await self.embedding_provider.initialize()
            logger.info("✅ Embedding provider ready")
            
            # Create processing pipeline
            logger.info("⚙️ Setting up semantic processing pipeline...")
            self.pipeline = SemanticProcessingPipeline(
                embedding_provider=self.embedding_provider,
                vector_repository=self.qdrant_backend.semantic_repository,
                batch_size=10
            )
            await self.pipeline.initialize()
            logger.info("✅ Processing pipeline ready")
            
            # Create search engine
            logger.info("🔍 Initializing semantic search engine...")
            # Mock repository for demo - in real use, this would be actual repository
            from unittest.mock import AsyncMock
            mock_repository = AsyncMock()
            
            self.search_engine = SemanticSearchEngine(
                repository=mock_repository,
                embedding_provider=self.embedding_provider,
                vector_repository=self.qdrant_backend.semantic_repository
            )
            logger.info("✅ Search engine ready")
            
            logger.info("🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def create_sample_candidates(self) -> list:
        """Create sample semantic candidates for demonstration."""
        return [
            SemanticCandidate(
                candidate_id="demo_1",
                term="smartphone",
                normalized_term="smartphone",
                frequency=150,
                contexts=["mobile communication device", "handheld computer"],
                score=0.95,
                language="en",
                domain="electronics"
            ),
            SemanticCandidate(
                candidate_id="demo_2",
                term="mobile phone",
                normalized_term="mobile phone",
                frequency=120,
                contexts=["cellular telephone", "wireless communication"],
                score=0.90,
                language="en",
                domain="electronics"
            ),
            SemanticCandidate(
                candidate_id="demo_3",
                term="tablet computer",
                normalized_term="tablet computer",
                frequency=80,
                contexts=["portable computing device", "touchscreen interface"],
                score=0.88,
                language="en",
                domain="electronics"
            ),
            SemanticCandidate(
                candidate_id="demo_4",
                term="laptop",
                normalized_term="laptop",
                frequency=100,
                contexts=["portable personal computer", "mobile workstation"],
                score=0.92,
                language="en",
                domain="electronics"
            ),
            SemanticCandidate(
                candidate_id="demo_5",
                term="smartwatch",
                normalized_term="smartwatch",
                frequency=60,
                contexts=["wearable technology", "fitness tracker"],
                score=0.85,
                language="en",
                domain="electronics"
            ),
            SemanticCandidate(
                candidate_id="demo_6",
                term="wireless headphones",
                normalized_term="wireless headphones",
                frequency=70,
                contexts=["bluetooth audio device", "cordless earphones"],
                score=0.87,
                language="en",
                domain="electronics"
            )
        ]
    
    async def demonstrate_pipeline_processing(self):
        """Demonstrate semantic candidate processing."""
        logger.info("\n📊 Demonstrating Semantic Processing Pipeline")
        
        # Create sample candidates
        candidates = self.create_sample_candidates()
        logger.info(f"Created {len(candidates)} sample candidates")
        
        # Process candidates through pipeline
        start_time = datetime.now(UTC)
        embedding_ids = await self.pipeline.process_semantic_candidates(candidates)
        end_time = datetime.now(UTC)
        
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Processed {len(candidates)} candidates in {processing_time:.2f}s")
        logger.info(f"Generated {len(embedding_ids)} embeddings")
        
        # Display pipeline statistics
        stats = self.pipeline.get_statistics()
        logger.info("Pipeline Statistics:")
        logger.info(f"  - Candidates processed: {stats['candidates_processed']}")
        logger.info(f"  - Embeddings generated: {stats['embeddings_generated']}")
        logger.info(f"  - Embeddings stored: {stats['embeddings_stored']}")
        logger.info(f"  - Errors: {stats['errors']}")
        
        return embedding_ids
    
    async def demonstrate_vector_search(self):
        """Demonstrate vector similarity search."""
        logger.info("\n🔍 Demonstrating Vector Similarity Search")
        
        # Test queries
        test_queries = [
            "mobile communication device",
            "portable computer",
            "wearable technology",
            "audio equipment",
            "computing hardware"
        ]
        
        for query_text in test_queries:
            logger.info(f"\nSearching for: '{query_text}'")
            
            # Create search query
            query = SemanticQuery(
                query_text=query_text,
                target_entity_types=["semantic_candidate"],
                limit=3,
                threshold=0.6
            )
            
            # Configure search
            config = SearchConfiguration(
                similarity_threshold=0.6,
                max_results=3
            )
            
            # Perform search
            try:
                results, metrics = await self.search_engine.search(query, config)
                
                logger.info(f"Found {len(results)} results in {metrics.query_time_ms:.1f}ms")
                
                for i, result in enumerate(results, 1):
                    logger.info(f"  {i}. {result.entity_text} (similarity: {result.similarity_score:.3f})")
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
    
    async def demonstrate_similarity_analysis(self):
        """Demonstrate similarity analysis between candidates."""
        logger.info("\n🔬 Demonstrating Similarity Analysis")
        
        # Create subset of candidates for analysis
        candidates = self.create_sample_candidates()[:4]
        
        # Calculate similarity matrix
        logger.info("Calculating similarity matrix...")
        similarity_matrix = await self.pipeline.calculate_similarity_matrix(candidates)
        
        if similarity_matrix:
            logger.info("Similarity Matrix:")
            logger.info("Terms: " + " | ".join(c.term for c in candidates))
            
            for i, row in enumerate(similarity_matrix):
                term = candidates[i].term
                similarities = " | ".join(f"{score:.3f}" for score in row)
                logger.info(f"{term:15} | {similarities}")
        else:
            logger.warning("Could not calculate similarity matrix")
    
    async def demonstrate_embedding_operations(self):
        """Demonstrate direct embedding operations."""
        logger.info("\n🎯 Demonstrating Direct Embedding Operations")
        
        # Generate embeddings for custom texts
        custom_texts = [
            "electronic gadget",
            "digital device",
            "tech product",
            "consumer electronics"
        ]
        
        logger.info("Generating embeddings for custom texts...")
        embeddings = await self.pipeline.generate_embeddings_for_texts(
            custom_texts,
            metadata_list=[{"category": "test", "index": i} for i in range(len(custom_texts))]
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Store embeddings in Qdrant
        repository = self.qdrant_backend.semantic_repository
        
        logger.info("Storing embeddings in Qdrant...")
        embedding_ids = await repository.batch_store_embeddings(embeddings)
        
        logger.info(f"Stored {len(embedding_ids)} embeddings with IDs: {embedding_ids}")
        
        # Test retrieval
        logger.info("Testing embedding retrieval...")
        if embedding_ids:
            retrieved = await repository.get_embedding(embedding_ids[0])
            if retrieved:
                logger.info(f"Successfully retrieved embedding: {retrieved.embedding_id}")
                logger.info(f"Vector dimensions: {len(retrieved.embedding_vector)}")
            else:
                logger.warning("Could not retrieve embedding")
    
    async def demonstrate_health_monitoring(self):
        """Demonstrate health monitoring capabilities."""
        logger.info("\n❤️ Demonstrating Health Monitoring")
        
        # Test Qdrant health
        is_healthy = await self.qdrant_backend.health_check()
        logger.info(f"Qdrant Health Status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
        # Get collection info
        collections = await self.qdrant_backend.list_collections()
        logger.info(f"Available Collections: {collections}")
        
        # Test embedding provider
        try:
            test_embedding = await self.embedding_provider.generate_embedding("test")
            logger.info(f"✅ Embedding Provider: Working (dimensions: {len(test_embedding)})")
        except Exception as e:
            logger.error(f"❌ Embedding Provider: Failed ({e})")
    
    async def cleanup(self):
        """Clean up demo resources."""
        logger.info("\n🧹 Cleaning up demo resources...")
        
        try:
            # Clean up test collections if needed
            collections = await self.qdrant_backend.list_collections()
            
            for collection_name in collections:
                if collection_name.startswith("demo_"):
                    logger.info(f"Removing demo collection: {collection_name}")
                    await self.qdrant_backend.delete_collection(collection_name)
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration."""
        try:
            await self.initialize()
            
            logger.info("\n" + "="*60)
            logger.info("🎬 Starting Qdrant Integration Demonstration")
            logger.info("="*60)
            
            # Run all demonstrations
            await self.demonstrate_pipeline_processing()
            await self.demonstrate_vector_search()
            await self.demonstrate_similarity_analysis()
            await self.demonstrate_embedding_operations()
            await self.demonstrate_health_monitoring()
            
            logger.info("\n" + "="*60)
            logger.info("🎉 Qdrant Integration Demo Completed Successfully!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"\n❌ Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main demo function."""
    print("🔗 Echo Roots - Qdrant Vector Storage Integration Demo")
    print("=" * 60)
    
    # Check if Qdrant is available
    try:
        import qdrant_client
        logger.info("✅ Qdrant client available")
    except ImportError:
        logger.error("❌ Qdrant client not installed. Please run: pip install qdrant-client")
        return 1
    
    # Check if sentence-transformers is available
    try:
        import sentence_transformers
        logger.info("✅ SentenceTransformers available")
    except ImportError:
        logger.error("❌ SentenceTransformers not installed. Please run: pip install sentence-transformers")
        return 1
    
    # Run demonstration
    demo = QdrantDemo()
    
    try:
        await demo.run_complete_demo()
        return 0
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
        return 0
    except ConnectionError as e:
        logger.error(f"\n🚫 Connection Error: {e}")
        logger.info("Please ensure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant:latest")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Demo failed with error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    
    # Run the demo
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
