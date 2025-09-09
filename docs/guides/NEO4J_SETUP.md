# Neo4j Integration Guide

This document explains how to set up and use Neo4j as the graph backend for taxonomy and semantic relationship management in echo-roots.

## Overview

Echo-roots supports a hybrid storage architecture:

- **DuckDB** (Required): Core ingestion, extraction results, analytics
- **Neo4j** (Optional): Taxonomy trees, controlled vocabulary, semantic relationships
- **Qdrant** (Optional): Vector embeddings and semantic search

When Neo4j is enabled, the system provides enhanced capabilities for:
- Efficient taxonomy tree navigation
- Complex semantic relationship queries  
- Graph-based clustering and analysis
- Advanced taxonomy optimization

## Installation

### 1. Install Neo4j

#### Option A: Docker (Recommended for Development)

```bash
# Start Neo4j with APOC plugin
docker run \
    --name neo4j-echo-roots \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/your-password \
    --env NEO4J_PLUGINS='["apoc"]' \
    --env NEO4J_apoc_export_file_enabled=true \
    --env NEO4J_apoc_import_file_enabled=true \
    neo4j:5.28
```

#### Option B: Local Installation

1. Download Neo4j Community Edition from https://neo4j.com/download/
2. Install following the platform-specific instructions
3. Install APOC plugin (optional but recommended):
   ```bash
   # Download APOC jar to plugins directory
   # Restart Neo4j
   ```

### 2. Configure Echo-Roots

Update your `.env` file:

```bash
# Enable Neo4j
NEO4J_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

### 3. Install Python Dependencies

```bash
# Install with graph support
pip install -e ".[graph]"

# Or install neo4j driver separately
pip install neo4j>=5.0.0
```

## Usage

### Basic Setup

```python
from echo_roots.storage import create_storage

# Create hybrid storage with Neo4j
config = {
    "duckdb": {
        "database_path": "./data/echo_roots.duckdb"
    },
    "neo4j": {
        "enabled": True,
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "your-password",
        "database": "neo4j"
    }
}

storage = await create_storage(config)

# Check storage configuration
info = await storage.get_taxonomy_storage_info()
print(f"Taxonomy backend: {info['taxonomy_backend']}")
print(f"Graph queries supported: {info['supports_graph_queries']}")
```

### Taxonomy Operations

```python
from echo_roots.models.taxonomy import Category

# Create categories - automatically synced to Neo4j
category = Category(
    name="Electronics",
    level=0,
    path=["Electronics"],
    description="Electronic devices and components"
)

category_id = await storage.taxonomy.store_category(category)

# Create subcategory with parent relationship
subcategory = Category(
    name="Smartphones",
    parent_id=category_id,
    level=1,
    path=["Electronics", "Smartphones"],
    description="Mobile phones and smartphones"
)

await storage.taxonomy.store_category(subcategory)
```

### Graph Queries

```python
from echo_roots.taxonomy.graph_queries import GraphQueryEngine

# Create query engine
query_engine = GraphQueryEngine(storage.taxonomy)

# Find category relationships
ancestors = await query_engine.get_category_ancestors("smartphone-category-id")
descendants = await query_engine.get_category_descendants("electronics-category-id")
siblings = await query_engine.get_sibling_categories("smartphone-category-id")

# Detect taxonomy issues
issues = await query_engine.detect_taxonomy_issues()
print(f"Found {len(issues['orphan_categories'])} orphan categories")

# Get taxonomy metrics
metrics = await query_engine.calculate_taxonomy_metrics()
print(f"Total categories: {metrics.total_categories}")
print(f"Max depth: {metrics.max_depth}")
```

### Data Migration

If you have existing taxonomy data in DuckDB and want to migrate to Neo4j:

```python
# Sync existing data to Neo4j
sync_result = await storage.sync_taxonomy_to_neo4j()
print(f"Synced {sync_result['categories_synced']} categories")
```

## Graph Schema

The Neo4j schema includes:

### Node Types

- **Category**: Taxonomy nodes with hierarchical relationships
- **Attribute**: Controlled vocabulary attributes  
- **AttributeValue**: Individual values within attributes
- **SemanticCandidate**: Semantic terms in the candidate network

### Relationship Types

- **HAS_CHILD** / **CHILD_OF**: Taxonomy hierarchy
- **HAS_ATTRIBUTE**: Category-to-attribute associations
- **HAS_VALUE**: Attribute-to-value relationships
- **SEMANTIC_RELATION**: Semantic relationships between candidates
- **SIMILAR_TO**: Similarity relationships
- **SYNONYM_OF**: Synonym relationships

### Example Cypher Queries

```cypher
-- Find all subcategories of Electronics
MATCH (root:Category {name: 'Electronics'})-[:HAS_CHILD*]->(sub:Category)
RETURN sub.name, sub.level

-- Find categories without any children (leaf nodes)
MATCH (c:Category)
WHERE NOT (c)-[:HAS_CHILD]->()
RETURN c.name, c.path

-- Find semantic candidates related to a specific term
MATCH (s:SemanticCandidate {normalized_term: 'smartphone'})-[:SEMANTIC_RELATION]-(related)
RETURN related.term, related.frequency
ORDER BY related.frequency DESC
```

## Performance Considerations

### Indexing Strategy

The system automatically creates indexes for:
- Category names and levels
- Attribute names  
- Semantic candidate terms
- Unique constraints on IDs

### Query Optimization

- Use `LIMIT` clauses for large result sets
- Leverage relationship direction for efficient traversal
- Consider using `EXPLAIN` and `PROFILE` for query optimization

### Memory Configuration

For production deployments, tune Neo4j memory settings:

```
# In neo4j.conf
server.memory.heap.initial_size=2G
server.memory.heap.max_size=4G
server.memory.pagecache.size=2G
```

## Monitoring and Maintenance

### Health Checks

```python
# Check Neo4j connectivity
health = await storage.health_check()
print(f"Neo4j status: {health['neo4j']['status']}")
```

### Backup and Recovery

```bash
# Create backup using neo4j-admin
neo4j-admin database dump --database=neo4j --to-path=/backups

# Restore from backup  
neo4j-admin database load --from-path=/backups --database=neo4j --overwrite-destination=true
```

### Performance Monitoring

Access Neo4j Browser at http://localhost:7474 to:
- Monitor query performance
- View graph visualizations
- Execute Cypher queries directly

## Troubleshooting

### Common Issues

1. **Connection refused**: Check that Neo4j is running and accessible
2. **Authentication failed**: Verify username/password in configuration
3. **Memory errors**: Increase Neo4j heap memory allocation
4. **Slow queries**: Add appropriate indexes and optimize Cypher queries

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("echo_roots.storage.neo4j_backend").setLevel(logging.DEBUG)
```

## Migration Path

### Phase 1: DuckDB Only (Current State)
- All data stored in DuckDB
- Basic taxonomy operations supported
- Limited graph query capabilities

### Phase 2: Hybrid Storage (Recommended)
- DuckDB for ingestion and analytics
- Neo4j for taxonomy and relationships
- Enhanced graph query capabilities

### Phase 3: Full Graph Integration (Future)
- Advanced semantic analysis
- Machine learning on graph structure  
- Real-time taxonomy optimization

## Best Practices

1. **Start with DuckDB only** for development and testing
2. **Enable Neo4j for production** when graph queries are needed
3. **Monitor memory usage** as taxonomy grows
4. **Regular backups** of both DuckDB and Neo4j data
5. **Use transactions** for multi-step operations
6. **Index frequently queried properties** for performance

## API Reference

See the [Storage Interface Documentation](../docs/knowledge_base/generated/api_reference.md) for detailed API information.
