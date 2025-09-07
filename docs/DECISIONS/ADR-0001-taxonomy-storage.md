# ADR-0001: Taxonomy Storage Model

## Status

Accepted
Date: 2025-09-06

## Context

The project requires a storage model that can support:

* **A (Taxonomy tree)**: hierarchical categories
* **C (Controlled vocab)**: attributes & values with governance
* **D (Semantic candidates)**: free-form, unstable terms
* **Versioning**: snapshots, mappings, rollback
* **Scalability**: lightweight for local dev, scalable for production

Several storage approaches were considered:

1. **Relational (Postgres/MySQL)** — strong schema control, but rigid for graph-like relationships and heavier to operate.
2. **Document DB (MongoDB)** — flexible, but lacks native graph traversal and governance-friendly structure.
3. **Property Graph (Neo4j)** — natural fit for categories, attributes, terms, and their relations.
4. **Vector DB (Qdrant)** — optimized for semantic candidate recall and embedding-based retrieval.
5. **Analytical DB (DuckDB/BigQuery)** — efficient for ingesting raw input, maintaining normalized snapshots, and enabling analytical queries.

## Decision

Adopt a **hybrid storage model** where **DuckDB is the central store** and Neo4j/Qdrant act as optional, specialized layers:

* **DuckDB** as the **core DB and ingestion layer**:

  * Stores raw + normalized item data, attribute tables, and evaluation metrics.
  * Acts as the staging area for ingestion before data flows into other stores.
  * Provides Parquet-based snapshots and SQL analytics.

* **Neo4j** (optional) for **A + C layers**:

  * Stores taxonomy trees, controlled vocab, and their relationships.
  * Supports governance operations (merge, split, alias, deprecate).

* **Qdrant** (optional) for **D layer**:

  * Maintains semantic candidate terms as embeddings.
  * Provides fast vector search for discovery and retrieval.

* Optional: **BigQuery/Postgres** for enterprise scale-out if DuckDB is insufficient.

Domain-specific packs (`domain.yaml`) define which of these layers are activated for a given domain.

This model balances **centralized data ingestion** (DuckDB) with **specialized indexing** (Neo4j/Qdrant).

## Consequences

### Positive

* Clear separation of responsibilities:

  * DuckDB = raw/normalized input + analytics
  * Neo4j = hierarchical + governance queries
  * Qdrant = semantic recall
* Unified ingestion path (all data enters via DuckDB first).
* Lightweight local dev with DuckDB + Parquet, scalable to cloud with BigQuery.
* Efficient governance (version snapshots live in DuckDB).
* **Extensible across domains**: new domains only require a `domain.yaml` pack; no changes to the storage model.

### Negative

* Requires managing multiple storage systems in parallel if all layers are enabled.
* Synchronization complexity between DuckDB and Neo4j/Qdrant.
* Higher ops overhead compared to a single DB design.

## Alternatives Considered

* **Single relational DB**: simpler ops, but poor fit for semantic + graph needs.
* **Single graph DB (Neo4j only)**: could hold attributes + embeddings, but inefficient for batch ingestion/analytics.
* **ElasticSearch/OpenSearch**: hybrid candidate for keyword + vector, but weaker for strict graph governance.

## References

* See [docs/ARCHITECTURE.md](../ARCHITECTURE.md) — functional layers
* See [docs/TECH\_STACK.md](../TECH_STACK.md) — storage and models
* See [docs/DATA\_SCHEMA.md](../DATA_SCHEMA.md) — schema contracts