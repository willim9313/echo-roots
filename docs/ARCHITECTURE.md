# System Architecture

## Design Principles

* **Python-first** pipelines and governance.
* **Multi-language** (Chinese + English).
* **Dual-track storage**: raw vs normalized.
* **Observable & versioned** at every stage.
* **Domain-adaptable** via pluggable `domain.yaml` packs.

---

## Functional Layers

1. **Ingestion**: batch/stream from files, DB, APIs.
2. **Domain Adapter**: maps raw fields into core schema (`title`, `description`, `language`, etc.) and injects domain-specific configs (attributes, rules, prompts).
3. **Processing**: segmentation, normalization, aliasing, aggregation.
4. **Storage**:

   * **DuckDB** (core): raw + normalized data, snapshots, analytics.
   * **Neo4j (optional)**: A/C layers — taxonomy tree + controlled vocab.
   * **Qdrant (optional)**: D layer — semantic candidates + embeddings.
5. **Serving**: query, compare, recommend, review.
6. **Governance**: version control, auditing, rollback, metrics.

---

## Storage Models

* **DuckDB (core)**: lightweight analytical backend, Parquet integration, canonical entry point for ingestion.
* **Graph (Neo4j, optional)**: categories, attributes, values, items, terms, mappings.
* **Vector DB (Qdrant, optional)**: semantic candidates, embeddings.

---

## JSON Contracts

* **Ingestion**: raw + evidence, mapped through the Domain Adapter.
* **LLM Extraction**: attributes, terms, evidence, metadata.

  * Attribute lists are extended via `domain.yaml.output_schema.attributes`.
* **Elevation Proposals**: candidate terms → C.
* **Mappings**: alias/merge/replace with versioning.

---

## Metrics (minimal set)

* Coverage of A/C/D layers.
* Normalization quality (attribute-level).
* Efficiency of D→C elevation.
* Version impact scope.
* User-facing experience success.