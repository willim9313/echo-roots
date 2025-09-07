# Technology Stack

## Principles
- Support Chinese + English (simplified & traditional).
- Avoid rigid ontology; prefer property graphs + controlled vocab.
- Python-first for pipelines and APIs.
- Maintain raw + normalized data in parallel .

## Storage & Graph
- **Neo4j**: core graph (A & C).
- **Qdrant**: semantic candidate retrieval (D).
- **NetworkX**: prototyping & in-memory analysis.
- **DuckDB**: central structured store (core DB) for raw and normalized data.  
  - Stores ingestion inputs, preprocessed attributes, and evaluation tables.  
  - Optimized for Parquet snapshots and analytical queries.  
  - Acts as the staging layer before data flows into graph/vector stores.
- Optional: OpenSearch/ES for keyword search .

## Models
- **Embeddings**: Gemma (main), multilingual-e5 (backup).
- **LLM Hosting**: vLLM (local), Gemini API (cloud).
- **Framework**: LlamaIndex (segmentation, extraction, clustering).
- **Utilities**: tiktoken (token budgeting), DeepEval (evaluation) .

## Services
- Query APIs: GraphQL/REST (`/taxonomy`, `/attributes`, `/semantic`).
- Governance UI: review candidates, merges, rollbacks.
- Export: CSV/Parquet/JSONL snapshots .
