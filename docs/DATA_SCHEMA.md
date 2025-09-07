# Data Schema & Contracts

This document defines the canonical data structures used across ingestion, processing, storage, and serving.
It serves as the **source of truth** for schemas consumed by pipelines, Copilot agents, and external APIs.

Domain-specific extensions (e.g., brand, material, named entities) are not hardcoded here.
They should be declared in a **domain.yaml** file (see `domains/<name>/domain.yaml`) and merged at runtime.

---

## 1. Ingestion Schema (Core)

```json
{
  "item_id": "string",
  "title": "string",
  "description": "string",
  "raw_category": "string | null",
  "raw_attributes": "object",     // domain-specific, pass-through
  "source": "string",             // data origin (API, DB, CSV, etc.)
  "language": "string",           // e.g. en, zh-tw, zh-cn
  "ingested_at": "timestamp"
}
```

👉 Any **extra fields** (e.g., `brand`, `color`, `size`) should be declared in the domain’s `input_mapping` inside `domain.yaml`.

---

## 2. LLM Extraction Schema (Core)

```json
{
  "item_id": "string",
  "attributes": [
    {
      "name": "string",        // normalized attribute name
      "value": "string",
      "evidence": "string"
    }
  ],
  "terms": [
    {
      "term": "string",
      "context": "string",
      "confidence": "float"    // 0.0–1.0
    }
  ],
  "metadata": {
    "model": "string",
    "run_id": "string",
    "extracted_at": "timestamp"
  }
}
```

👉 The **list of attribute keys** (e.g., `brand`, `material`, `author`) is provided by the domain pack.
Core schema only enforces the structure.

---

## 3. Elevation Proposal Schema

Used when moving a term from **D (semantic candidates)** → **C (controlled attributes)**.

```json
{
  "term": "string",
  "proposed_attribute": "string",
  "justification": "string",
  "metrics": {
    "frequency": "int",
    "coverage": "float",
    "stability_score": "float"
  },
  "submitted_by": "string",
  "status": "pending | approved | rejected",
  "reviewed_at": "timestamp"
}
```

---

## 4. Mapping Schema

Tracks merges, splits, and aliases with versioning.

```json
{
  "mapping_id": "string",
  "from_term": "string",
  "to_term": "string",
  "relation_type": "alias | merge | replace",
  "valid_from": "timestamp",
  "valid_to": "timestamp | null",
  "created_by": "string",
  "notes": "string"
}
```

---

## 5. Storage Models

### Graph (Neo4j)

* **Category (A)**: nodes with hierarchy edges (`PARENT_OF`, `CHILD_OF`).
* **Attribute (C)**: nodes with controlled vocab (`HAS_VALUE`, `ALIAS_OF`).
* **Semantic Term (D)**: candidate nodes linked by (`RELATED_TO`, `VARIANT_OF`).
* **Item**: nodes connected via (`BELONGS_TO`, `HAS_ATTRIBUTE`, `MENTIONS`).

### Vector DB (Qdrant)

* **Collection**: `semantic_terms`
* **Embedding**: 768-dim (multilingual-e5) or 1024-dim (Gemma)
* **Payload fields**:

  * `term`
  * `context`
  * `frequency`
  * `status` (active, deprecated, elevated)

### DuckDB

* **Tables**:

  * `items_raw`
  * `items_normalized`
  * `taxonomy_snapshots`
  * `attribute_mappings`
  * `evaluation_metrics`

---

## 6. Versioning & Auditing

* Every schema change is logged in **`docs/DECISIONS/`** as an ADR.
* Snapshots (taxonomy, attributes, semantic layer) are stored in Parquet + JSONL.
* Rollback supported via timestamped versions.

---

## 7. Domain Extensions

All domain-specific elements (extra attributes, normalization rules, custom prompts) should be declared in the domain pack:

**Example: `domains/ecommerce/domain.yaml`**

```yaml
output_schema:
  attributes:
    - key: brand
      type: categorical
    - key: color
      type: categorical
    - key: size
      type: text
```

Core schema stays fixed. Pipelines automatically merge these extensions during runtime.

