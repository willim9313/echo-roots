# Evaluation Guide

This document defines the **metrics, datasets, and protocols** used to evaluate taxonomy construction, normalization, and semantic enrichment pipelines.

---

## 1. Core Metrics

We track a **minimal but critical set** of metrics across layers A (taxonomy), C (controlled vocab), and D (semantic candidates):

| Metric                        | Definition                                                   | Target |
|-------------------------------|--------------------------------------------------------------|--------|
| **Coverage (A/C/D)**          | % of items correctly assigned to a category, attribute, or candidate term | > 90% |
| **Normalization Quality**     | % of raw values correctly mapped to controlled values (C)     | > 95% |
| **D→C Elevation Efficiency**  | % of candidate terms elevated that remain stable in C         | > 80% |
| **Version Impact Scope**      | #/% of items affected when taxonomy or vocab changes          | < 5% |
| **User-Facing Success**       | Precision@K, Recall@K in search, recommendations, retrieval   | task-dependent |

> Source: derived from **tech_draft.md** governance/evaluation section:contentReference[oaicite:0]{index=0}.

---

## 2. Datasets

### 2.1 Gold Sets
- **Purpose:** Validate extraction and normalization.
- **Composition:** Small hand-labeled sets of product titles/descriptions with:
  - Expected category (A)
  - Expected attributes (C)
  - Candidate terms (D)
- **Format:** JSONL following [DATA_SCHEMA.md](DATA_SCHEMA.md).

### 2.2 Synthetic Sets
- **Purpose:** Stress-test edge cases (multilingual, ambiguous terms).
- **Generation:** Combine templates + synonyms + controlled vocab.

### 2.3 Live Samples
- **Purpose:** Validate pipeline robustness with real-world messiness.
- **Source:** Random samples from ingested raw data.
- **Use:** Spot checks, regression testing.

---

## 3. Protocols

### 3.1 Extraction Accuracy
- Input: Raw item text
- Output: JSON (see LLM Extraction Schema in [DATA_SCHEMA.md](DATA_SCHEMA.md))
- Scoring:
  - Precision / Recall for attributes
  - Term-level clustering accuracy
  - Evidence alignment

### 3.2 Normalization Quality
- Compare raw → normalized mappings against gold controlled vocab.
- Metrics:
  - **Exact Match %**
  - **Alias Resolution Accuracy**
  - **False Merge Rate**

### 3.3 D→C Elevation
- Track:
  -  proposals submitted
  - % approved
  - Stability (do they persist across 3+ releases?)
- Regression: Ensure no duplicate values creep into C.

### 3.4 Retrieval & Ranking
- Evaluate hybrid Qdrant (vector) + Neo4j (graph) retrieval:
  - Precision@K, Recall@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (nDCG)

### 3.5 Versioning & Rollback
- For each release:
  - Record changes (added/removed/merged values)
  - Measure % of items requiring reprocessing
  - Run regression tests to check compatibility

---

## 4. Tooling

- **DeepEval**: scenario-driven LLM evaluation
- **pytest**: integration with gold sets
- **DuckDB**: quick aggregations & metrics
- **Qdrant / Neo4j queries**: retrieval evaluation

---

## 5. Reporting

- CI must run evaluation scripts against gold sets.
- Weekly/Monthly reports:
  - Coverage trend
  - Normalization drift
  - Elevation throughput
- Reports stored as CSV + Markdown in `/reports/`.

---

## 6. Future Extensions

- User-centric evaluation (search click-through, recommendation acceptance).
- Multi-lingual evaluation sets (zh-TW, zh-CN, en).
- Cross-domain validation (media, food, travel).
