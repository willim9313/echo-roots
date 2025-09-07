# Project Roadmap

This roadmap outlines milestones and scoped tasks for the taxonomy framework project.  
It helps track **short-term deliverables**, **medium-term improvements**, and **long-term goals**.

---

## 1. Current Focus (Q3–Q4 2025)

- ✅ Establish documentation hub (`/docs`)
- ✅ Define taxonomy framework (A/C/D layers)
- ✅ Draft data schemas (JSON, SQL, DuckDB, Qdrant, Neo4j)
- ✅ Initial CI pipeline (lint, format, tests)
- 🚧 Implement baseline pipelines:
  - Ingestion → Extraction → Normalization → Storage
  - Elevation workflow (D→C proposals)

---

## 2. Near-Term (Next 3–6 Months)

- **Evaluation**
  - Gold dataset creation for 3 domains (e-commerce, media, travel)
  - Automated pytest metrics (coverage, normalization, elevation efficiency)

- **Governance**
  - Elevation judge UI for reviewing proposals
  - Version snapshot management with rollback support

- **Retrieval**
  - Hybrid search (Qdrant + Neo4j filter)
  - Ranking module with nDCG and MRR evaluation

- **Ops**
  - Playbooks for: add category, retrain embeddings, rollback release
  - Monitoring dashboards (coverage, drift, latency)

---

## 3. Mid-Term (6–12 Months)

- **Scalability**
  - Batch + streaming ingestion support
  - Distributed storage options (Postgres/BigQuery for scale-out)

- **Multilingual**
  - Full zh-TW, zh-CN, en support
  - Term alignment across languages

- **Evaluation**
  - Multi-domain evaluation (gaming, food, travel)
  - Longitudinal drift analysis (taxonomy stability over time)

- **User-facing**
  - Simple query API (GraphQL/REST)
  - Frontend governance dashboard

---

## 4. Long-Term (12+ Months)

- **Knowledge Graph Expansion**
  - Cross-domain relationships (e.g., category-to-category links)
  - Ontology-lite layer for reasoning (lightweight constraints)

- **Advanced Features**
  - Semi-automated ontology suggestion (Propp/Vogler archetypes for StorySphere-like domains)
  - Attribute importance weighting for personalization

- **Ecosystem**
  - Integration with RAG pipelines
  - Export formats for 3rd-party tools (Neo4j, Elastic, KGDBs)

---

## 5. Out of Scope (for Now)

- Heavy ontology engineering (rigid OWL/RDF)
- End-user product recommender UI (only APIs planned)
- Proprietary model training (focus on orchestration, not model building)

---

## 6. Tracking

- Roadmap is updated quarterly.
- Significant design decisions recorded in `docs/DECISIONS/` (ADRs).
- Progress is tagged in GitHub milestones.
