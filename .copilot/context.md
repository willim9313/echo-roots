# Copilot Context Guide

This file tells Copilot (and other coding agents) **what to read first**, **where the source of truth lives**, and **how to answer** when spec conflicts arise.

## 1) Priority Reading Order (Source of Truth)

1. docs/TAXONOMY.md — core A/C/D framework and governance flows  
   - A: classification tree; C: controlled attributes; D: semantic candidate network:contentReference[oaicite:0]{index=0}  
   - Evolution stages & governance (merge/split/deprecate, D→C elevation):contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}

2. docs/DATA_SCHEMA.md — canonical data contracts  
   - JSON contracts for ingestion, extraction, elevation, mappings:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}  
   - SQL & DuckDB schemas aligned with graph/vector stores:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}

3. docs/ARCHITECTURE.md — system layers & storage model  
   - Ingestion → Processing → Storage → Serving → Governance:contentReference[oaicite:7]{index=7}  
   - Graph (Neo4j), Vector (Qdrant), DuckDB as lightweight analytics:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}

4. docs/TECH_STACK.md — implementation choices & rationale  
   - Neo4j (A/C), Qdrant (D), optional OpenSearch; LlamaIndex + vLLM/Gemini; tiktoken/DeepEval:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}

## 2) Golden Rules for Suggestions

- **Don’t invent attributes or values**. Use C-layer controlled vocab; propose new ones via D→C elevation workflow:contentReference[oaicite:12]{index=12}.  
- **Preserve raw vs normalized** (dual-track) in all pipelines:contentReference[oaicite:13]{index=13}.  
- **Version everything** (A/C/D), provide old→new mappings and rollback notes:contentReference[oaicite:14]{index=14}.  
- **Prefer property graph** semantics for A/C; treat D as a candidate network fused with vector retrieval:contentReference[oaicite:15]{index=15}:contentReference[oaicite:16]{index=16}.

## 3) Data Models to Trust

- **Graph nodes/edges** for Category/Attribute/Value/Item/Term/Evidence/Mapping, plus projection views for “A-only” and “A+C” queries:contentReference[oaicite:17]{index=17}.  
- **Qdrant collections** for term/snippet embeddings with payload fields (lang, evidence, state):contentReference[oaicite:18]{index=18}.  
- **DuckDB tables** mirroring SQL schema; use Parquet + `CREATE TABLE AS SELECT` or `CREATE VIEW` for performance:contentReference[oaicite:19]{index=19}:contentReference[oaicite:20]{index=20}.

## 4) JSON Contracts (IO)

- Ingestion item, LLM extraction output, D→C proposal, and mapping updates — exact field shapes here; **return valid JSON only**:contentReference[oaicite:21]{index=21}:contentReference[oaicite:22]{index=22}.

## 5) Minimal Metrics (for CI gates & dashboards)

Track: Coverage (A/C/D), Normalization quality, D→C elevation efficiency, Version impact, and user/search experience:contentReference[oaicite:23]{index=23}.

## 6) When in Doubt

- Ask for missing definitions with an elevation proposal draft (don’t silently add).  
- Favor **L1/L2/L3 lightweight storage paths** unless scale dictates otherwise:contentReference[oaicite:24]{index=24}.  
- Keep prompts and schemas in sync with `/docs/PROMPTS` & `DATA_SCHEMA.md`.
