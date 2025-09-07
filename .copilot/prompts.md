# Copilot Prompt Library

This file contains **canonical prompts** for Copilot or coding agents.  
They serve as examples of how to interact with the system in a consistent way.

---

## 1. Taxonomy Extraction

**Task:** Extract attributes and candidate terms from raw item text.

**Prompt Example:**
Given the following product text:

"【台灣現貨】Nike Air Zoom Pegasus 39 男鞋 運動 跑鞋 白黑色"

Extract attributes and candidate terms following docs/DATA_SCHEMA.md#2-llm-extraction-schema.

Return only valid JSON in this shape:
{
"item_id": "string",
"attributes": [...],
"terms": [...],
"metadata": {...}
}


---

## 2. Normalization & Mapping

**Task:** Normalize raw attribute values against the controlled vocabulary (C-layer).

**Prompt Example:**
Input:
attribute: "color"
value_raw: "深藍色"
context: "商品描述中顯示"

Action:

Map to normalized value if exists in C

Otherwise output mapping proposal (see docs/DATA_SCHEMA.md#4-mapping-schema)


---

## 3. D→C Elevation Proposal

**Task:** Judge candidate terms in D for elevation to C.

**Prompt Example:**
Candidate terms: ["rose gold", "玫瑰金"]
Attribute: "color"
Frequency: 87
Coverage: 0.63

Decide if elevation is justified. Output JSON following docs/DATA_SCHEMA.md#3-elevation-proposal-schema.

---

## 4. Retrieval & Ranking

**Task:** Use vector DB (Qdrant) recall + Neo4j filtering for hybrid search.

**Prompt Example:**
Query: "running shoes black"
Steps:

Search Qdrant embeddings collection semantic_terms.

Expand with Neo4j using Category=A:shoes, Attribute=color:black.

Rank results by score + evidence count.

Return JSON:
[
{ "id": "item123", "score": 0.91, "why": "...", "provenance": {...} }
]

---

## 5. Governance Workflow

**Task:** Merge/split attributes or deprecate values with versioning.

**Prompt Example:**
Request: Merge "bluish" → "blue"
Output mapping:
{
"mapping_id": "merge-2025-001",
"from_term": "bluish",
"to_term": "blue",
"relation_type": "merge",
"valid_from": "2025-09-06T00:00:00Z",
"created_by": "copilot"
}

---

## Notes
- All examples assume schemas defined in `docs/DATA_SCHEMA.md`.  
- Always return **valid JSON** when the task requires structured output.  
- Copilot should **prioritize controlled vocab (C)** and use D only for candidate exploration.  
- Keep logs and metadata (`model`, `run_id`, `timestamp`) in every JSON response.
