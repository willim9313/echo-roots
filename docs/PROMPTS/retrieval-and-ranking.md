# Prompt — Retrieval & Ranking (Qdrant + Graph)

## Objective
Combine vector recall (Qdrant) with structural filtering/expansion (Neo4j).

## Playbook
1. Embed query → search Qdrant collection(s) for terms/snippets:contentReference[oaicite:38]{index=38}.
2. Take top-K → fetch related nodes/paths in Neo4j; restrict by A/C tags (projection views):contentReference[oaicite:39]{index=39}.
3. Re-rank using:
   - Match to target attribute/value in C (exact > alias > candidate).  
   - Evidence count / last_seen freshness from payload.  
4. Return item/value/term IDs with scores and explanation features.

## Output
A JSON list of results with `id`, `score`, `why` (features used), and `provenance` (Qdrant payload + graph hops).