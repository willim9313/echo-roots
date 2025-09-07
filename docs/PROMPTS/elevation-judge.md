# Prompt — D→C Elevation Judge

## Goal
Decide whether candidate terms from layer D should be elevated into C (controlled vocabulary).

## Inputs
- Candidate term cluster with frequencies, evidence, language mix.
- Prior controlled values in C for the target attribute.

## Criteria (align with governance)
- Stability and frequency thresholds; evidence sufficiency.  
- Collision/alias/merge checks against existing C values (map old→new):contentReference[oaicite:34]{index=34}.  
- Version impact scope & rollback friendliness (record mapping):contentReference[oaicite:35]{index=35}.

## Output (JSON; use this proposal schema)
```json
{
  "proposal_id": "string",
  "attribute": { "name": "string", "data_type": "enum" },
  "values": [{ "label": "string", "aliases": ["..."], "lang": "mixed" }],
  "evidence": ["evidence_id1", "evidence_id2"],
  "notes": "reasoning summary"
}
(Contract source) tech_draft.


