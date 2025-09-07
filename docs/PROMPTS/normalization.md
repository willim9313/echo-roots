# Prompt — Normalization & Mapping (C-layer)

## Purpose
Normalize extracted attributes using the C-layer controlled vocab; propose merges/splits when needed.

## Guidance
- Use existing values where possible; otherwise output a **mapping** proposal:
```json
{
  "mapping": { "from": "string", "to": "string", "type": "alias|merge|replace" },
  "version": "C-YYYY.MM"
}

Notes

Record language as zh|en|mixed.

Keep auditability (who/when/why) in pipeline metadata.