# Operations Guide

This document provides **runbooks, troubleshooting notes, and SRE guidance** for managing the taxonomy system in production.

---

## 1. Daily Operations

### 1.1 Pipelines
- **Ingestion**: Monitor logs for missing/duplicate items.
- **Processing**: Ensure normalization runs complete without schema drift.
- **Storage**:
  - DuckDB: Check space & Parquet partition sizes.
  - Qdrant: Monitor memory usage, shard balance.
  - Neo4j: Monitor heap and page cache.

### 1.2 Metrics
- Coverage (A/C/D) trend should remain stable.
- Elevation proposals should move from D→C regularly.
- Watch for sudden drops in normalization accuracy.

---

## 2. Deployment

### 2.1 Environments
- **Local**: DuckDB + in-memory Neo4j/Qdrant.
- **Staging**: Full stack with smaller datasets.
- **Production**: Scaled storage, CI/CD deployment.

### 2.2 CI/CD
- CI: Lint, format, test (`.github/workflows/ci.yml`).
- CD: Versioned deployments, tagged with release notes.

### 2.3 Configuration
- Stored in `configs/*.yaml`.
- Use environment variables for secrets.

---

## 3. Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| **Pipeline stalls** | Schema mismatch, invalid JSON | Validate against `docs/DATA_SCHEMA.md` |
| **Neo4j OOM** | Large queries without limits | Add `LIMIT`/`SKIP`, increase page cache |
| **Qdrant recall too low** | Wrong embedding model | Check embedding dimensions, refresh index |
| **DuckDB queries slow** | Oversized Parquet files | Re-partition, use column pruning |
| **Normalization drift** | New uncontrolled terms in input | Propose elevation or mapping |

---

## 4. Incident Response

### 4.1 Runbook — Pipeline Failure
1. Check logs in `logs/`.
2. Validate JSON contract for failing stage.
3. Run pipeline step locally with minimal input.
4. If schema change needed → file ADR in `docs/DECISIONS/`.

### 4.2 Runbook — Bad Elevation
1. Identify proposal ID in logs.
2. Roll back C-layer to previous snapshot (stored in `/snapshots/`).
3. Update mapping with `relation_type=deprecate`.
4. Reprocess affected items.

### 4.3 Runbook — Retrieval Drift
1. Run evaluation suite (`pytest -m retrieval`).
2. Compare metrics to baseline in `docs/EVALUATION.md`.
3. If embeddings outdated → re-embed candidate set in Qdrant.

---

## 5. Monitoring & Alerting

- **Dashboards**: Grafana/Metabase on metrics tables.
- **Alerts**:
  - Coverage < threshold (A/C/D).
  - Elevation proposals stuck > 7 days.
  - API latency > 500ms (p95).
  - Storage utilization > 80%.

---

## 6. Backup & Recovery

- **DuckDB**: nightly snapshot to S3/GCS.
- **Neo4j**: use `neo4j-admin dump` weekly.
- **Qdrant**: snapshot collections (follow vendor guide).
- Verify backups monthly via restore test.

---

## 7. SRE Notes

- Keep taxonomy releases small and versioned (avoid mass merges).
- Always test governance proposals in staging before prod.
- Rollbacks should rely on `taxonomy_snapshots` + mapping tables.
- Document every incident with date, cause, fix, and prevention.

---

## 8. Contacts

- **Primary Maintainer**: [Your Name / Team Alias]
- **Security Reports**: see [SECURITY.md](../SECURITY.md)
- **Ops Issues**: open an internal ticket or GitHub issue with `ops` label
