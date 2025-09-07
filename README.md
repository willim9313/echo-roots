# EchoRoots

This project(EchoRoots) provides a practical framework for **taxonomy construction, attribute normalization, and semantic enrichment** in domains like e-commerce, media, and knowledge graphs.

## Quick Start

1. Read [docs/OVERVIEW.md](docs/OVERVIEW.md) to understand the project scope.
2. Explore [docs/TAXONOMY.md](docs/TAXONOMY.md) for the domain model.
3. Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design.
4. Check [docs/TECH\_STACK.md](docs/TECH_STACK.md) for chosen technologies.

### Run with a Domain Pack

By default the framework runs with a **generic schema**.
To adapt to a specific domain, provide a **Single-File Domain Pack** (`domain.yaml`):

```bash
uv run app --domain-pack domains/ecommerce
uv run app --domain-pack domains/zh-news
```

A `domain.yaml` defines:

* **input\_mapping** → map source fields (e.g., `name`, `desc`) to core keys (`title`, `description`)
* **output\_schema** → specify additional attributes or constraints
* **rules / prompts** (optional) → normalization maps, blocked terms, or custom extraction prompts

👉 See `domains/ecommerce/domain.yaml` for a working example.

## Repo Layout

* `src/` → pipelines, taxonomy, retrieval, models, utils
* `configs/` → schema & pipeline configs
* `domains/` → domain packs (`domain.yaml` files for each domain)
* `docs/` → knowledge hub
* `tests/` → unit & integration tests

## Documentation Index

See [docs/OVERVIEW.md](docs/OVERVIEW.md) for a full table of contents.
For details on extending to new domains, see [docs/DATA\_SCHEMA.md](docs/DATA_SCHEMA.md).

---

## Copilot & Coding Agents

This repository is optimized for GitHub Copilot and other coding agents.
We provide a `.copilot/` folder to guide them toward the **right context** and **prompt patterns**.

* **.copilot/context.md**
  Defines the **priority reading order** of docs (e.g., TAXONOMY.md, DATA\_SCHEMA.md, ARCHITECTURE.md).
  Agents should always consult this file to understand **what is authoritative**.

* **.copilot/prompts.md**
  Contains a **prompt library** with canonical examples (extraction, normalization, retrieval, governance).
  These ensure Copilot generates **consistent, schema-valid outputs**.

> 💡 Tip: If Copilot seems confused, check `.copilot/context.md` to ensure the right docs are being loaded, and use `.copilot/prompts.md` as starting points for new tasks.

---
