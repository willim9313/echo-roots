# EchoRoots

This project(EchoRoots) provides a practical framework for **taxonomy construction, attribute normalization, and semantic enrichment** in domains like e-commerce, media, and knowledge graphs.

## Requirements

- **Python 3.13+** 
- **UV package manager** (recommended) or pip
- **Git** for version control

## Quick Start

### 1. Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/willim9313/echo-roots.git
cd echo-roots

# Install UV package manager (if not already installed)
pip install uv

# Set up the project (creates virtual environment, installs dependencies)
python scripts/setup.py
```

### 2. Verify Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Run tests
uv run pytest

# Check code quality
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Verify CLI is working
uv run echo-roots --help
```

### 3. Run with a Domain Pack

By default the framework runs with a **generic schema**.
To adapt to a specific domain, provide a **Single-File Domain Pack** (`domain.yaml`):

```bash
uv run echo-roots --domain-pack domains/ecommerce
uv run echo-roots --domain-pack domains/zh-news
```

A `domain.yaml` defines:

* **input\_mapping** → map source fields (e.g., `name`, `desc`) to core keys (`title`, `description`)
* **output\_schema** → specify additional attributes or constraints
* **rules / prompts** (optional) → normalization maps, blocked terms, or custom extraction prompts

👉 See `domains/ecommerce/domain.yaml` for a working example.

### 4. Development Workflow

```bash
# Run development utilities
python scripts/dev.py test                    # Run tests
python scripts/dev.py test --coverage         # Run with coverage
python scripts/dev.py lint --fix              # Auto-fix linting issues
python scripts/dev.py typecheck               # Run type checking
python scripts/dev.py clean                   # Clean build artifacts
```

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
