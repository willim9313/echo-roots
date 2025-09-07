# Contributing Guidelines

Thank you for your interest in contributing!  
This project focuses on **taxonomy frameworks, data pipelines, and semantic enrichment**.  
To keep contributions consistent and usable by Copilot/coding agents, please follow these guidelines.

---

## 1. Getting Started
- Fork the repo and create a feature branch:
  ```bash
  git checkout -b feature/my-feature
  ```

* Keep changes atomic: one feature or fix per pull request (PR).
* Run tests locally before submitting (`pytest` in `tests/`).

---

## 2. Code Style

* **Python 3.13+** required.
* Follow **PEP8** with [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.
* Use **type hints** and **docstrings**:

  ```python
  def normalize_attribute(value: str) -> str:
      """
      Normalize raw attribute values (e.g., casing, spacing).
      """
      ...
  ```
* Prefer **Pydantic models** for schema validation.
* Keep functions small and single-purpose.

---

## 3. Commit Messages

* Use [Conventional Commits](https://www.conventionalcommits.org/):

  * `feat:` new feature
  * `fix:` bug fix
  * `docs:` documentation update
  * `refactor:` code change that doesn’t affect behavior
  * `test:` adding or updating tests
  * `chore:` maintenance

Example:

```
feat(taxonomy): add D→C elevation scoring logic
```

---

## 4. Documentation

* Update relevant files under `/docs` when adding or modifying features:

  * Taxonomy logic → `TAXONOMY.md`
  * Data models → `DATA_SCHEMA.md`
  * Pipeline design → `ARCHITECTURE.md`
* Add comments for new JSON contracts or prompts.
* Cross-link docs where possible.

---

## 5. Testing

* Write unit tests for new modules in `tests/`.
* Use **pytest** with clear naming (`test_*.py`).
* Mock external APIs (Gemini, vLLM, Qdrant) when possible.
* All PRs must pass GitHub Actions CI.

---

## 6. Pull Requests

* Ensure your branch is rebased on `main`.
* Include:

  * Description of changes
  * Reference to related issue(s)
  * Checklist:

    * [ ] Tests added/updated
    * [ ] Docs updated
    * [ ] Linting passed

---

## 7. Governance & ADRs

* Significant changes to taxonomy or schema require an **ADR** in `docs/DECISIONS/`.
* ADRs must explain motivation, alternatives, and final decision.

---

## 8. Questions & Support

* Open a GitHub Issue with `question` label.
* For urgent matters, check `docs/PLAYBOOKS/` for how-to guides.
