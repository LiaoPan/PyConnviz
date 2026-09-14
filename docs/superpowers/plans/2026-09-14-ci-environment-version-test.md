# CI Environment-Version Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the upstream environment-version test validate the versions actually installed by each CI job instead of one hard-coded lockfile snapshot.

**Architecture:** Change only the test oracle. The production collector continues to use `importlib.metadata.version()`, and the test independently calls the same standard-library metadata API for the packages under assertion.

**Tech Stack:** Python 3.10+, `importlib.metadata`, pytest, Ruff

## Global Constraints

- Do not change production code, dependency ranges, `constraints-dev.txt`, or the GitHub Actions workflow.
- Preserve the exact expected environment-version key set.
- Do not use subagents or push; local commits are allowed.

---

### Task 1: Replace fixed package versions with environment-aware assertions

**Files:**
- Modify: `tests/test_upstream_validation.py`

**Interfaces:**
- Consumes: `importlib.metadata.version(distribution_name) -> str`
- Verifies: `environment_versions() -> dict[str, str]`

- [ ] **Step 1: Use the CI failure as the red-state reproduction**

The supplied CI run fails because `versions["nilearn"]` is `0.14.1` while the
test expects `0.14.0`. Confirm locally that the same test passes only because
the local installed Nilearn version is 0.14.0.

- [ ] **Step 2: Replace the hard-coded assertions**

Import `version` from `importlib.metadata`, then assert:

```python
assert all(isinstance(value, str) and value for value in versions.values())
assert versions["nilearn"] == version("nilearn")
assert versions["mne-connectivity"] == version("mne-connectivity")
```

- [ ] **Step 3: Run focused and module verification**

```bash
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest \
  tests/test_upstream_validation.py::test_environment_versions_records_validation_stack -q
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest \
  tests/test_upstream_validation.py -q
```

Expected: both commands pass.

- [ ] **Step 4: Run CI-equivalent quality checks**

```bash
.venv/bin/python -m ruff check src tests examples scripts
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest -q \
  --cov=pyconnviz --cov-report=term-missing --cov-fail-under=80
```

Expected: Ruff passes, all tests pass, and coverage remains at least 80%.

- [ ] **Step 5: Commit locally**

```bash
git add tests/test_upstream_validation.py
git add -f docs/superpowers/specs/2026-09-14-ci-environment-version-test-design.md \
  docs/superpowers/plans/2026-09-14-ci-environment-version-test.md
git commit -m "test: accept installed validation stack versions"
```

Do not push.
