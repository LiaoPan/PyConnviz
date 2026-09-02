# PyConnviz Bilingual README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. The user prohibited subagents, so
> execution is inline.

**Goal:** Add a complete Simplified Chinese README with reciprocal language
switching while retaining English as the package metadata README.

**Architecture:** `README.md` remains canonical for PyPI and gains one language
navigation line. `README.zh-CN.md` mirrors its section and code structure in
Chinese, and `MANIFEST.in` includes both files in the sdist. Packaging tests
enforce reciprocal links, code-block identity, gallery/backend coverage, and
release-command coverage.

**Tech Stack:** Markdown, Python 3.10+, pytest, setuptools/PEP 517, Twine.

## Global Constraints

- Keep `pyproject.toml` project `readme = "README.md"`.
- Use the conventional filename `README.zh-CN.md`.
- Preserve every fenced code block verbatim between the two documents.
- Preserve gallery paths and their Plotly/glass/static/circle ordering.
- Translate the full README, including scientific caveats and release steps.
- Add only relative repository links for language switching.
- Do not modify APIs, generated figures, scientific behavior, or `LICENSE`.
- Do not use subagents or push; local commits are authorized.

---

### Task 1: Lock the bilingual documentation contract

**Files:**

- Modify: `tests/test_packaging.py`

**Interfaces:**

- Consumes: `README.md`, `README.zh-CN.md`, `MANIFEST.in`, `pyproject.toml`.
- Produces: packaging assertions that fail until the bilingual files are
  complete and packaged.

- [x] **Step 1: Extend the manifest expectation**

Require the exact manifest sequence to contain `include README.zh-CN.md`
immediately after `include README.md`.

- [x] **Step 2: Add reciprocal-link and completeness tests**

Add helpers and assertions equivalent to:

```python
def _fenced_blocks(markdown: str) -> list[str]:
    return re.findall(r"```[^\n]*\n.*?```", markdown, flags=re.DOTALL)


def test_readmes_offer_language_switch_and_chinese_is_complete() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "[简体中文](README.zh-CN.md)" in "\n".join(english.splitlines()[:8])
    assert "[English](README.md)" in "\n".join(chinese.splitlines()[:8])
    assert _fenced_blocks(chinese) == _fenced_blocks(english)
```

Also require the Chinese major headings, every backend/engine signature, all
five gallery paths, `0.08 / 0.08 / 0.18`, `python -m build`, and
`twine check dist/*`.

- [x] **Step 3: Verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_packaging.py
```

Expected: failures because `README.zh-CN.md` and its manifest line do not yet
exist.

---

### Task 2: Create and package the complete Chinese README

**Files:**

- Modify: `README.md`
- Create: `README.zh-CN.md`
- Modify: `MANIFEST.in`
- Verify: `tests/test_packaging.py`

**Interfaces:**

- `README.md` starts with `English | [简体中文](README.zh-CN.md)`.
- `README.zh-CN.md` starts with `[English](README.md) | 简体中文`.
- `MANIFEST.in` contains both README files.

- [x] **Step 1: Add the English-to-Chinese link**

Insert the navigation line immediately after the English H1 and before its
project description.

- [x] **Step 2: Translate the complete document**

Create `README.zh-CN.md` with the same section order, HTML gallery markup,
fenced code blocks, links, numerical values, and commands. Translate prose,
headings, table labels, image alternative text, and explanatory comments
outside code fences into Simplified Chinese.

- [x] **Step 3: Include the translation in source distributions**

Insert this line after `include README.md`:

```text
include README.zh-CN.md
```

- [x] **Step 4: Verify GREEN and Markdown integrity**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_packaging.py
.venv/bin/python -m ruff check tests/test_packaging.py
git diff --check
```

Expected: every command succeeds.

- [x] **Step 5: Commit the bilingual README**

Stage only `README.md`, `README.zh-CN.md`, `MANIFEST.in`, and
`tests/test_packaging.py`, then commit with:

```bash
git commit -m "docs: add Simplified Chinese README"
```

---

### Task 3: Verify repository and distribution integrity

**Files:**

- Verify: all tracked project files and distribution contents.
- Modify: this plan only to record completed checks.

**Interfaces:**

- Both READMEs remain readable from a source checkout and the sdist.
- The wheel/sdist metadata continues to render `README.md` as English.

- [x] **Step 1: Run static and full test gates**

```bash
.venv/bin/python -m compileall -q src tests examples scripts
.venv/bin/python -m ruff check src tests examples scripts
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/python -m pytest -q --cov=pyconnviz \
  --cov-report=term-missing --cov-fail-under=80
```

- [x] **Step 2: Build and inspect both distributions**

Build into a fresh `/private/tmp/pyconnviz-readme-dist.*` directory, run Twine
against the exact wheel and sdist paths, and inspect the tar listing for both
`README.md` and `README.zh-CN.md`.

- [x] **Step 3: Record final Git state**

Confirm all bilingual README task files are committed, the user's separate
`LICENSE` edit remains unstaged, and no push occurred.

Completion note: both READMEs were verified in the sdist, English remains the
wheel metadata description, all task files are committed, the separate
`LICENSE` edit remains unstaged, and no push occurred.
