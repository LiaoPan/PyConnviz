# PyPI Installation README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. The user prohibited subagents, so
> execution remains inline.

**Goal:** Clearly document installation from the published PyPI project in
both language versions of the README.

**Architecture:** Extend the existing installation section rather than adding
a second top-level section. One identical shell block covers core,
interactive, and static-export installations; localized prose explains the
choices and links to the canonical PyPI page. Existing source-development
instructions remain in place.

**Tech Stack:** Markdown, pytest, PEP 517/setuptools, Twine.

## Global Constraints

- Canonical URL: `https://pypi.org/project/pyconnviz/`.
- Do not pin the install command to version `0.1.0`.
- Core: `python -m pip install --upgrade pyconnviz`.
- Interactive: `python -m pip install --upgrade "pyconnviz[interactive]"`.
- Static Plotly export:
  `python -m pip install --upgrade "pyconnviz[interactive,export]"`.
- Keep fenced code blocks identical between `README.md` and
  `README.zh-CN.md`.
- Do not modify package dependencies, APIs, `LICENSE`, or generated figures.
- Do not use subagents or push; local commits are authorized.

---

### Task 1: Add a failing PyPI-install documentation contract

**Files:**

- Modify: `tests/test_packaging.py`
- Verify: `README.md`
- Verify: `README.zh-CN.md`

**Interfaces:**

- Both READMEs expose the canonical PyPI URL and the same three commands.
- English uses `### Install from PyPI`; Chinese uses `### 从 PyPI 安装`.

- [ ] **Step 1: Write the failing test**

Add a parameterized assertion over both README paths:

```python
@pytest.mark.parametrize(
    ("filename", "heading"),
    (("README.md", "### Install from PyPI"),
     ("README.zh-CN.md", "### 从 PyPI 安装")),
)
def test_readmes_document_installation_from_pypi(filename: str, heading: str) -> None:
    readme = (ROOT / filename).read_text(encoding="utf-8")
    assert heading in readme
    assert "https://pypi.org/project/pyconnviz/" in readme
    assert "python -m pip install --upgrade pyconnviz" in readme
    assert 'python -m pip install --upgrade "pyconnviz[interactive]"' in readme
    assert (
        'python -m pip install --upgrade "pyconnviz[interactive,export]"'
        in readme
    )
```

- [ ] **Step 2: Verify RED**

Run `.venv/bin/python -m pytest -q tests/test_packaging.py` and confirm the new
test fails because the subsection and upgrade commands are absent.

---

### Task 2: Implement, verify, and package the installation guide

**Files:**

- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/superpowers/plans/2026-09-02-pypi-install-readme.md`

**Interfaces:**

- The existing top-level installation headings and source-development blocks
  remain unchanged.
- The new PyPI command block is byte-for-byte identical in both READMEs.

- [ ] **Step 1: Add both localized subsections**

Insert the PyPI link and the exact three-command bash block immediately after
the Python version sentence in each README. Explain core, interactive, and
static-export purposes in localized prose.

- [ ] **Step 2: Verify GREEN and bilingual parity**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_packaging.py
.venv/bin/python -m ruff check tests/test_packaging.py
git diff --check
```

- [ ] **Step 3: Commit the README change**

Stage only both READMEs and `tests/test_packaging.py`, then commit:

```bash
git commit -m "docs: document installation from PyPI"
```

- [ ] **Step 4: Run repository and distribution verification**

Run the full test suite with the 80% coverage gate. Build a wheel and sdist in
a fresh temporary directory, run Twine on both files, and confirm the sdist
contains both README files. Record that the user's separate `LICENSE` edit
remains unstaged and no push occurred.
