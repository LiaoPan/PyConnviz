# Static Surface Edge Legibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. The user explicitly prohibited
> subagents, so execution remains inline.

**Goal:** Keep every scientific edge and the fixed-view depth cue while making
real MSDL connections visible at README thumbnail scale.

**Architecture:** Add fixed-view-only visual defaults to each built-in style.
Matplotlib and native Nilearn consume those defaults when explicit caller
options are absent; Plotly keeps the existing interactive defaults. Regenerate
the same audited top-40 MSDL edge set and inspect both original and thumbnail
renders.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib, Nilearn, Plotly, pytest, Ruff.

## Global Constraints

- Do not change connectivity preparation, edge filtering, ranking, masks, or
  the contents of `PreparedConnectome.edges`.
- Edge RGB and width keep their existing weight semantics; projected depth
  changes alpha only.
- Explicit `cortex_alpha`, `edge_alpha`, and `edge_width_range` arguments must
  override style defaults.
- Do not modify Plotly, glass-brain, HTML, or circle rendering defaults.
- Do not use subagents or push. Local add/commit is authorized.

---

### Task 1: Lock the fixed-view visibility contract with failing tests

**Files:**

- Modify: `tests/test_styles.py`
- Modify: `tests/test_surface_matplotlib.py`
- Modify: `tests/test_surface_nilearn.py`
- Verify: `tests/test_surface_plotly.py`

**Interfaces:**

- Built-in styles produce `fixed_cortex_alpha`, `fixed_edge_alpha`, and
  `fixed_edge_width_range`.
- Static edge alpha must remain depth-varying but never fall below
  `fixed_edge_alpha * depth_cue_min_alpha`.

- [x] **Step 1: Write the failing style contract tests**

Assert the exact fixed-view values from the design table and these invariants:

```python
assert style["fixed_edge_width_range"][0] >= 1.3
assert style["fixed_edge_alpha"] * style["depth_cue_min_alpha"] >= 0.58
assert style["fixed_cortex_alpha"] < style["cortex_alpha"]
```

- [x] **Step 2: Write failing renderer tests**

Render one default paper panel with each static engine. Assert surface face
alpha is `0.08`, minimum line width is at least `1.4`, line alpha varies with
depth, and its minimum is at least `0.665`. Keep the existing explicit override
tests and assert Plotly's soft-style mesh opacity remains `0.20`.

- [x] **Step 3: Verify RED**

Run:

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/pytest -q tests/test_styles.py \
  tests/test_surface_matplotlib.py tests/test_surface_nilearn.py \
  tests/test_surface_plotly.py -k 'fixed or thumbnail or opacity'
```

Expected: failures for missing fixed-view keys and old static values; the
Plotly opacity assertion continues to pass.

---

### Task 2: Implement renderer-specific visibility defaults

**Files:**

- Modify: `src/pyconnviz/styles.py`
- Modify: `src/pyconnviz/plotting/surface_matplotlib.py`
- Modify: `src/pyconnviz/plotting/surface_nilearn.py`

**Interfaces:**

- Static renderers resolve defaults with:

```python
resolved_cortex_default = visual["fixed_cortex_alpha"]
resolved_edge_widths = visual["fixed_edge_width_range"]
resolved_edge_alpha = visual["fixed_edge_alpha"]
```

- [x] **Step 1: Implement exact style values**

Add the three fixed-view keys for `paper`, `soft`, and `dark`, and raise
`depth_cue_min_alpha` to `0.70`, `0.65`, and `0.70`, respectively. Preserve the
existing non-fixed keys so Plotly is unchanged.

- [x] **Step 2: Route defaults through both static renderers**

Use each fixed-view key only when its corresponding public argument is `None`.
Do not change `depth_cued_line_data`, edge selection, normalization, or panel
composition.

- [x] **Step 3: Verify GREEN and regression scope**

Run all four style/surface test modules plus Ruff on the changed files. Confirm
the Plotly test still passes and both static renderers preserve explicit caller
overrides.

- [x] **Step 4: Commit implementation**

```bash
git add src/pyconnviz/styles.py \
  src/pyconnviz/plotting/surface_matplotlib.py \
  src/pyconnviz/plotting/surface_nilearn.py \
  tests/test_styles.py tests/test_surface_matplotlib.py \
  tests/test_surface_nilearn.py
git commit -m "fix: preserve surface edges at thumbnail scale"
```

---

### Task 3: Refresh documentation and visual evidence

**Files:**

- Modify: `README.md`
- Modify: `scripts/upstream_validation.py`
- Regenerate: `artifacts/acceptance/*`
- Regenerate: `artifacts/upstream_validation/comparisons/surface_msdl_fsaverage/*`
- Modify: `docs/images/surface-matplotlib.png`
- Modify: `docs/images/surface-nilearn.png`

**Interfaces:**

- README explains fixed-view defaults and the lateral/whole-brain edge scope.
- The real-data manifest retains exactly 40 prepared edges and identical
  backend edge tuples.

- [x] **Step 1: Update README and generation calls**

Document the fixed-view values and explain that single-hemisphere lateral
panels show only intra-hemisphere edges. Remove the old explicit MSDL
`cortex_alpha=0.20` so the paper fixed-view style is exercised by acceptance
evidence.

- [x] **Step 2: Regenerate deterministic and real MSDL artifacts**

Run the acceptance generator/checker, then invoke `run_surface_case` with the
cached matrix, labels, coordinates, and `data/fsaverage`.

- [x] **Step 3: Refresh stable gallery images and inspect them**

Copy the two real MSDL PNGs to `docs/images`. Inspect original images and
dedicated approximately 499-pixel lateral previews. Confirm all 9 left and 10
right panel edges exist in artists/manifests and are visually traceable; verify
the whole-brain panel retains all 40 edges.

- [x] **Step 4: Commit documentation and evidence**

```bash
git add README.md scripts/upstream_validation.py artifacts/acceptance \
  artifacts/upstream_validation/comparisons/surface_msdl_fsaverage \
  docs/images/surface-matplotlib.png docs/images/surface-nilearn.png
git commit -m "docs: refresh legible surface connectivity examples"
```

---

### Task 4: Complete repository verification

**Files:**

- Verify all source, tests, examples, scripts, documentation, and artifacts.

- [ ] **Step 1: Run static gates**

```bash
.venv/bin/python -m compileall -q src tests examples scripts
.venv/bin/ruff check src tests examples scripts
git diff --check
```

- [ ] **Step 2: Run full CI-equivalent tests**

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/pytest -q --cov=pyconnviz \
  --cov-report=term-missing --cov-fail-under=80
```

- [ ] **Step 3: Build and validate distributions**

Build wheel and sdist into a fresh temporary directory and run Twine against
the exact two output paths. Do not upload.

- [ ] **Step 4: Review local Git state**

Confirm the worktree is clean, list commits ahead of `origin/main`, and report
that no push occurred.
