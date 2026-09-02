# Depth-aware Surface Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. The user explicitly prohibited
> subagents, so execution stays inline in this session.

**Goal:** Improve the spatial reading of fixed-view Matplotlib and native
Nilearn surface figures, then make Plotly surface HTML and the Nilearn glass
brain the first documented recommendations.

**Architecture:** Add one projection-based, display-only alpha calculation in
`_surface_common.py` and reuse it in both static renderers. Keep every prepared
edge and every scientific encoding unchanged. Change recommendation order in
documentation and regenerate repository evidence with the existing real-data
and deterministic acceptance pipelines.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib 3D collections, Nilearn 0.14,
Plotly, pytest, Ruff.

## Global Constraints

- Do not change `PreparedConnectome.edges`, thresholds, masks, rankings, or
  normalization domains.
- Depth cueing changes alpha only; edge color/width and node color/size retain
  their documented data meanings.
- Keep `plot_connectome` dispatcher defaults for compatibility. Plotly remains
  optional, and the glass backend still requires valid MNI coordinates.
- Use only public Nilearn calls for rendering; do not depend on Nilearn private
  functions.
- Do not use subagents and do not push. Git add/commit is authorized.

---

### Task 1: Shared projected-depth line data

**Files:**

- Modify: `tests/test_surface_matplotlib.py`
- Modify: `src/pyconnviz/plotting/_surface_common.py`

**Interfaces:**

- Produces: `projected_depth_factors(points, reference_points, projection,
  minimum) -> NDArray[np.float64]`.
- Produces: `depth_cued_line_data(curves, colors, widths, *,
  reference_points, projection, minimum, alpha, enabled) -> tuple[list[
  NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]`.
- Consumes: finite `N x 3` coordinates and the public 4x4 matrix returned by
  `Axes3D.get_proj()`.

- [x] **Step 1: Write the failing projection tests**

Add tests that use a simple deterministic 4x4 perspective matrix and assert:

```python
factors = projected_depth_factors(points, reference, projection, minimum=0.2)
assert factors.shape == (len(points),)
assert np.all((0.2 <= factors) & (factors <= 1.0))
assert factors[near_index] > factors[far_index]
```

Also test a zero-depth reference range returns ones, and reject invalid point
shapes, projection shapes, non-finite values, and minimum values outside
`[0, 1]`.

- [x] **Step 2: Write the failing line-data tests**

Pass two 3-point curves with two base RGBA colors and widths. With
`enabled=True`, assert that the result has four adjacent line segments,
repeated widths, preserved RGB channels, and more than one alpha value. With
`enabled=False`, assert that the two original curves remain two paths and all
alphas equal the requested base alpha.

- [x] **Step 3: Verify RED**

Run:

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg \
  .venv/bin/pytest -q tests/test_surface_matplotlib.py \
  -k 'projected_depth or depth_cued_line'
```

Expected: four failures at the calls because the two helpers do not yet exist.

- [x] **Step 4: Implement the minimal shared helpers**

Use homogeneous projection (`homogeneous @ projection.T`) and divide clip-space
z by w. Normalize against projected reference depth using the same near-to-far
direction as Matplotlib's depth shading. Split curves only when enabled and
multiply the base edge alpha by each depth factor.

- [x] **Step 5: Verify GREEN and commit**

Run the focused tests and `ruff check` for the two files, then:

```bash
git add src/pyconnviz/plotting/_surface_common.py tests/test_surface_matplotlib.py
git commit -m "feat: add projected depth cues for surface edges"
```

---

### Task 2: Matplotlib translucent depth-aware surface

**Files:**

- Modify: `src/pyconnviz/styles.py`
- Modify: `src/pyconnviz/api.py`
- Modify: `src/pyconnviz/plotting/surface_matplotlib.py`
- Modify: `tests/test_styles.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_surface_matplotlib.py`

**Interfaces:**

- `plot_surface_matplotlib(..., depth_cue: bool = True) -> PlotResult`.
- `_MATPLOTLIB_OPTIONS` accepts `depth_cue`; other backends reject it unless
  explicitly supported.
- Every style exposes `depth_cue_min_alpha` and the cortex alpha values from the
  design table.

- [x] **Step 1: Write failing style and renderer tests**

Assert exact style defaults (`paper=0.24`, `soft=0.20`, `dark=0.32`) and bounded
`depth_cue_min_alpha`. Replace the uniform-foreground renderer assertion with:

```python
colors = np.asarray(edges.get_colors())
assert len(colors) > len(result.panel_edges["left-lateral"])
assert np.ptp(colors[:, 3]) > 0
assert nodes.get_depthshade() is True
```

Add a `depth_cue=False` case asserting one color/path per visible edge,
uniform alpha, and disabled node depth shading. Assert `panel_edges` is
identical in both calls.

- [x] **Step 2: Write the failing dispatcher test**

Call `plot_connectome(..., engine="matplotlib", depth_cue=False)` through a
capturing renderer and assert the option is forwarded. Passing `depth_cue` to
Plotly, glass, HTML, or circle must remain a strict unexpected-option error.

- [x] **Step 3: Verify RED**

Run the named style, API, and Matplotlib tests. Expected failures: old opacity
values, missing dispatcher option, and uniform edge colors.

- [x] **Step 4: Implement the Matplotlib behavior**

After Nilearn establishes the camera, call `depth_cued_line_data` with
`axis.get_proj()` and `display.surface_coordinates`. Construct one batched
`Line3DCollection` from the returned paths/colors/widths without a collection-
wide alpha override. Set `depthshade=depth_cue` for the node collection.
Validate `depth_cue` is a real boolean before rendering.

- [x] **Step 5: Verify GREEN and commit**

Run all Matplotlib, style, and public API tests plus Ruff, then commit the six
files with:

```bash
git commit -m "feat: make Matplotlib surfaces depth aware"
```

---

### Task 3: Native Nilearn depth-aware network overlay

**Files:**

- Modify: `src/pyconnviz/api.py`
- Modify: `src/pyconnviz/plotting/surface_nilearn.py`
- Modify: `tests/test_public_api.py`
- Modify: `tests/test_surface_nilearn.py`

**Interfaces:**

- `plot_surface_nilearn(..., depth_cue: bool = True) -> PlotResult`.
- `_NILEARN_SURFACE_OPTIONS` accepts `depth_cue`.
- Nilearn's public `plot_img_on_surf(..., alpha=resolved_alpha)` continues to
  create the cortical panels; PyConnviz only overlays depth-cued network
  artists afterward.

- [ ] **Step 1: Write failing native-renderer tests**

Capture `plot_img_on_surf` and assert the paper style now passes `alpha=0.24`.
On real test axes, assert depth-cued edge colors contain multiple alpha levels,
nodes have depth shading enabled, and `depth_cue=False` restores uniform edge
alpha without changing `panel_edges`.

- [ ] **Step 2: Write the failing native dispatcher test**

Assert `depth_cue=False` routes to `engine="nilearn"` and is rejected by
non-static backends.

- [ ] **Step 3: Verify RED**

Run `tests/test_surface_nilearn.py` and the named public API test. Expected:
missing parameter/routing and uniform line alpha.

- [ ] **Step 4: Implement with the shared line-data helper**

Use each Nilearn-returned panel axis projection and the matching display mesh
coordinates. Keep all existing hemisphere edge scopes, colorbars, arrows, and
output formats unchanged. Apply `depthshade=depth_cue` to nodes.

- [ ] **Step 5: Verify GREEN and commit**

Run native, API, and shared surface tests plus Ruff, then commit:

```bash
git commit -m "feat: depth cue native Nilearn surface overlays"
```

---

### Task 4: Recommended-backend documentation and examples

**Files:**

- Modify: `README.md`
- Modify: `examples/04_all_backends.py`
- Modify: `scripts/generate_acceptance_artifacts.py`
- Modify: `scripts/upstream_validation.py`
- Modify: `tests/test_packaging.py`
- Modify: `tests/test_acceptance_scripts.py`

**Interfaces:**

- README first gallery row: Plotly surface and Nilearn glass brain.
- Quickstart first renders Plotly HTML and a glass-brain static file from one
  `PreparedConnectome`; a clearly labelled base-install Matplotlib fallback
  remains included.
- All six rendering modes remain listed with complete example calls.

- [ ] **Step 1: Write failing README/source-contract tests**

Assert the Plotly gallery image occurs before the Matplotlib surface image,
README contains a `## Recommended views` section, and the recommended calls
occur in Plotly-then-glass order. Assert acceptance and upstream scripts use
`depth_cue=True` and no explicit static cortex alpha above `0.24`.

- [ ] **Step 2: Verify RED**

Run the packaging and acceptance source-contract tests. Expected: missing
recommendation section and old ordering/alpha values.

- [ ] **Step 3: Update documentation and examples**

Reorder the gallery and renderer table. Explain that Plotly is the primary
rotatable anatomical view, glass brain the primary static overview, and fixed
surface montages are supplementary camera projections. Update the Quickstart
and all-backend example without changing scientific input. Replace the old
claim that uniform foreground order represents all edges with the new
depth-cue limitation.

- [ ] **Step 4: Update artifact-generation calls**

Use the new defaults or explicit `cortex_alpha<=0.24`, set `depth_cue=True`,
and update titles/labels from “publication” to “depth-aware context” where
appropriate.

- [ ] **Step 5: Verify GREEN and commit**

Run packaging, acceptance, and example compilation tests, then commit:

```bash
git commit -m "docs: recommend Plotly and glass connectivity views"
```

---

### Task 5: Regenerate and inspect visual evidence

**Files:**

- Regenerate: `artifacts/acceptance/*`
- Regenerate: `artifacts/upstream_validation/comparisons/surface_msdl_fsaverage/*`
- Modify: `docs/images/surface-matplotlib.png`
- Modify: `docs/images/surface-nilearn.png`

**Interfaces:**

- Acceptance generator remains deterministic in topology and edge manifests.
- Real MSDL surface case continues to share one top-40 edge set among all
  surface engines.

- [ ] **Step 1: Generate deterministic acceptance artifacts**

Run:

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/python scripts/generate_acceptance_artifacts.py \
  --outdir artifacts/acceptance
.venv/bin/python scripts/check_acceptance_artifacts.py artifacts/acceptance
```

- [ ] **Step 2: Regenerate the cached real MSDL surface comparison**

Run the upstream validator with `--skip-download --strict`, or invoke its
surface case with the checked-in correlation matrix, labels, MNI coordinates,
and `data/fsaverage` when a full audit would perform unrelated work.

- [ ] **Step 3: Refresh the two static gallery images**

Copy the newly generated real-data Matplotlib and native Nilearn PNGs to their
stable `docs/images` names. Do not replace the already preferred Plotly and
glass images unless their source renderer changed.

- [ ] **Step 4: Visual QA at full resolution**

Inspect the two regenerated static surfaces plus Plotly and glass reference
images. Confirm cortex is visibly translucent, near/far edge alpha differs,
edge sign colors remain legible, no statistical surface color appears in pure
connectivity mode, titles are accurate, and panels/colorbars do not collide.

- [ ] **Step 5: Validate manifests and commit generated evidence**

Run strict artifact checks and the surface audit assertions. Stage only files
generated by these commands and commit:

```bash
git commit -m "docs: refresh depth-aware surface examples"
```

---

### Task 6: Complete verification and release check

**Files:**

- Verify: all changed source, tests, docs, examples, and artifacts

**Interfaces:**

- Produces a clean committed working tree ahead of `origin/main`.
- Performs no network push or upload.

- [ ] **Step 1: Run static gates**

```bash
.venv/bin/python -m compileall -q src tests examples scripts
.venv/bin/ruff check src tests examples scripts
git diff --check HEAD~1
```

- [ ] **Step 2: Run complete tests and coverage**

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/pytest -q --cov=pyconnviz \
  --cov-report=term-missing --cov-fail-under=80
```

Expected: all tests pass and total coverage remains at least 80%.

- [ ] **Step 3: Run package build checks**

Build into a fresh temporary directory with `python -m build`, then run
`python -m twine check` against the exact new sdist and wheel paths.

- [ ] **Step 4: Review and commit any final in-scope corrections**

Inspect `git status`, `git diff`, and recent commits. Commit only verified
in-scope corrections. Do not push.

## Plan self-review

Every design requirement maps to one task: shared depth calculation (Task 1),
both static engines and strict routing (Tasks 2–3), recommendation order and
limitations (Task 4), real-data visual evidence (Task 5), and full quality gates
(Task 6). Function signatures and style keys are consistent across tasks. The
plan has no implementation placeholders and keeps all external-state changes
inside authorized local commits.
