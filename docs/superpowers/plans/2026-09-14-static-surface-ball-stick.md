# Static Surface True-3D Ball-and-Stick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. The user prohibited subagents, so
> execution remains inline.

**Goal:** Render Matplotlib three-view and native Nilearn six-view surface
networks as lit physical spheres, tubes, and optional cones.

**Architecture:** Extend the deterministic triangular primitives with a cone,
then add a Matplotlib adapter that batches those meshes into named
`Poly3DCollection` artists. Both static renderers retain their current data and
surface pipelines but replace screen-space scatter, line, and quiver artists
with the shared adapter.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib 3.7+
`Poly3DCollection`/`LightSource`, Nilearn 0.14, pytest, Pillow.

## Global Constraints

- Do not change connectivity preparation, thresholding, selected edges, node
  ordering, panel keys, or native Nilearn cortical rendering.
- Keep PNG, SVG, PDF, `show=False`, colorbars, and rasterized-cortex behavior.
- Use physical sphere/tube diameters in surface coordinates; FreeSurfer
  surface-RAS uses millimetres.
- Add `fixed_node_diameter_range=[6.0, 16.0]` to every style; keep current
  `fixed_edge_width_range` values as static tube-diameter defaults.
- Use 8 sphere latitude steps, 12 longitude steps, and 8 tube/cone sides.
- Keep Matplotlib/Nilearn imports lazy where they are currently lazy; add no
  dependency.
- Local commits are allowed; never push.

---

### Task 1: Shared cone and Matplotlib mesh adapter

**Files:**

- Modify: `src/pyconnviz/plotting/_plotly_primitives.py`
- Create: `src/pyconnviz/plotting/_matplotlib_ball_stick.py`
- Modify: `tests/test_plotly_primitives.py`
- Create: `tests/test_matplotlib_ball_stick.py`

**Interfaces:**

- Produces `cone_mesh(base_center, tip, radius, *, sides) -> TriangleMesh`.
- Produces `BallStickCollection(Poly3DCollection)` with `diameters`,
  `mesh_vertex_count`, and `mesh_triangle_count` audit attributes.
- Produces `sphere_collection(...)`, `tube_collection(...)`, and
  `cone_collection(...)`, each returning one batched collection or `None` for
  an empty input.

- [ ] **Step 1: Add cone and collection RED tests**

Test that a cone has one tip, one base-center vertex, `sides` rim vertices,
`2 * sides` nondegenerate outward faces, and exact tip/base coordinates. Test
that two spheres become one `Poly3DCollection` with gid `pyconnviz-nodes`,
requested diameter metadata, deterministic vertex/triangle counts, and no
polygon outlines. Test tubes similarly with gid `pyconnviz-edges` and exact
RGBA alpha.

- [ ] **Step 2: Verify RED**

Run:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/python -m pytest -q tests/test_plotly_primitives.py \
  tests/test_matplotlib_ball_stick.py
```

Expected: collection fails because `cone_mesh` and
`_matplotlib_ball_stick` do not exist.

- [ ] **Step 3: Implement `cone_mesh`**

Validate finite `(3,)` endpoints, distinct base/tip, positive radius, and at
least three sides. Build an orthonormal base frame, a rim, outward side faces,
and an outward-facing triangular base cap.

- [ ] **Step 4: Implement batched collections**

Create a fixed `LightSource(azdeg=315, altdeg=45)`. Convert each mesh to
`mesh.vertices[mesh.faces]`; repeat its source RGBA once per triangle; create
one `BallStickCollection` with `shade=depth_cue`, `zsort="average"`, no edge
stroke, the stable gid, physical diameter metadata, and z-order 10/12/11 for
edges/nodes/directions respectively.

- [ ] **Step 5: Verify GREEN and commit**

Run the two test files and Ruff for the four files, then commit:

```bash
git commit -m "feat: add shared Matplotlib ball-and-stick meshes"
```

---

### Task 2: Matplotlib three-view renderer

**Files:**

- Modify: `src/pyconnviz/styles.py`
- Modify: `src/pyconnviz/plotting/surface_matplotlib.py`
- Modify: `tests/test_styles.py`
- Modify: `tests/test_surface_matplotlib.py`

**Interfaces:**

- Consumes Task 1 `sphere_collection`, `tube_collection`, and
  `cone_collection`.
- Produces named physical mesh artists without network
  `Path3DCollection`/`Line3DCollection` artists.

- [ ] **Step 1: Replace artist assertions with RED mesh assertions**

Require each populated panel to contain `pyconnviz-nodes` and
`pyconnviz-edges` `BallStickCollection` objects, no network scatter/line
collection, default node diameters spanning 6–16, edge diameters matching
`scale_values(abs(weights), fixed_edge_width_range)`, preserved panel edges,
and a `pyconnviz-directions` collection for a directed fixture.

- [ ] **Step 2: Verify RED**

Run `tests/test_styles.py` and `tests/test_surface_matplotlib.py`. Expected:
style lacks `fixed_node_diameter_range` and renderer still produces scatter
and line collections.

- [ ] **Step 3: Add physical style defaults**

Add `fixed_node_diameter_range: [6.0, 16.0]` to `paper`, `soft`, and `dark`.
Keep the existing point-area `node_size_range` for non-surface renderers.

- [ ] **Step 4: Replace screen-space artists**

Resolve node diameters from explicit `node_size_range` or
`fixed_node_diameter_range`. Keep current Bezier curves and color
normalization, but pass complete curves, physical tube diameters, and RGBA
colors to the shared adapter. Pass node positions/colors/diameters to the
sphere adapter. For directed graphs, create cones ending one target radius
before the center. Remove `depth_cued_line_data`, `Line3DCollection`,
`axis.scatter`, and `axis.quiver` from the network path.

- [ ] **Step 5: Verify GREEN and commit**

Run focused renderer/style tests, public API tests, and Ruff. Commit:

```bash
git commit -m "feat: render Matplotlib surfaces with 3D ball-and-stick meshes"
```

---

### Task 3: Native Nilearn six-view renderer

**Files:**

- Modify: `src/pyconnviz/plotting/surface_nilearn.py`
- Modify: `tests/test_surface_nilearn.py`

**Interfaces:**

- Consumes the same Task 1 collections and Task 2 style defaults.
- Preserves `plot_img_on_surf`, native panel ordering, and all panel keys.

- [ ] **Step 1: Add native RED mesh assertions**

Require every populated cortical axis to contain one named node mesh and one
named edge mesh, no network scatter/line collection, correct per-hemisphere
node centers and diameter metadata, correct edge identities and tube diameter
metadata, and real cone collections for directed input.

- [ ] **Step 2: Verify RED**

Run `tests/test_surface_nilearn.py`. Expected: current native panels expose
`Path3DCollection`, `Line3DCollection`, and quiver rather than named meshes.

- [ ] **Step 3: Replace native network artists**

Resolve physical diameters identically to Matplotlib. Preserve external mesh
alignment, inflated/pial selection, views, hemisphere iteration, and cortex
collections. Replace only the overlay network artists with the shared sphere,
tube, and cone collections.

- [ ] **Step 4: Verify GREEN and commit**

Run native, Matplotlib, primitive, adapter, and public API tests plus Ruff.
Commit:

```bash
git commit -m "feat: render native Nilearn surfaces with 3D ball-and-stick meshes"
```

---

### Task 4: Documentation and deterministic previews

**Files:**

- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `tests/test_packaging.py`
- Modify: `scripts/check_acceptance_artifacts.py`
- Modify: `tests/test_acceptance_scripts.py`
- Modify: `docs/images/surface-matplotlib.png`
- Modify: `docs/images/surface-nilearn.png`

**Interfaces:**

- Both READMEs describe true-3D supplementary static rendering and units.
- Acceptance checks distinguish physical static mesh output from legacy
  scatter/line output through generated metadata.

- [ ] **Step 1: Add documentation/acceptance RED tests**

Require both READMEs to state that supplementary Matplotlib and native Nilearn
use true-3D ball-and-stick `Poly3DCollection` geometry and physical diameters in
millimetres. Extend acceptance manifests with `render_mode`, node/edge mesh
counts, and static renderer names; reject legacy values.

- [ ] **Step 2: Verify RED**

Run packaging and acceptance-script tests. Expected: README phrases and static
manifest metadata are absent.

- [ ] **Step 3: Update documentation and generator/checker**

Describe physical static geometry in both languages without changing code
examples. Record and strictly validate `render_mode="ball-and-stick"`, one
node/edge collection per populated panel, and the renderer identifiers for
both static backends.

- [ ] **Step 4: Generate, inspect, and install previews**

Generate artifacts into a fresh `/private/tmp` directory, run the strict
checker, and inspect `surface_paper.png` and
`surface_nilearn_native_three_views.png` at original resolution. Copy only
verified outputs to the two README gallery image paths and confirm all panels
show shaded spheres and tubes.

- [ ] **Step 5: Verify and commit**

Run documentation/acceptance tests, Ruff, and `git diff --check`; commit:

```bash
git commit -m "docs: show 3D static surface ball-and-stick previews"
```

---

### Task 5: Full verification and distribution audit

**Files:**

- Modify: `docs/superpowers/plans/2026-09-14-static-surface-ball-stick.md`

- [ ] **Step 1: Run quality and CI gates**

Run `compileall`, Ruff, `git diff --check`, then the full suite with
`--cov=pyconnviz --cov-fail-under=80`. Record exact pass count and coverage.

- [ ] **Step 2: Build and inspect clean distributions**

Build wheel/sdist in a fresh temporary directory, run `twine check`, and
verify the wheel contains `_matplotlib_ball_stick.py`; verify the sdist
contains that module, its test, the updated READMEs, and both gallery images.

- [ ] **Step 3: Record and commit verification**

Record exact tests, coverage, visual dimensions, distributions, final status,
and confirmation that no push occurred. Commit only this plan with:

```bash
git commit -m "docs: record static ball-and-stick verification"
```
