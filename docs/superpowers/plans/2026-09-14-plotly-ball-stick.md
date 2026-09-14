# Plotly True-3D Ball-and-Stick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. The user prohibited subagents, so
> execution remains inline.

**Goal:** Render Plotly surface nodes as lit three-dimensional spheres and
connections as lit three-dimensional tubes, including deterministic static
four-view output.

**Architecture:** Add a Plotly-independent triangular-primitives module for UV
spheres, parallel-transport tubes, and mesh merging. The Plotly renderer merges
all nodes into one `Mesh3d` trace and all edges into one `Mesh3d` trace, while
preserving scientific mappings and hover data. Directed arrows use a `Cone`
trace.

**Tech Stack:** Python 3.10+, NumPy, Plotly `Mesh3d`/`Cone`, Nilearn, pytest,
Kaleido, Pillow.

## Global Constraints

- Do not change connectivity preparation, thresholding, edge identity, or
  node ordering.
- Keep Plotly optional and lazy-imported; add no dependency.
- Keep `node_size_range=(6.0, 16.0)` and interpret it as sphere diameter in
  surface coordinate units (mm for FreeSurfer geometry).
- Interpret the resolved Plotly `edge_width_range` as tube diameter in the
  same coordinate units.
- Use 10 sphere latitude steps, 16 longitude steps, and 8 tube sides.
- Preserve HTML, PNG, SVG, PDF, hover, colorbar, and `show=False` behavior.
- Preserve the user's `.gitignore` and `LICENSE` edits; never stage them.
- Local commits are allowed; never push.

---

### Task 1: Deterministic triangular primitives

**Files:**

- Create: `src/pyconnviz/plotting/_plotly_primitives.py`
- Create: `tests/test_plotly_primitives.py`

**Interfaces:**

- Produces `TriangleMesh(vertices: float64[n, 3], faces: int64[m, 3])`.
- Produces `sphere_mesh(center, radius, *, latitude_steps, longitude_steps)`.
- Produces `tube_mesh(points, radius, *, sides)`.
- Produces `merge_meshes(meshes)`.

- [x] **Step 1: Add sphere and merge RED tests**

Create tests that assert a sphere centered at `[1, 2, 3]` with radius `2`
has `2 + (latitude_steps - 1) * longitude_steps` vertices,
`2 * longitude_steps * (latitude_steps - 1)` faces, every vertex exactly two
units from the center, valid face indices, and no zero-area face. Assert that
merging two meshes offsets every face in the second mesh by the first mesh's
vertex count.

- [x] **Step 2: Verify sphere/merge RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_plotly_primitives.py
```

Expected: collection fails because `_plotly_primitives` does not exist.

- [x] **Step 3: Implement `TriangleMesh`, `sphere_mesh`, and `merge_meshes`**

Use unique north/south poles and longitude rings for the sphere. Use outward
winding `(north, current, next)`, `(a, b, c)`, `(a, c, d)`, and
`(current, south, next)`. Validate finite `(3,)` centers, finite positive
radii, `latitude_steps >= 3`, `longitude_steps >= 3`, mesh shapes, and face
index ranges. An empty merge returns `(0, 3)` arrays of the declared dtypes.

- [x] **Step 4: Verify sphere/merge GREEN**

Run the primitive tests and Ruff for the new files. Expected: all added tests
pass and Ruff reports no issues.

- [x] **Step 5: Add tube RED tests**

For a straight three-point centerline on the z axis, assert every ring has the
requested radial distance, each ring center equals the source point, face
indices are valid, and outward face normals have positive dot product with the
radial direction. Parameterize invalid input checks for a single point,
duplicate consecutive points, non-positive radius, and fewer than three sides.

- [x] **Step 6: Implement parallel-transport `tube_mesh` and verify GREEN**

Use normalized endpoint/central tangents. Choose the axis least aligned with
the first tangent, then project the previous frame normal into each following
tangent plane. Emit `sides` vertices per ring and two outward triangles per
quad. Run all primitive tests plus Ruff.

- [x] **Step 7: Commit the primitive layer**

Stage only the new module and its test, then commit:

```bash
git commit -m "feat: add deterministic 3D mesh primitives"
```

---

### Task 2: Replace Plotly screen-space nodes and edges

**Files:**

- Modify: `src/pyconnviz/plotting/surface_plotly.py`
- Modify: `tests/test_surface_plotly.py`

**Interfaces:**

- Consumes all Task 1 primitive helpers.
- Produces one `Nodes` `Mesh3d` trace and, for non-empty graphs, one `Edges`
  `Mesh3d` trace.
- Adds trace `meta.diameters`, `meta.vertex_counts`, and figure-level
  `meta.render_mode`, `meta.network_vertices`, `meta.network_triangles`.

- [x] **Step 1: Replace node/edge trace assertions with RED mesh assertions**

Update the coordinate, colormap, and HTML tests to require:

```python
nodes = next(trace for trace in figure.data if trace.name == "Nodes")
edges = next(trace for trace in figure.data if trace.name == "Edges")
assert nodes.type == "mesh3d"
assert edges.type == "mesh3d"
assert not any(
    trace.type == "scatter3d" and trace.mode in {"markers", "lines"}
    for trace in figure.data
)
assert figure.layout.meta["render_mode"] == "ball-and-stick"
```

Check that node and edge hover arrays contain every expected name/direction,
and that `meta.diameters` equals the output of existing `scale_values` calls.

- [x] **Step 2: Verify renderer RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_surface_plotly.py
```

Expected: failures show `Nodes` and `Edge ...` are still `scatter3d` and no
`Edges` mesh exists.

- [x] **Step 3: Add mesh assembly helpers in `surface_plotly.py`**

Add constants for resolution and two lighting dictionaries. Add helpers to:

- convert `TriangleMesh` arrays to `x/y/z/i/j/k` keyword arguments;
- widen equal color limits without changing ordering;
- construct and merge node spheres while repeating intensity/hover arrays;
- construct and merge edge tubes while repeating vertexcolor/hover arrays.

Node radius is scaled diameter divided by two. Tube radius is scaled edge width
divided by two. Use existing Bezier paths without changing their samples or
scientific edge order.

- [x] **Step 4: Replace `Scatter3d` construction with `Mesh3d`**

Create `Edges` before `Nodes`. Set explicit triangle indices,
`flatshading=False`, lighting and light position. Use repeated node intensity
with `colorscale`, `cmin`, `cmax`, `showscale=True`, and the “Node value”
colorbar. Use repeated resolved hex edge colors as `vertexcolor`. Retain the
existing hover template and set both network traces to `showlegend=False`.

- [x] **Step 5: Update warning and metadata**

Change the many-edge warning to describe generated tube geometry rather than
individual traces. Record `render_mode="ball-and-stick"`, edge count, directed
state, and merged network vertex/triangle counts in layout metadata.

- [x] **Step 6: Verify renderer GREEN**

Run the Plotly and public API tests plus Ruff:

```bash
.venv/bin/python -m pytest -q tests/test_plotly_primitives.py tests/test_surface_plotly.py tests/test_public_api.py
.venv/bin/python -m ruff check src/pyconnviz/plotting/_plotly_primitives.py src/pyconnviz/plotting/surface_plotly.py tests/test_plotly_primitives.py tests/test_surface_plotly.py
```

- [x] **Step 7: Commit the mesh renderer**

Stage only the Plotly renderer and Plotly tests, then commit:

```bash
git commit -m "feat: render Plotly connectomes as 3D ball-and-stick meshes"
```

---

### Task 3: Directed arrows and static montage compatibility

**Files:**

- Modify: `src/pyconnviz/plotting/surface_plotly.py`
- Modify: `tests/test_surface_plotly.py`

**Interfaces:**

- Produces an optional `Directions` Plotly `Cone` trace.
- Preserves one node colorbar across all static montage panels.

- [x] **Step 1: Add directed-arrow RED test and retain the montage contract**

Render a directed fixture with `show_arrows=True` and require exactly one
`cone` trace named `Directions`, no `Direction markers` scatter trace, finite
tip/vector arrays, and one hover item per prepared edge. The montage test was
updated in Task 2 because Mesh3d nodes require `trace.showscale` immediately;
it requires `[True, False, False, False]`.

- [x] **Step 2: Verify RED**

Run the focused direction test. Expected: the direction trace is still a
`scatter3d` marker and no `Directions` cone exists.

- [x] **Step 3: Implement 3D cones and retain generalized colorbar cloning**

Normalize each final Bezier tangent. Place each cone tip on the incoming side
of its target sphere with
`tip = target_center - tangent * target_radius`. Create one `Cone` trace using
`anchor="tip"`, `sizemode="absolute"`, a constant black colorscale, no scale,
and retained hover text. `_build_static_montage` already sets
`copied_trace.showscale = index == 0` only for the `Nodes` trace.

- [x] **Step 4: Verify GREEN and commit**

Run the complete Plotly test file and Ruff, then commit only the renderer and
test changes:

```bash
git commit -m "feat: add true 3D Plotly direction cones"
```

---

### Task 4: Documentation and deterministic visual preview

**Files:**

- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `tests/test_packaging.py`
- Modify: `scripts/check_acceptance_artifacts.py`
- Modify: `tests/test_acceptance_scripts.py`
- Modify: `docs/images/surface-plotly.png`

**Interfaces:**

- Both READMEs describe the same true-3D behavior and physical size units.
- The checker audits Plotly HTML for the ball-and-stick trace contract.
- The gallery displays the regenerated four-view image.

- [x] **Step 1: Add documentation and artifact-contract RED tests**

Require both README files to contain “ball-and-stick”, `Mesh3d`, and the phrase
describing millimetre node/tube diameters. Extend HTML checking for
`surface_interactive.html` to require the serialized trace names `Nodes` and
`Edges` plus `"render_mode":"ball-and-stick"`. Update the valid synthetic
fixture to include those exact JSON tokens.

- [x] **Step 2: Verify RED**

Run packaging and acceptance-script tests. Expected: README assertions fail and
the current generated HTML lacks `render_mode`.

- [x] **Step 3: Update both READMEs**

Describe that Plotly is the true-3D BrainNet-style backend: node spheres have
physical diameter mapped from node values, tubes have physical diameter mapped
from absolute connection weight, and lighting/occlusion respond to camera
rotation. Keep all fenced code blocks byte-identical across languages.

- [x] **Step 4: Extend the acceptance checker**

Make `_check_html` accept `require_ball_stick: bool = False`. Pass `True` only
for `surface_interactive.html`; when enabled, require lowercase tokens
`"name":"nodes"`, `"name":"edges"`, and
`"render_mode":"ball-and-stick"` after whitespace-insensitive normalization.

- [x] **Step 5: Generate and inspect the Plotly artifacts**

Run the deterministic acceptance generator in a fresh temporary output
directory, run the strict checker, and inspect the full-resolution
`surface_interactive.png`. Copy the verified Plotly-generated image to
`docs/images/surface-plotly.png`; do not copy a Matplotlib fallback. Confirm the
image is 1400 x 1000 and visibly shows shaded spheres and tubes in all four
views.

- [x] **Step 6: Verify docs and commit**

Run packaging and acceptance tests, Ruff, and `git diff --check`. Stage only
the two READMEs, checker/tests, and verified gallery image, then commit:

```bash
git commit -m "docs: show true 3D Plotly ball-and-stick output"
```

---

### Task 5: Repository and distribution verification

**Files:**

- Modify: `docs/superpowers/plans/2026-09-14-plotly-ball-stick.md`

- [x] **Step 1: Run code quality gates**

```bash
.venv/bin/python -m compileall -q src tests examples scripts
.venv/bin/python -m ruff check src tests examples scripts
git diff --check
```

- [x] **Step 2: Run the full CI-equivalent suite**

```bash
MNE_DONTWRITE_HOME=true MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/python -m pytest -q --cov=pyconnviz \
  --cov-report=term-missing --cov-fail-under=80
```

Expected: zero failures and coverage at least 80%.

- [x] **Step 3: Build and inspect clean distributions**

Build wheel and sdist into a fresh `/private/tmp` directory with
`python -m build`, run `twine check` on both exact files, verify the wheel
contains `_plotly_primitives.py`, and verify the sdist contains the new module,
its test, both READMEs, and the updated gallery image.

- [x] **Step 4: Record verification and commit the completed plan**

Record exact test count, coverage, artifact names, visual dimensions, and
unchanged user-owned files in this plan. Commit only the plan:

```bash
git commit -m "docs: record Plotly ball-and-stick verification"
```

- [x] **Step 5: Keep the local branch without pushing**

Report the local commits and final working-tree state. The only permitted
remaining modifications are the user's pre-existing `.gitignore` and
`LICENSE` changes.

## Verification Record

- 2026-09-14 code-quality gates: `compileall` completed, Ruff reported
  `All checks passed!`, and `git diff --check` completed without errors.
- Full CI-equivalent suite: 289 passed in 133.35 seconds; total coverage was
  90.87%, exceeding the required 80%.
- Deterministic acceptance output: strict checker inspected 14 files and 40
  consistently selected backend edges. The Plotly export used
  `plotly-kaleido` with no fallback; `surface_interactive.png` was 1400 x 1000
  pixels and was visually inspected at original resolution for shaded spheres
  and tubes in left, right, dorsal, and ventral views.
- Clean distributions: `pyconnviz-0.1.0-py3-none-any.whl` and
  `pyconnviz-0.1.0.tar.gz`; both passed `twine check`.
- Wheel inspection confirmed `pyconnviz/plotting/_plotly_primitives.py` and
  `pyconnviz/plotting/surface_plotly.py`. Sdist inspection confirmed the new
  module and test, both READMEs, and `docs/images/surface-plotly.png`.
- The user's pre-existing `.gitignore` and `LICENSE` modifications were not
  staged or changed by this implementation. No push was performed.
