# Static Surface Sphere Smoothing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace visibly faceted static surface nodes with smooth, high-resolution true-3D spheres in both supplementary renderers.

**Architecture:** Keep the shared batched `BallStickCollection` architecture and increase only its sphere primitive resolution from 8-by-12 to 32-by-48. Extend the generated geometry manifest with a per-node sphere triangle count so the acceptance checker can reject low-poly regressions independently of the number of panels or nodes.

**Tech Stack:** Python 3.10+, NumPy, Matplotlib `Poly3DCollection`, Nilearn, pytest, Pillow, build, Twine

## Global Constraints

- Static node coordinates, physical diameters, colours, and depth cueing remain unchanged.
- Edge tubes, direction cones, cortex settings, views, and the Plotly renderer remain unchanged.
- Every static node sphere has exactly 2,976 triangular faces (32 latitude steps by 48 longitude steps).
- Matplotlib three-view and native Nilearn six-view outputs remain deterministic true-3D ball-and-stick figures.
- Work inline without subagents; local commits are allowed, but never push.

---

### Task 1: Smooth shared static sphere geometry

**Files:**
- Modify: `tests/test_matplotlib_ball_stick.py`
- Modify: `src/pyconnviz/plotting/_matplotlib_ball_stick.py`

**Interfaces:**
- Consumes: `sphere_mesh(center, radius, *, latitude_steps, longitude_steps) -> TriangleMesh`
- Produces: `sphere_collection(...) -> BallStickCollection | None` with 2,976 triangles per sphere

- [ ] **Step 1: Write the failing topology regression test**

Change the node collection assertions to:

```python
assert collection.mesh_vertex_count == 2 * (2 + 31 * 48)
assert collection.mesh_triangle_count == 2 * (2 * 48 * 31)
assert collection.mesh_triangle_count // len(collection.diameters) == 2_976
```

- [ ] **Step 2: Run the focused test and confirm the red state**

Run:

```bash
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest \
  tests/test_matplotlib_ball_stick.py::test_spheres_are_one_auditable_shaded_poly_collection -q
```

Expected: failure because the current collection contains 170 vertices and 336 triangles for two 8-by-12 spheres.

- [ ] **Step 3: Implement the minimal geometry change**

Set:

```python
_SPHERE_LATITUDE_STEPS = 32
_SPHERE_LONGITUDE_STEPS = 48
```

Do not change tube or cone side counts.

- [ ] **Step 4: Verify the shared adapter and both renderer integrations**

Run:

```bash
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest \
  tests/test_matplotlib_ball_stick.py \
  tests/test_surface_matplotlib.py \
  tests/test_surface_nilearn.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the geometry correction**

```bash
git add src/pyconnviz/plotting/_matplotlib_ball_stick.py \
  tests/test_matplotlib_ball_stick.py
git commit -m "fix: smooth static surface node spheres"
```

### Task 2: Enforce smooth spheres in acceptance auditing

**Files:**
- Modify: `scripts/generate_acceptance_artifacts.py`
- Modify: `scripts/check_acceptance_artifacts.py`
- Modify: `tests/test_acceptance_scripts.py`

**Interfaces:**
- Consumes: `BallStickCollection.mesh_triangle_count` and `.diameters`
- Produces: `static_surface_geometry.<renderer>.node_sphere_triangles: int`

- [ ] **Step 1: Add a valid fixture field and a low-resolution rejection test**

Add `"node_sphere_triangles": 2976` to each static renderer fixture record and add:

```python
def test_checker_rejects_low_resolution_static_node_spheres(tmp_path: Path) -> None:
    root = tmp_path / "low-poly-static"
    write_valid_fixture(root)
    path = root / "edge_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["static_surface_geometry"]["surface_nilearn"][
        "node_sphere_triangles"
    ] = 168
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AcceptanceError, match="smooth sphere"):
        check_artifacts(root, strict=False)
```

- [ ] **Step 2: Run the new test and confirm the red state**

Run:

```bash
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest \
  tests/test_acceptance_scripts.py::test_checker_rejects_low_resolution_static_node_spheres -q
```

Expected: failure because the checker does not yet validate `node_sphere_triangles`.

- [ ] **Step 3: Record and validate per-sphere resolution**

In `_static_geometry_audit`, calculate the total node part count from all node
collection diameters, ensure it is positive, and return:

```python
"node_sphere_triangles": node_mesh_triangles // node_part_count,
```

In `_check_manifest`, require an integer value of at least `2_976`; otherwise
raise an `AcceptanceError` whose message contains `smooth sphere`.

- [ ] **Step 4: Verify acceptance audit tests**

Run:

```bash
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest \
  tests/test_acceptance_scripts.py -q
```

Expected: all acceptance script tests pass.

- [ ] **Step 5: Commit the audit contract**

```bash
git add scripts/generate_acceptance_artifacts.py \
  scripts/check_acceptance_artifacts.py tests/test_acceptance_scripts.py
git commit -m "test: enforce smooth static surface spheres"
```

### Task 3: Regenerate and visually verify published figures

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Regenerate: `artifacts/acceptance/edge_manifest.json`
- Regenerate: affected files under `artifacts/acceptance/`
- Regenerate: `artifacts/upstream_validation/comparisons/surface_msdl_fsaverage/pyconnviz_surface_paper.png`
- Regenerate: `artifacts/upstream_validation/comparisons/surface_msdl_fsaverage/pyconnviz_surface_nilearn_native.png`
- Regenerate: `artifacts/upstream_validation/comparisons/surface_msdl_fsaverage/surface_comparison.png`
- Regenerate: `docs/images/surface-matplotlib.png`
- Regenerate: `docs/images/surface-nilearn.png`

**Interfaces:**
- Consumes: the shared prepared MSDL connectome and the full-resolution `data/fsaverage` pial surfaces
- Produces: updated three-view and six-view gallery PNGs with smooth true-3D spheres

- [ ] **Step 1: Update English and Chinese rendering descriptions**

Describe the supplementary nodes as `high-resolution smooth spheres` in
`README.md` and `高分辨率光滑球体` in `README.zh-CN.md`, without changing the
recommended backend order.

- [ ] **Step 2: Regenerate deterministic acceptance artifacts**

Run:

```bash
MNE_DONTWRITE_HOME=true MPLCONFIGDIR=/private/tmp/pyconnviz-mpl \
  .venv/bin/python scripts/generate_acceptance_artifacts.py \
  --outdir artifacts/acceptance
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python \
  scripts/check_acceptance_artifacts.py --artifacts artifacts/acceptance --strict
```

Expected: the strict checker passes and both static records contain
`"node_sphere_triangles": 2976`.

- [ ] **Step 3: Regenerate the cached real-MSDL surface case**

Run `run_surface_case` with the cached correlation matrix, labels, MNI centres,
and `data/fsaverage`, writing to `artifacts/upstream_validation`. Copy the new
three-view and six-view PNGs to `docs/images/surface-matplotlib.png` and
`docs/images/surface-nilearn.png`.

- [ ] **Step 4: Inspect both images at original resolution**

Confirm that large nodes have circular silhouettes and no coarse checkerboard
facets; confirm that tubes remain connected, the cortex remains translucent,
and all three/six views remain present.

- [ ] **Step 5: Run complete verification**

Run:

```bash
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m pytest -q \
  --cov=pyconnviz --cov-report=term-missing --cov-fail-under=80
MPLCONFIGDIR=/private/tmp/pyconnviz-mpl .venv/bin/python -m ruff check \
  src tests scripts
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Expected: all tests pass with at least 80% coverage, Ruff passes, both package
artifacts build, and Twine validates both distributions.

- [ ] **Step 6: Commit regenerated evidence and documentation**

```bash
git add README.md README.zh-CN.md artifacts docs/images
git commit -m "docs: publish smooth static surface spheres"
```

Do not push.
