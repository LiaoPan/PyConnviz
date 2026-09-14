# Plotly True-3D Ball-and-Stick Surface Design

## Goal

Replace the Plotly surface backend's screen-space circles and lines with
genuine three-dimensional ball-and-stick geometry. Nodes must read as lit
spheres and connections as lit tubes from every camera angle, while preserving
PyConnviz's scientific value mappings, hover information, deterministic output,
and static four-view export.

## Root Cause

The current renderer uses `Scatter3d(mode="markers")` for nodes and
`Scatter3d(mode="lines")` for edges. Plotly defines both marker size and line
width in pixels. Their centers follow three-dimensional coordinates, but the
glyphs themselves remain screen-space primitives without physical depth,
surface lighting, or volume-dependent occlusion. Styling those traces cannot
produce a BrainNet Viewer-like ball-and-stick rendering.

## Chosen Architecture

Use explicit triangular geometry rendered with Plotly `Mesh3d`:

1. Generate one UV sphere mesh per node, scaled around the existing surface
   display coordinate.
2. Merge all node spheres into one disconnected `Mesh3d` trace named `Nodes`.
3. Generate one smooth tube along each existing quadratic Bezier edge path.
4. Merge all tubes into one disconnected `Mesh3d` trace named `Edges`.
5. Render tubes before spheres so spheres visually terminate the tubes at node
   centers.
6. For directed plots with `show_arrows=True`, use Plotly `Cone` geometry
   instead of two-dimensional diamond markers.

This gives the desired physical depth while keeping trace count bounded. The
alternative of one trace per node and edge would preserve simple per-trace
hover labels, but it multiplies trace overhead in interactive HTML and again in
the four-view static montage. A merged mesh retains hover data by repeating the
owning node or edge text for each mesh vertex.

## Geometry Layer

Create `src/pyconnviz/plotting/_plotly_primitives.py` as a Plotly-independent
numerical module. It will expose a small immutable `TriangleMesh` container and
the following helpers:

- `sphere_mesh(center, radius, *, latitude_steps, longitude_steps)` builds a
  closed UV sphere with unique poles and non-degenerate triangles.
- `tube_mesh(points, radius, *, sides)` builds rings along a polyline and joins
  adjacent rings with consistently wound triangles.
- `merge_meshes(meshes)` concatenates vertices and offsets face indices.

Tube frames use parallel transport. The first cross-section chooses the
coordinate axis least aligned with the initial tangent. Each later normal is
projected into the new tangent plane, normalized, and paired with its binormal.
This prevents sudden ring flips and twisted faces along curved connections.

All helpers validate finite arrays, positive radii, minimum resolutions, and
non-zero curve segments. Invalid internal geometry raises `ValueError` with a
specific message rather than emitting corrupt Plotly arrays.

## Scientific and Visual Mappings

The existing data semantics remain unchanged:

- Node color is mapped from `node_color_values` through `node_cmap`.
- Node size is mapped from `node_size_values` through `node_size_range`.
- Edge color is mapped from signed edge weight through `edge_cmap` and the
  existing normalization.
- Edge width is mapped from absolute edge weight through `edge_width_range`.
- Edge selection, thresholding, direction, and `PreparedConnectome` identity
  are untouched.

For the Plotly backend, size ranges now describe physical diameters in surface
coordinate units, which are millimetres for supported FreeSurfer geometry:

- `node_size_range=(6.0, 16.0)` becomes sphere diameter 6--16 mm.
- the style's Plotly edge width range becomes tube diameter, so tube radius is
  half the scaled width.

This preserves the public parameter names and numerical defaults while making
them camera-independent. The README will state the unit change explicitly.

## Plotly Trace Contract

The interactive figure contains:

1. Nilearn's cortical `Mesh3d` trace or traces, with the existing resolved
   cortex opacity.
2. An `Edges` `Mesh3d` trace when at least one edge exists. Vertex colors repeat
   the resolved solid color of each owning edge. Hover text repeats the edge
   direction, signed weight, and optional distance for every tube vertex.
3. A `Nodes` `Mesh3d` trace. Vertex intensity repeats the owning node's color
   value, with shared `cmin`, `cmax`, colorscale, and the existing “Node value”
   colorbar. Hover text repeats node name, hemisphere, group, and strength.
4. An optional `Directions` `Cone` trace for directed graphs.

Sphere and tube traces use smooth shading and explicit ambient, diffuse,
specular, roughness, Fresnel, and light-position settings. The cortex remains
visually subordinate through its existing transparency; network geometry is
opaque enough to preserve depth cues.

Figure metadata adds `render_mode="ball-and-stick"` plus vertex and triangle
counts. These values support automated audit without relying on subjective
pixel inspection.

## Static Montage

The existing static montage deep-copies every trace into each requested scene.
Its node-colorbar handling will be generalized from
`trace.marker.showscale` to `trace.showscale`, because the `Nodes` trace is now
`Mesh3d`. Only the first panel displays the node colorbar. Scene cameras,
titles, output dimensions, and exported file types remain unchanged.

## Performance Budget

Use fixed internal quality levels chosen for smooth appearance without
excessive HTML size:

- sphere latitude steps: 10;
- sphere longitude steps: 16;
- tube sides: 8;
- edge path samples: the existing public `edge_samples` value, default 40.

For the existing recommendation of at most 200 visible edges, this is bounded
to roughly 64,000 tube vertices before nodes. `warn_if_many_edges` will retain
the 200-edge threshold but describe geometry/interaction cost rather than
incorrectly claiming one trace per edge.

## Compatibility

- No new mandatory dependency is introduced.
- Plotly remains optional and lazy-imported.
- The implementation uses long-supported `Mesh3d` attributes: explicit
  triangle indices, `intensity`, `vertexcolor`, `lighting`, and
  `lightposition`.
- Existing HTML, PNG, SVG, and PDF output paths continue to work.
- Existing APIs and backends other than Plotly remain unchanged.
- `show=False` continues to prohibit calls to `Figure.show()`.

## Testing and Acceptance

Numerical tests will verify:

- sphere vertices lie at the requested radius and faces are valid;
- tube vertices lie at the requested radius from a straight centerline;
- merged face indices are correctly offset;
- invalid resolutions, radii, and degenerate paths fail explicitly.

Renderer tests will verify:

- `Nodes` and `Edges` are `Mesh3d`, not `Scatter3d` markers or lines;
- node intensity and sphere diameter still follow supplied node values;
- tube diameter still follows absolute edge weight;
- hover text contains every node and edge identity;
- metadata records `ball-and-stick` and correct geometry counts;
- only the first static panel shows the node colorbar;
- directed arrows use a `Cone` trace;
- offline HTML and static export behavior remain intact.

Finally, regenerate `docs/images/surface-plotly.png` from the deterministic
acceptance data, inspect the full-resolution result, run the strict acceptance
checker, the complete test suite with its coverage gate, Ruff, compileall, and
a fresh PEP 517 wheel/sdist plus Twine validation.

## Repository Safety

The user's existing `.gitignore` and `LICENSE` edits are out of scope and must
remain untouched and unstaged. Work may be committed locally, but no push or
other remote mutation is permitted.
