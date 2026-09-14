# Static Surface True-3D Ball-and-Stick Design

## Objective

Replace the screen-space network glyphs in the supplementary Matplotlib
three-view and native Nilearn six-view surface figures with physically sized,
lit three-dimensional spheres, tubes, and direction cones. Preserve all
scientific data preparation, edge selection, cortical rendering, layouts,
colorbars, and export formats.

## Root Cause

Both static renderers currently place network coordinates in a 3D axis but
draw nodes with `Axes3D.scatter` and edges with `Line3DCollection`. Marker area
and line width are display-space quantities, so they do not gain perspective,
surface normals, or true solid occlusion. Depth-dependent alpha improves
legibility but cannot turn those artists into 3D geometry.

## Approaches Considered

1. **Shared triangular meshes (selected).** Reuse PyConnviz's deterministic UV
   spheres and parallel-transport tubes, add a deterministic cone primitive,
   and render their planar triangles with `Poly3DCollection`. This provides
   physical dimensions, face shading, perspective, and the same geometry in
   both static engines without adding a dependency.
2. **Enhance the existing scatter and lines.** More aggressive depth shading,
   outlines, and alpha would be faster, but marker and line sizes would remain
   screen-space and would not satisfy the true-3D requirement.
3. **Post-process exported images.** Raster highlights could mimic volume from
   one camera only, would not work consistently for SVG/PDF or custom views,
   and would visually imply geometry that the renderer does not contain.

## Architecture

The Plotly-independent geometry in `_plotly_primitives.py` remains the single
source for `TriangleMesh`, `sphere_mesh`, `tube_mesh`, and `merge_meshes`. Its
module docstring will explicitly describe shared rendering use. A new
`cone_mesh` primitive will create a closed cone from base center, tip, radius,
and side count.

A new `_matplotlib_ball_stick.py` adapter will:

- turn triangular faces into planar `(face, vertex, xyz)` arrays accepted by
  `Poly3DCollection`;
- merge spheres, tubes, and optional cones while retaining one color per face;
- create separate shaded collections named through stable `gid` values:
  `pyconnviz-edges`, `pyconnviz-nodes`, and `pyconnviz-directions`;
- use a fixed `LightSource` so repeated renders are deterministic;
- return collection metadata containing physical diameters and mesh counts for
  tests and downstream audits.

`surface_matplotlib.py` and `surface_nilearn.py` will continue to compute node
positions, normals, Bezier curves, panel edge sets, color normalization, and
surface figures. They will pass those arrays to the shared adapter instead of
calling `scatter`, `Line3DCollection`, or `quiver`.

## Visual and Scientific Semantics

- `node_size_range` is a sphere-diameter range in surface coordinate units;
  both static functions default to `(6.0, 16.0)`, matching Plotly.
- The resolved `edge_width_range` is a tube-diameter range in the same units.
  Existing static style defaults remain the source when the argument is
  omitted.
- For FreeSurfer surface-RAS geometry these units are millimetres.
- Node color still maps the selected node values. Edge color still maps signed
  edge weights. Node diameter still maps the selected node-size values; tube
  diameter still maps absolute edge weight.
- The cortical mesh and its alpha are unchanged. Physical network meshes are
  drawn after the translucent cortex so internal connections remain legible.
- `depth_cue=True` enables face lighting; `False` keeps uniform RGB while
  retaining true 3D geometry. Alpha stays the explicit edge alpha rather than
  encoding a second, view-dependent data variable.
- Directed arrow cones use the final Bezier tangent. Their tips stop on the
  target sphere surface so they do not protrude through the node.

## Compatibility and Performance

No public backend name, prepared data, panel key, output type, or optional
dependency changes. PNG, SVG, and PDF continue through Matplotlib; native
Nilearn still calls `plot_img_on_surf` to create the cortical panels.

Each panel receives at most one node collection, one edge collection, and one
direction collection. Sphere resolution is 8 latitude by 12 longitude steps;
tubes and cones use 8 sides. This is intentionally lighter than the interactive
Plotly geometry because static figures repeat the network across three or six
panels.

## Testing and Acceptance

Tests will first fail against the existing artists by requiring:

- `Poly3DCollection` network artists with the stable `gid` values;
- no network `Path3DCollection` or `Line3DCollection`;
- deterministic vertex/triangle counts and requested physical diameters;
- shaded face colors when depth cueing is enabled and uniform source colors
  when disabled;
- unchanged prepared edges and panel-specific edge identities;
- real 3D direction cones for directed inputs;
- non-empty PNG/SVG output and unchanged colorbar/layout contracts.

The deterministic acceptance generator will then recreate the Matplotlib
three-view and native Nilearn six-view images. Both images must be inspected at
full resolution and visibly show spherical nodes and tubular connections in
every panel before replacing the README gallery assets.
