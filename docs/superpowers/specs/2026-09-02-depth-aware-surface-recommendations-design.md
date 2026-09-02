# Depth-aware static surfaces and recommended backends design

## Goal

Make the Matplotlib and native Nilearn surface figures read as three-dimensional
anatomical context rather than a flat ball-and-stick layer, while making Plotly
surface HTML and the Nilearn glass brain the first recommended connectivity
views in the README and examples.

The existing public scientific contract remains unchanged: renderers consume
one immutable `PreparedConnectome`, and no renderer filters, thresholds, ranks,
or reconstructs connectivity.

## Scientific visual contract

The static surface figure must defend one limited claim: nodes and prepared
connections are anatomically located relative to a cortical mesh. It must not
imply that line depth, line opacity, or a neutral cortical texture is a measured
functional quantity.

- Edge color continues to encode signed or one-sided connection weight.
- Edge width continues to encode absolute connection magnitude.
- Node color and size continue to use the explicitly selected node values.
- Cortical luminance continues to show only neutral sulcal anatomy unless the
  caller supplies a real surface or statistical map.
- Camera-depth fading is a display-only occlusion cue. It never changes the
  edge set, color normalization, width normalization, or node values.

## Considered approaches

### 1. Lower cortical opacity only

This is cheap and makes hidden connections easier to see, but every edge still
has the same salience. The network therefore continues to look like a flat
foreground overlay.

### 2. Transparent anatomy plus projected-depth cueing — selected

Keep the complete hemisphere-scoped edge set in every static panel, reduce the
built-in cortical opacities, split each Bezier curve into short display
segments, and multiply only segment alpha by normalized camera depth. Near
segments remain prominent and far segments fade smoothly. Matplotlib's native
node depth shading is enabled by the same `depth_cue` switch.

This approach improves spatial reading without hiding data. It also works for
both custom Matplotlib montages and figures returned by Nilearn's public
`plot_img_on_surf` interface.

### 3. Physical tubes/spheres or strict mesh occlusion

True tubes and spheres greatly increase artist count and full-fsaverage render
time. Matplotlib's collection-level 3D sorting cannot provide reliable
per-segment z-buffer occlusion, while hiding back-facing nodes or edges makes a
camera-dependent subset look like scientific filtering. Plotly already offers
the more appropriate WebGL, rotatable representation, so this approach is not
added to the static engines.

## Rendering architecture

`pyconnviz.plotting._surface_common` will own two backend-neutral helpers:

1. `projected_depth_factors(points, reference_points, projection,
   minimum)` transforms finite 3D points with a Matplotlib 4x4 projection
   matrix and returns factors in `[minimum, 1]`, where nearer points have the
   larger factor.
2. `depth_cued_line_data(curves, colors, widths, reference_points,
   projection, minimum, alpha, enabled)` returns one batched collection's
   segments, RGBA colors, and widths. With cueing enabled, curves are split into
   adjacent segments and receive per-segment alpha. With cueing disabled, it
   preserves one uniformly opaque artist path per input curve.

Both static renderers gain `depth_cue: bool = True`. They retain
`axis.computed_zorder = False` so the prepared edge set stays reviewable above
the translucent cortex, but uniform foreground alpha is replaced by the depth
gradient. `depth_cue=False` is the explicit uniform-foreground compatibility
mode.

The visual styles gain `depth_cue_min_alpha` and more transparent cortex
defaults:

| Style | Cortex alpha | Minimum depth factor |
|---|---:|---:|
| `paper` | 0.24 | 0.20 |
| `soft` | 0.20 | 0.15 |
| `dark` | 0.32 | 0.28 |

An explicit `cortex_alpha` continues to override the style. The minimum depth
factor is intentionally not a new public option in v0.1; keeping it in the
style prevents an unnecessary parameter surface while retaining inspectability
through `get_style`.

## Recommended backend policy

The dispatcher defaults remain `backend="surface", engine="matplotlib"` for
backward compatibility and because Plotly is optional while glass rendering
requires MNI coordinates. Documentation recommendations change instead:

1. Use `engine="plotly"` for cortical anatomy, rotation, hover inspection, and
   a transparent mesh.
2. Use `backend="glass"` for the clearest static whole-connectome overview when
   valid MNI coordinates exist.
3. Use Matplotlib or native Nilearn surface montages as supplementary fixed-view
   anatomical context, not as evidence of physical edge depth.
4. Continue listing HTML `view_connectome`, circle, and every surface mode so
   the README remains a complete API guide.

The gallery, Quickstart, all-renderer table, and `examples/04_all_backends.py`
will put Plotly and glass first. The gallery image files remain the same stable
package assets; only their order and the two regenerated static surface images
change.

## Tests and visual acceptance

- Unit-test projection factors for bounds, monotonic near/far behavior,
  degenerate depth ranges, and invalid inputs.
- Unit-test that depth-cued lines have varying per-segment alpha while disabled
  cueing retains one path and uniform alpha per original curve.
- Test both static renderers for the new style opacity, depth-cued edge colors,
  node depth shading, and unchanged `panel_edges`.
- Test strict public routing of `depth_cue` and README recommendation order.
- Regenerate the deterministic acceptance set and the real MSDL-to-fsaverage
  gallery views, then inspect the PNGs at full resolution.
- Run focused tests, the complete coverage gate, Ruff, compileall, and package
  build checks before committing.

## Known limitation stated explicitly

Static Matplotlib/Nilearn output remains a camera projection, not a physical
ray-traced model. Depth alpha is an optical cue and must not be compared across
panels as a connectivity value. Plotly surface HTML is the primary anatomical
3D recommendation; the glass brain is the primary static connectivity
recommendation.

## Self-review

The design contains no placeholders. It preserves all scientific mappings and
edge identities, distinguishes display depth from data, does not require a new
dependency, keeps optional-backend compatibility, covers both static surface
implementations, and directly addresses the requested README recommendation
order.
