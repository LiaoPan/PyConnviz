# Static Surface Edge Legibility Design

## Problem and evidence

The real MSDL gallery figures contain the expected connections, but their
fixed-view lateral panels are not legible after documentation-scale
downsampling. At a 499-pixel display width, the current Matplotlib surface
uses edge widths of 0.32--1.65 pixels (0.89-pixel median) and segment alpha of
0.15--0.61. Antialiasing therefore removes many low-weight or far-depth
segments even though the artists and edge manifest are correct.

Lowering cortex alpha alone from 0.20 to 0.08 produced almost no improvement.
Disabling depth cueing made the edges legible, but restored the flat
foreground appearance that the previous surface redesign was intended to
avoid.

The MSDL top-40 set also contains 21 cross-hemisphere edges. A single-
hemisphere lateral panel can correctly show only its intra-hemisphere subset:
9 left edges and 10 right edges. The whole-brain dorsal, Plotly, and glass-
brain views remain the complete-network views.

## Considered approaches

1. **Make only the cortex more transparent.** Rejected as insufficient by the
   controlled render comparison; the edge artists are already drawn above the
   surface and remain subpixel and low-alpha.
2. **Disable depth cueing or draw every edge fully opaque.** Rejected because
   it improves visibility by recreating the visually flat ball-and-stick
   overlay.
3. **Give fixed-view renderers a bounded visibility budget.** Selected. Keep a
   meaningful near/far alpha difference, but prevent far segments and thin
   edges from falling below documentation-scale visibility.
4. **Add a contrasting halo or duplicate edge layer.** Deferred. It changes
   apparent thickness and risks making the network look pasted onto the
   cortex; it is unnecessary if one colored line layer is sufficient.

## Selected behavior

Built-in styles gain fixed-view-only defaults. Matplotlib and native Nilearn
use these values only when the caller does not pass an explicit override;
Plotly remains unchanged.

| Style | fixed cortex alpha | depth factor floor | fixed edge alpha | fixed edge width (pt) |
|---|---:|---:|---:|---:|
| `paper` | 0.08 | 0.70 | 0.95 | 1.4--5.4 |
| `soft` | 0.08 | 0.65 | 0.90 | 1.3--5.0 |
| `dark` | 0.18 | 0.70 | 0.96 | 1.4--5.6 |

For the paper style, the lowest possible colored-segment alpha becomes
`0.95 * 0.70 = 0.665`, while alpha still varies with projected depth. At the
observed README scale, the smallest Matplotlib edge becomes approximately
0.75 pixels instead of 0.32 pixels. Explicit `cortex_alpha`, `edge_alpha`, and
`edge_width_range` arguments continue to take priority.

The new style keys are `fixed_cortex_alpha`, `fixed_edge_alpha`, and
`fixed_edge_width_range`. Existing `cortex_alpha`, `edge_alpha`, and
`edge_width_range` remain the Plotly defaults, avoiding changes to the already
accepted interactive renderer.

## Scientific invariants

- No matrix, mask, threshold, ranking, weight, node, or `PreparedConnectome`
  edge changes.
- Edge RGB still represents sign/magnitude and colored-core width still
  represents absolute weight.
- Depth changes alpha only; it is explicitly a view cue, not another measured
  variable.
- Lateral panels continue to show only scientifically valid intra-hemisphere
  subsets. Cross-hemisphere edges are not invented or projected onto a single
  hemisphere.
- Pure-connectivity surfaces remain neutral grayscale and contain no inferred
  statistical surface map.

## Verification

Regression tests will assert the fixed-view style values, the minimum rendered
line width and alpha, explicit-option precedence, and unchanged Plotly
defaults. The real MSDL and deterministic acceptance figures will be
regenerated, inspected both at original resolution and near 499 pixels wide,
and checked against the existing identical-edge manifests. Full pytest
coverage, Ruff, compileall, package build, and Twine checks remain required.
