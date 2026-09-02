"""Copy-safe visual style presets.

Styles intentionally contain no scientific filtering or dimensional-selection
parameters. Changing a style can never alter a :class:`PreparedConnectome`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

SCIENTIFIC_STYLE_KEYS = frozenset(
    {
        "complex_mode",
        "directed",
        "edge_mask",
        "edge_threshold",
        "epoch",
        "freq",
        "keep_sign",
        "max_edges",
        "min_distance_mm",
        "reduction",
        "strength_mode",
        "symmetrize",
        "time",
    }
)

_STYLES: dict[str, dict[str, Any]] = {
    "paper": {
        "background": "white",
        "cortex_color": (0.82, 0.82, 0.82),
        "cortex_alpha": 0.24,
        "depth_cue_min_alpha": 0.20,
        "sulc_contrast": 0.55,
        "surface_cmap": "RdBu_r",
        "node_cmap": "viridis",
        "node_size_range": [24.0, 100.0],
        "node_edgecolor": "white",
        "edge_cmap": "auto",
        "edge_width_range": [0.6, 4.0],
        "edge_alpha": 0.72,
        "font_size": 10.0,
        "default_views": "paper",
        "default_overlay": "none",
    },
    "soft": {
        "background": "white",
        "cortex_color": (0.86, 0.86, 0.86),
        "cortex_alpha": 0.20,
        "depth_cue_min_alpha": 0.15,
        "sulc_contrast": 0.45,
        "surface_cmap": "RdBu_r",
        "node_cmap": "magma",
        "node_size_range": [28.0, 108.0],
        "node_edgecolor": "white",
        "edge_cmap": "auto",
        "edge_width_range": [0.7, 3.6],
        "edge_alpha": 0.58,
        "font_size": 10.0,
        "default_views": "paper",
        "default_overlay": "none",
    },
    "dark": {
        "background": "#111318",
        "cortex_color": (0.72, 0.74, 0.78),
        "cortex_alpha": 0.32,
        "depth_cue_min_alpha": 0.28,
        "sulc_contrast": 0.65,
        "surface_cmap": "RdBu_r",
        "node_cmap": "plasma",
        "node_size_range": [26.0, 105.0],
        "node_edgecolor": "#eceff4",
        "edge_cmap": "auto",
        "edge_width_range": [0.7, 4.2],
        "edge_alpha": 0.82,
        "font_size": 10.0,
        "default_views": "paper",
        "default_overlay": "none",
    },
}


def get_style(name: str = "paper") -> dict[str, Any]:
    """Return an independent copy of a built-in visual style."""

    try:
        style = _STYLES[name]
    except KeyError as error:
        raise ValueError("Unknown style; choose one of: paper, soft, dark") from error
    if SCIENTIFIC_STYLE_KEYS.intersection(style):  # defensive invariant
        raise RuntimeError(f"Style {name!r} contains scientific parameters")
    return deepcopy(style)
