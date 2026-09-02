"""Small visual mappings shared by non-surface plotting backends."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from ..models import PreparedConnectome
from ..styles import get_style
from ._surface_common import edge_color_norm, scale_values


def node_visuals(
    prepared: PreparedConnectome,
    *,
    style: str,
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    size_range: tuple[float, float] = (25.0, 90.0),
) -> tuple[list[tuple[float, float, float, float]], np.ndarray]:
    """Map node values to copy-safe RGBA colors and visible sizes."""

    base_values = (
        np.asarray(prepared.node_strength, dtype=np.float64)
        if node_values is None
        else np.asarray(node_values, dtype=np.float64)
    )
    color_values = (
        base_values
        if node_color_values is None
        else np.asarray(node_color_values, dtype=np.float64)
    )
    size_values = (
        base_values
        if node_size_values is None
        else np.asarray(node_size_values, dtype=np.float64)
    )
    expected = (len(prepared.node_names),)
    for name, values in (
        ("node_values", base_values),
        ("node_color_values", color_values),
        ("node_size_values", size_values),
    ):
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain {len(prepared.node_names)} finite values")
    minimum = float(np.min(color_values)) if len(color_values) else 0.0
    maximum = float(np.max(color_values)) if len(color_values) else 1.0
    if np.isclose(minimum, maximum):
        minimum, maximum = (0.0, max(maximum, 1.0)) if maximum >= 0 else (
            min(minimum, -1.0),
            0.0,
        )
    cmap = colormaps[get_style(style)["node_cmap"]]
    normalization = Normalize(vmin=minimum, vmax=maximum)
    colors = [tuple(cmap(normalization(value))) for value in color_values]
    sizes = scale_values(size_values, size_range)
    return colors, sizes


def edge_visuals(
    prepared: PreparedConnectome, *, edge_cmap: str | None = None
) -> tuple[str, float | None, float | None, bool]:
    """Return consistent edge colormap semantics for wrapper backends."""

    weights = np.array([edge.weight for edge in prepared.edges], dtype=np.float64)
    normalization, automatic_cmap = edge_color_norm(weights)
    cmap = automatic_cmap if edge_cmap in {None, "auto"} else edge_cmap
    signed = bool(np.any(weights < 0) and np.any(weights > 0))
    return cmap, normalization.vmin, normalization.vmax, signed
