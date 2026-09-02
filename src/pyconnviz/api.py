"""Strict public preparation-and-rendering dispatcher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .connectivity import prepare_connectome
from .models import ConnectomeGeometry, PlotResult, PreparedConnectome, ViewSpec

_PREPARE_OPTIONS = {
    "complex_mode",
    "directed",
    "edge_mask",
    "edge_threshold",
    "epoch",
    "freq",
    "keep_sign",
    "max_edges",
    "min_distance_mm",
    "node_axes",
    "reduce_axes",
    "reduction",
    "strength_mode",
    "symmetrize",
    "symmetry_atol",
    "symmetry_rtol",
    "time",
}

_MATPLOTLIB_OPTIONS = {
    "colorbar",
    "cortex_alpha",
    "depth_cue",
    "dpi",
    "edge_alpha",
    "edge_arc_height",
    "edge_arc_max_mm",
    "edge_arc_min_mm",
    "edge_samples",
    "edge_vmax",
    "edge_vmin",
    "edge_width_range",
    "figsize",
    "node_cmap",
    "node_color_values",
    "node_offset_mm",
    "node_size_range",
    "node_size_values",
    "overlay_distance",
    "overlay_radius_mm",
    "overlay_reduce",
    "overlay_sigma_mm",
    "rasterize_surface",
    "show_arrows",
    "surface_cmap",
    "title",
    "transparent",
}
_PLOTLY_OPTIONS = {
    "cortex_alpha",
    "edge_arc_height",
    "edge_arc_max_mm",
    "edge_arc_min_mm",
    "edge_samples",
    "edge_vmax",
    "edge_vmin",
    "edge_width_range",
    "include_plotlyjs",
    "image_height",
    "image_scale",
    "image_width",
    "node_cmap",
    "node_color_values",
    "node_offset_mm",
    "node_size_range",
    "node_size_values",
    "overlay_distance",
    "overlay_radius_mm",
    "overlay_reduce",
    "overlay_sigma_mm",
    "show_arrows",
    "static_views",
    "surface_cmap",
    "view",
}
_NILEARN_SURFACE_OPTIONS = {
    "bg_on_data",
    "colorbar",
    "cortex_alpha",
    "depth_cue",
    "dpi",
    "edge_alpha",
    "edge_arc_height",
    "edge_arc_max_mm",
    "edge_arc_min_mm",
    "edge_cmap",
    "edge_samples",
    "edge_vmax",
    "edge_vmin",
    "edge_width_range",
    "figsize",
    "hemispheres",
    "inflate",
    "node_cmap",
    "node_offset_mm",
    "node_size_range",
    "rasterize_surface",
    "show_arrows",
    "stat_map",
    "surf_mesh",
    "surface_cmap",
    "surface_vmax",
    "surface_vmin",
    "symmetric_cbar",
    "symmetric_cmap",
    "threshold",
    "title",
    "transparent",
}
_GLASS_OPTIONS = {"colorbar", "display_mode", "title"}
_HTML_OPTIONS = {"colorbar", "title"}
_CIRCLE_OPTIONS = {"colorbar", "node_order", "title"}


def _load_surface_matplotlib():
    from .plotting.surface_matplotlib import plot_surface_matplotlib

    return plot_surface_matplotlib


def _load_surface_plotly():
    from .plotting.surface_plotly import plot_surface_plotly

    return plot_surface_plotly


def _load_surface_nilearn():
    from .plotting.surface_nilearn import plot_surface_nilearn

    return plot_surface_nilearn


def _load_glass():
    from .plotting.nilearn_connectome import plot_glass_connectome

    return plot_glass_connectome


def _load_html():
    from .plotting.nilearn_connectome import plot_html_connectome

    return plot_html_connectome


def _load_circle():
    from .plotting.circle import plot_circle_connectome

    return plot_circle_connectome


def _take_options(options: dict[str, Any], allowed: set[str], *, target: str) -> dict[str, Any]:
    unknown = sorted(set(options) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise TypeError(f"Unsupported {target} option(s): {joined}")
    return options


def _finite_node_values(values: Any, node_count: int, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (node_count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {node_count} finite values")
    return np.array(array, copy=True)


def _category_codes(values: Sequence[str]) -> np.ndarray:
    codes: dict[str, float] = {}
    return np.asarray(
        [codes.setdefault(value, float(len(codes))) for value in values],
        dtype=np.float64,
    )


def _resolve_node_colors(
    mode: str,
    *,
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    node_values: Any,
) -> np.ndarray:
    if mode in {"strength", "custom"}:
        if mode == "custom" and node_values is None:
            raise ValueError("node_color_by='custom' requires node_values")
        values = prepared.node_strength if node_values is None else node_values
        return _finite_node_values(values, len(prepared.node_names), name="node_values")
    if mode == "hemisphere":
        return _category_codes(geometry.hemispheres)
    if mode == "group":
        if geometry.groups is None:
            raise ValueError("node_color_by='group' requires geometry.groups")
        return _category_codes(geometry.groups)
    raise ValueError(
        "node_color_by must be 'strength', 'hemisphere', 'group', or 'custom'"
    )


def _resolve_node_sizes(
    mode: str,
    *,
    prepared: PreparedConnectome,
    node_values: Any,
) -> np.ndarray:
    if mode not in {"strength", "custom"}:
        raise ValueError("node_size_by must be 'strength' or 'custom'")
    if mode == "custom" and node_values is None:
        raise ValueError("node_size_by='custom' requires node_values")
    values = prepared.node_strength if node_values is None else node_values
    return _finite_node_values(values, len(prepared.node_names), name="node_values")


def plot_connectome(
    connectivity_or_prepared: Any,
    geometry: ConnectomeGeometry,
    *,
    backend: str = "surface",
    engine: str = "matplotlib",
    style: str = "paper",
    views: str | Sequence[ViewSpec] = "paper",
    surface_values: Mapping[str, Any] | None = None,
    node_overlay: str | None = None,
    node_values: Any = None,
    node_color_by: str = "strength",
    node_size_by: str = "strength",
    edge_cmap: str = "auto",
    output: str | Path | Sequence[str | Path] | None = None,
    show: bool = False,
    **kwargs: Any,
) -> PlotResult:
    """Prepare raw connectivity once and dispatch it to one strict backend.

    Scientific preparation keyword arguments are accepted only for raw input.
    Passing them with :class:`PreparedConnectome` raises rather than silently
    preparing or filtering a second time.
    """

    if not isinstance(geometry, ConnectomeGeometry):
        raise TypeError("geometry must be a ConnectomeGeometry")
    scientific = {key: kwargs.pop(key) for key in list(kwargs) if key in _PREPARE_OPTIONS}
    if isinstance(connectivity_or_prepared, PreparedConnectome):
        if scientific:
            joined = ", ".join(sorted(scientific))
            raise ValueError(
                f"Connectivity is already prepared; scientific options cannot be repeated: {joined}"
            )
        prepared = connectivity_or_prepared
    else:
        prepared = prepare_connectome(
            connectivity_or_prepared, geometry=geometry, **scientific
        )
    color_values = _resolve_node_colors(
        node_color_by,
        prepared=prepared,
        geometry=geometry,
        node_values=node_values,
    )
    size_values = _resolve_node_sizes(
        node_size_by,
        prepared=prepared,
        node_values=node_values,
    )
    if "node_color_values" in kwargs:
        color_values = _finite_node_values(
            kwargs.pop("node_color_values"),
            len(prepared.node_names),
            name="node_color_values",
        )
    if "node_size_values" in kwargs:
        size_values = _finite_node_values(
            kwargs.pop("node_size_values"),
            len(prepared.node_names),
            name="node_size_values",
        )

    common = {
        "style": style,
        "node_values": node_values,
        "node_color_values": color_values,
        "node_size_values": size_values,
        "output": output,
    }
    if backend == "surface":
        common.update(
            {
                "surface_values": surface_values,
                "node_overlay": node_overlay,
                "edge_cmap": edge_cmap,
                "show": show,
            }
        )
        if engine == "matplotlib":
            renderer = _load_surface_matplotlib()
            options = _take_options(kwargs, _MATPLOTLIB_OPTIONS, target="surface/matplotlib")
            options.update(common)
            options["views"] = views
        elif engine == "nilearn":
            renderer = _load_surface_nilearn()
            options = _take_options(
                kwargs,
                _NILEARN_SURFACE_OPTIONS,
                target="surface/nilearn",
            )
            options.update(common)
            options["views"] = views
        elif engine == "plotly":
            renderer = _load_surface_plotly()
            options = _take_options(kwargs, _PLOTLY_OPTIONS, target="surface/plotly")
            options.update(common)
            if views != "paper":
                raise TypeError(
                    "views is a Matplotlib montage option; use Plotly's singular view option"
                )
        else:
            raise ValueError(
                "surface engine must be 'matplotlib', 'nilearn', or 'plotly'"
            )
    elif backend == "glass":
        if engine != "matplotlib":
            raise ValueError("glass backend engine must be 'matplotlib'")
        if surface_values is not None or node_overlay is not None or views != "paper":
            raise TypeError("surface_values, node_overlay, and views are surface-only options")
        renderer = _load_glass()
        options = _take_options(kwargs, _GLASS_OPTIONS, target="glass")
        options.update(common)
        options["edge_cmap"] = edge_cmap
        options["show"] = show
    elif backend == "html":
        if engine != "matplotlib":
            raise ValueError("html backend does not accept an alternate engine")
        if surface_values is not None or node_overlay is not None or views != "paper":
            raise TypeError("surface_values, node_overlay, and views are surface-only options")
        if show:
            raise TypeError("html backend does not accept show=True")
        renderer = _load_html()
        options = _take_options(kwargs, _HTML_OPTIONS, target="html")
        options.update(common)
        options["edge_cmap"] = edge_cmap
    elif backend == "circle":
        if engine != "matplotlib":
            raise ValueError("circle backend engine must be 'matplotlib'")
        if surface_values is not None or node_overlay is not None or views != "paper":
            raise TypeError("surface_values, node_overlay, and views are surface-only options")
        renderer = _load_circle()
        options = _take_options(kwargs, _CIRCLE_OPTIONS, target="circle")
        options.update(common)
        options["edge_cmap"] = edge_cmap
        options["show"] = show
    else:
        raise ValueError("backend must be 'surface', 'glass', 'html', or 'circle'")
    return renderer(prepared, geometry, **options)
