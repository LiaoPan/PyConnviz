"""Nilearn-quality static cortical surfaces with Matplotlib network overlays."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..geometry import node_surface_normals, to_nilearn_polymesh
from ..models import ConnectomeGeometry, PlotResult, PreparedConnectome, ViewSpec
from ..styles import get_style
from ..surface_overlay import build_surface_overlay, overlay_limits
from ._matplotlib_ball_stick import (
    cone_collection,
    sphere_collection,
    tube_collection,
)
from ._surface_common import (
    edge_color_norm,
    has_visible_surface_values,
    panel_key,
    quadratic_bezier,
    resolve_cortex_alpha,
    resolve_surface_hemi,
    resolve_views,
    scale_values,
    select_panel_edges,
    surface_background_values,
    surface_display_coordinates,
    transparent_surface_cmap,
)

OverlayMode = Literal["none", "roi", "gaussian", "vertex"]


def _finite_node_values(values: Any, node_count: int, *, name: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.shape != (node_count,):
        raise ValueError(f"{name} must have shape ({node_count},); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _node_norm(values: np.ndarray) -> Normalize:
    minimum = float(np.min(values)) if len(values) else 0.0
    maximum = float(np.max(values)) if len(values) else 1.0
    if np.isclose(minimum, maximum):
        if maximum >= 0:
            minimum, maximum = 0.0, max(maximum, 1.0)
        else:
            minimum, maximum = min(minimum, -1.0), 0.0
    return Normalize(vmin=minimum, vmax=maximum)


def _panel_hemispheres(view: ViewSpec) -> tuple[str, ...]:
    return ("left", "right") if view.hemi == "both" else (view.hemi,)


def _as_output_paths(output: str | Path | Sequence[str | Path] | None) -> tuple[Path, ...]:
    if output is None:
        return ()
    if isinstance(output, (str, Path)):
        return (Path(output),)
    paths = tuple(Path(path) for path in output)
    if not paths:
        raise ValueError("output sequence must not be empty")
    return paths


def _surface_image(surface: Any, mesh: Any, values: Mapping[str, Any]):
    return surface.SurfaceImage(
        mesh=mesh,
        data={hemi: np.asarray(data, dtype=np.float64) for hemi, data in values.items()},
    )


def plot_surface_matplotlib(
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    *,
    style: str = "paper",
    views: str | Sequence[ViewSpec] = "paper",
    surface_values: Mapping[str, Any] | None = None,
    node_overlay: OverlayMode | None = None,
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    surface_cmap: str | None = None,
    node_cmap: str | None = None,
    node_size_range: tuple[float, float] | None = None,
    edge_cmap: str | None = None,
    edge_width_range: tuple[float, float] | None = None,
    edge_alpha: float | None = None,
    edge_vmin: float | None = None,
    edge_vmax: float | None = None,
    cortex_alpha: float | None = None,
    depth_cue: bool = True,
    node_offset_mm: float = 1.5,
    overlay_sigma_mm: float = 12.0,
    overlay_radius_mm: float = 30.0,
    overlay_distance: Literal["geodesic", "euclidean"] = "geodesic",
    overlay_reduce: Literal["sum", "max"] = "sum",
    edge_arc_height: float = 0.16,
    edge_arc_min_mm: float = 4.0,
    edge_arc_max_mm: float = 35.0,
    edge_samples: int = 40,
    show_arrows: bool = False,
    title: str | None = None,
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    transparent: bool = False,
    rasterize_surface: bool = False,
    output: str | Path | Sequence[str | Path] | None = None,
    show: bool = False,
) -> PlotResult:
    """Render a static surface connectome without changing its edge set."""

    if len(geometry.node_names) != len(prepared.node_names):
        raise ValueError("geometry and prepared connectome must have the same node count")
    if tuple(geometry.node_names) != tuple(prepared.node_names):
        raise ValueError("geometry node_names must match prepared node_names in order")
    if not isinstance(depth_cue, (bool, np.bool_)):
        raise TypeError("depth_cue must be a bool")
    resolved_depth_cue = bool(depth_cue)
    visual = get_style(style)
    resolved_cortex_alpha = resolve_cortex_alpha(
        cortex_alpha, default=visual["fixed_cortex_alpha"]
    )
    panels = resolve_views(views)
    node_count = len(prepared.node_names)
    base_node_values = (
        prepared.node_strength
        if node_values is None
        else _finite_node_values(node_values, node_count, name="node_values")
    )
    color_values = (
        base_node_values
        if node_color_values is None
        else _finite_node_values(node_color_values, node_count, name="node_color_values")
    )
    size_values = (
        base_node_values
        if node_size_values is None
        else _finite_node_values(node_size_values, node_count, name="node_size_values")
    )
    selected_overlay = cast(
        OverlayMode,
        "vertex" if surface_values is not None else (
            visual["default_overlay"] if node_overlay is None else node_overlay
        ),
    )
    overlay = build_surface_overlay(
        selected_overlay,
        geometry,
        node_values=base_node_values,
        surface_values=surface_values,
        overlay_sigma_mm=overlay_sigma_mm,
        overlay_radius_mm=overlay_radius_mm,
        overlay_distance=overlay_distance,
        overlay_reduce=overlay_reduce,
    )

    plotting = importlib.import_module("nilearn.plotting")
    surface = importlib.import_module("nilearn.surface")
    pyplot = importlib.import_module("matplotlib.pyplot")
    if figsize is None:
        figsize = (4.8 * len(panels), 4.5)
    figure = pyplot.figure(figsize=figsize, dpi=dpi, facecolor=visual["background"])
    axes = [
        figure.add_subplot(1, len(panels), index + 1, projection="3d")
        for index in range(len(panels))
    ]
    for axis in axes:
        axis.computed_zorder = False
    node_cmap_name = visual["node_cmap"] if node_cmap is None else node_cmap
    node_diameters = scale_values(
        size_values,
        tuple(
            visual["fixed_node_diameter_range"]
            if node_size_range is None
            else node_size_range
        ),
    )
    node_normalization = _node_norm(color_values)
    node_colormap = colormaps[node_cmap_name]
    normals = node_surface_normals(geometry)
    nilearn_mesh = to_nilearn_polymesh(geometry)
    background = _surface_image(
        surface,
        nilearn_mesh,
        surface_background_values(
            geometry,
            cortex_color=visual["cortex_color"],
            sulc_contrast=visual["sulc_contrast"],
        ),
    )
    overlay_values = overlay.values
    has_overlay = has_visible_surface_values(overlay_values)
    if has_overlay:
        assert overlay_values is not None
        displayed_surface_values = overlay_values
        overlay_range = overlay_limits(overlay_values)
    else:
        displayed_surface_values = {
            hemi: np.zeros(len(mesh.coordinates), dtype=np.float64)
            for hemi, mesh in geometry.meshes.items()
        }
        overlay_range = (-1.0, 1.0)
    surface_map = _surface_image(surface, nilearn_mesh, displayed_surface_values)
    surface_cmap_name = visual["surface_cmap"] if surface_cmap is None else surface_cmap
    surface_colormap = surface_cmap_name if has_overlay else transparent_surface_cmap()
    all_weights = np.array([edge.weight for edge in prepared.edges], dtype=np.float64)
    edge_normalization, automatic_edge_cmap = edge_color_norm(
        all_weights, vmin=edge_vmin, vmax=edge_vmax
    )
    edge_cmap_name = automatic_edge_cmap if edge_cmap in {None, "auto"} else edge_cmap
    edge_colormap = colormaps[edge_cmap_name]
    widths = scale_values(
        np.abs(all_weights),
        tuple(
            visual["fixed_edge_width_range"]
            if edge_width_range is None
            else edge_width_range
        ),
    )
    edge_width_by_pair = {
        (edge.source, edge.target): float(width)
        for edge, width in zip(prepared.edges, widths, strict=True)
    }
    panel_edges: dict[str, tuple] = {}
    signed_overlay = overlay_range[0] < 0.0 < overlay_range[1]
    surface_threshold = (
        np.nextafter(0.0, 1.0) if has_overlay and not signed_overlay else None
    )

    for axis, panel in zip(axes, panels, strict=True):
        axis.set_facecolor(visual["background"])
        effective_hemi = resolve_surface_hemi(geometry, panel.hemi)
        display = None
        if effective_hemi is not None:
            display = surface_display_coordinates(
                geometry,
                panel.hemi,
                node_offset_mm=node_offset_mm,
                center=True,
            )
            before = len(axis.collections)
            plotting.plot_surf(
                surf_mesh=nilearn_mesh,
                surf_map=surface_map,
                bg_map=background,
                hemi=effective_hemi,
                view=panel.view,
                engine="matplotlib",
                cmap=surface_colormap,
                colorbar=False,
                alpha=resolved_cortex_alpha,
                bg_on_data=has_overlay,
                threshold=surface_threshold,
                vmin=overlay_range[0],
                vmax=overlay_range[1],
                axes=axis,
                figure=figure,
            )
            for collection in axis.collections[before:]:
                collection.set_zorder(0)
                if rasterize_surface:
                    collection.set_rasterized(True)
        visible_edges = select_panel_edges(prepared.edges, geometry, panel.hemi)
        panel_edges[panel_key(panel)] = visible_edges
        curves = []
        colors = []
        edge_diameters = []
        for edge in visible_edges if display is not None else ():
            assert display is not None
            cross_hemi = geometry.hemispheres[edge.source] != geometry.hemispheres[edge.target]
            curve = quadratic_bezier(
                display.node_positions[edge.source],
                display.node_positions[edge.target],
                normals[edge.source],
                normals[edge.target],
                cross_hemisphere=cross_hemi,
                brain_center=display.brain_center,
                arc_height=edge_arc_height,
                arc_min_mm=edge_arc_min_mm,
                arc_max_mm=edge_arc_max_mm,
                samples=edge_samples,
            )
            curves.append(curve)
            colors.append(edge_colormap(edge_normalization(edge.weight)))
            edge_diameters.append(edge_width_by_pair[(edge.source, edge.target)])
        if curves:
            base_edge_alpha = (
                visual["fixed_edge_alpha"] if edge_alpha is None else edge_alpha
            )
            collection = tube_collection(
                curves,
                np.asarray(edge_diameters, dtype=np.float64),
                colors,
                alpha=base_edge_alpha,
                depth_cue=resolved_depth_cue,
            )
            assert collection is not None
            axis.add_collection3d(collection)
            if show_arrows and prepared.directed:
                tips = []
                bases = []
                cone_diameters = []
                for edge, curve, diameter in zip(
                    visible_edges,
                    curves,
                    edge_diameters,
                    strict=True,
                ):
                    tangent = curve[-1] - curve[-2]
                    tangent /= np.linalg.norm(tangent)
                    tip = curve[-1] - tangent * (node_diameters[edge.target] / 2.0)
                    cone_diameter = max(2.0, diameter * 1.8)
                    length = max(3.0, cone_diameter * 1.8)
                    tips.append(tip)
                    bases.append(tip - tangent * length)
                    cone_diameters.append(cone_diameter)
                arrows = cone_collection(
                    np.asarray(bases),
                    np.asarray(tips),
                    np.asarray(cone_diameters),
                    colors,
                    alpha=base_edge_alpha,
                    depth_cue=resolved_depth_cue,
                )
                if arrows is not None:
                    axis.add_collection3d(arrows)
        node_indices = [
            index
            for index, hemi in enumerate(geometry.hemispheres)
            if panel.hemi == "both" or hemi == panel.hemi
        ]
        if display is not None and node_indices:
            points = display.node_positions[node_indices]
            node_colors = node_colormap(node_normalization(color_values[node_indices]))
            nodes = sphere_collection(
                points,
                node_diameters[node_indices],
                node_colors,
                depth_cue=resolved_depth_cue,
            )
            assert nodes is not None
            axis.add_collection3d(nodes)
        if panel.title:
            axis.set_title(panel.title, color="white" if style == "dark" else "black")
        axis.set_axis_off()
        if display is not None:
            extent = np.ptp(display.surface_coordinates, axis=0)
            axis.set_box_aspect(np.maximum(extent, 1.0))

    if title:
        figure.suptitle(title, color="white" if style == "dark" else "black")
    figure.subplots_adjust(
        left=0.01,
        right=0.84 if colorbar else 0.99,
        bottom=0.02,
        top=0.9,
        wspace=0.02,
    )
    if colorbar:
        if len(prepared.edges):
            edge_axis = figure.add_axes((0.86, 0.25, 0.012, 0.5))
            edge_mappable = ScalarMappable(norm=edge_normalization, cmap=edge_colormap)
            edge_bar = figure.colorbar(edge_mappable, cax=edge_axis)
            edge_bar.set_label("Edge weight")
            node_axis_left = 0.94
        else:
            node_axis_left = 0.92
        node_axis = figure.add_axes((node_axis_left, 0.25, 0.012, 0.5))
        node_mappable = ScalarMappable(norm=node_normalization, cmap=node_colormap)
        node_bar = figure.colorbar(node_mappable, cax=node_axis)
        node_bar.set_label("Node value")
    output_paths = _as_output_paths(output)
    for path in output_paths:
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise ValueError("Matplotlib surface output must use .png, .svg, or .pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=dpi,
            transparent=transparent,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Matplotlib did not create a non-empty output file: {path}")
    if show:
        pyplot.show()
    return PlotResult(
        backend="surface",
        engine="matplotlib",
        artist=figure,
        prepared=prepared,
        output_files=output_paths,
        panel_edges=panel_edges,
    )
