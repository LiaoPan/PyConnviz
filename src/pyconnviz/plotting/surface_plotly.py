"""Interactive Nilearn + Plotly cortical surface connectome backend."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_hex
from numpy.typing import NDArray

from .._compat import get_public_plotly_figure
from ..geometry import node_surface_normals, to_nilearn_polymesh
from ..models import ConnectomeGeometry, OptionalDependencyError, PlotResult, PreparedConnectome
from ..styles import get_style
from ..surface_overlay import build_surface_overlay, overlay_limits
from ._plotly_primitives import TriangleMesh, merge_meshes, sphere_mesh, tube_mesh
from ._surface_common import (
    edge_color_norm,
    has_visible_surface_values,
    quadratic_bezier,
    resolve_cortex_alpha,
    scale_values,
    surface_background_values,
    surface_display_coordinates,
    transparent_surface_cmap,
)

OverlayMode = Literal["none", "roi", "gaussian", "vertex"]
StaticView = str | tuple[float, float]

DEFAULT_STATIC_VIEWS: tuple[StaticView, ...] = (
    "left",
    "right",
    "dorsal",
    "ventral",
)

_SPHERE_LATITUDE_STEPS = 10
_SPHERE_LONGITUDE_STEPS = 16
_TUBE_SIDES = 8
_NODE_LIGHTING = {
    "ambient": 0.34,
    "diffuse": 0.88,
    "specular": 0.62,
    "roughness": 0.34,
    "fresnel": 0.08,
}
_EDGE_LIGHTING = {
    "ambient": 0.38,
    "diffuse": 0.86,
    "specular": 0.42,
    "roughness": 0.38,
    "fresnel": 0.06,
}
_NETWORK_LIGHT_POSITION = {"x": 1200, "y": -1800, "z": 2200}


def _require_plotly() -> None:
    if importlib.util.find_spec("plotly") is None:
        raise OptionalDependencyError(
            "Plotly surface rendering requires the interactive extra: "
            "python -m pip install 'pyconnviz[interactive]'"
        )


def warn_if_many_edges(edge_count: int, *, recommended_max: int = 200) -> int:
    """Warn about HTML/interaction cost without changing the requested data."""

    if edge_count > recommended_max:
        warnings.warn(
            f"Plotly will generate tube geometry for all {edge_count} edges; "
            f"for responsive HTML, prepare at most {recommended_max} visible edges",
            RuntimeWarning,
            stacklevel=2,
        )
    return edge_count


def _mesh_coordinates(mesh: TriangleMesh) -> dict[str, NDArray[Any]]:
    """Return explicit Plotly Mesh3d coordinate and face arrays."""

    return {
        "x": mesh.vertices[:, 0],
        "y": mesh.vertices[:, 1],
        "z": mesh.vertices[:, 2],
        "i": mesh.faces[:, 0],
        "j": mesh.faces[:, 1],
        "k": mesh.faces[:, 2],
    }


def _color_limits(values: NDArray[np.float64]) -> tuple[float, float]:
    """Return finite non-zero color limits even when every value is equal."""

    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    return lower, upper


def _node_mesh_data(
    positions: NDArray[np.float64],
    diameters: NDArray[np.float64],
    color_values: NDArray[np.float64],
    hover: Sequence[str],
) -> tuple[TriangleMesh, NDArray[np.float64], list[str], list[int]]:
    parts = [
        sphere_mesh(
            center,
            diameter / 2.0,
            latitude_steps=_SPHERE_LATITUDE_STEPS,
            longitude_steps=_SPHERE_LONGITUDE_STEPS,
        )
        for center, diameter in zip(positions, diameters, strict=True)
    ]
    counts = [len(part.vertices) for part in parts]
    intensities = np.concatenate(
        [
            np.full(count, value, dtype=np.float64)
            for count, value in zip(counts, color_values, strict=True)
        ]
    )
    hovertext = [
        text
        for text, count in zip(hover, counts, strict=True)
        for _ in range(count)
    ]
    return merge_meshes(parts), intensities, hovertext, counts


def _output_paths(output: str | Path | Sequence[str | Path] | None) -> tuple[Path, ...]:
    if output is None:
        return ()
    if isinstance(output, (str, Path)):
        return (Path(output),)
    result = tuple(Path(path) for path in output)
    if not result:
        raise ValueError("output sequence must not be empty")
    return result


def _check_static_export_dependencies(paths: tuple[Path, ...]) -> None:
    if any(path.suffix.lower() != ".html" for path in paths) and importlib.util.find_spec(
        "kaleido"
    ) is None:
        raise OptionalDependencyError(
            "Plotly static export requires Kaleido: "
            "python -m pip install 'pyconnviz[export]'"
        )


def _resolve_static_views(
    views: Sequence[StaticView] | None,
) -> tuple[StaticView, ...] | None:
    if views is None:
        return None
    if isinstance(views, (str, bytes)):
        raise TypeError("static_views must be a sequence of views, not a string")
    resolved = tuple(views)
    if not resolved:
        raise ValueError("static_views must not be empty")
    return resolved


def _static_view_title(view: StaticView) -> str:
    if isinstance(view, str):
        return view.replace("_", " ").title()
    elevation, azimuth = view
    return f"Elevation {elevation:g}\N{DEGREE SIGN}, azimuth {azimuth:g}\N{DEGREE SIGN}"


def _surface_image(surface: Any, mesh: Any, values: Mapping[str, Any]):
    return surface.SurfaceImage(
        mesh=mesh,
        data={hemi: np.asarray(data, dtype=np.float64) for hemi, data in values.items()},
    )


def _set_surface_opacity(figure: Any, alpha: float) -> None:
    """Set opacity on Nilearn's public Plotly mesh trace(s)."""

    mesh_traces = [
        trace for trace in figure.data if getattr(trace, "type", None) == "mesh3d"
    ]
    if not mesh_traces:
        raise RuntimeError("Nilearn Plotly surface did not expose a Mesh3d trace")
    for trace in mesh_traces:
        trace.update(opacity=alpha)


def _build_base_surface(
    geometry: ConnectomeGeometry,
    overlay_values: Mapping[str, NDArray[np.float64]] | None,
    *,
    surface_cmap: str,
    cortex_color: str | Sequence[float],
    sulc_contrast: float,
    hemi: str,
    view: str | tuple[float, float],
):
    plotting = importlib.import_module("nilearn.plotting")
    surface = importlib.import_module("nilearn.surface")
    mesh = to_nilearn_polymesh(geometry)
    background = _surface_image(
        surface,
        mesh,
        surface_background_values(
            geometry,
            cortex_color=cortex_color,
            sulc_contrast=sulc_contrast,
        ),
    )
    has_overlay = has_visible_surface_values(overlay_values)
    if has_overlay:
        assert overlay_values is not None
        displayed_surface_values = overlay_values
        limits = overlay_limits(overlay_values)
    else:
        displayed_surface_values = {
            name: np.zeros(len(part.coordinates), dtype=np.float64)
            for name, part in geometry.meshes.items()
        }
        limits = (-1.0, 1.0)
    surface_map = _surface_image(surface, mesh, displayed_surface_values)
    surface_colormap = surface_cmap if has_overlay else transparent_surface_cmap()
    signed_overlay = limits[0] < 0.0 < limits[1]
    return plotting.plot_surf(
        surf_mesh=mesh,
        surf_map=surface_map,
        bg_map=background,
        hemi=hemi,
        view=view,
        engine="plotly",
        cmap=surface_colormap,
        symmetric_cmap=limits[0] is not None and limits[0] < 0 < limits[1],
        colorbar=False,
        bg_on_data=has_overlay,
        threshold=(
            np.nextafter(0.0, 1.0) if has_overlay and not signed_overlay else None
        ),
        vmin=limits[0],
        vmax=limits[1],
    )


def _nilearn_scene_layout(
    geometry: ConnectomeGeometry,
    overlay_values: Mapping[str, NDArray[np.float64]] | None,
    *,
    surface_cmap: str,
    cortex_color: str | Sequence[float],
    sulc_contrast: float,
    hemi: str,
    view: StaticView,
) -> dict[str, Any]:
    wrapper = _build_base_surface(
        geometry,
        overlay_values,
        surface_cmap=surface_cmap,
        cortex_color=cortex_color,
        sulc_contrast=sulc_contrast,
        hemi=hemi,
        view=view,
    )
    figure = get_public_plotly_figure(wrapper)
    return dict(figure.layout.scene.to_plotly_json())


def _build_static_montage(
    figure: Any,
    views: tuple[StaticView, ...],
    scene_layouts: Sequence[Mapping[str, Any]],
) -> Any:
    if len(scene_layouts) != len(views):
        raise ValueError("scene_layouts must contain one layout per static view")
    columns = min(2, len(views))
    rows = math.ceil(len(views) / columns)
    specs = [
        [
            {"type": "scene"}
            if row * columns + column < len(views)
            else None
            for column in range(columns)
        ]
        for row in range(rows)
    ]
    titles = [
        _static_view_title(views[index]) if index < len(views) else ""
        for index in range(rows * columns)
    ]
    make_subplots = importlib.import_module("plotly.subplots").make_subplots
    montage = make_subplots(
        rows=rows,
        cols=columns,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )
    for index, scene_layout in enumerate(scene_layouts):
        row, column = divmod(index, columns)
        for trace in figure.data:
            copied_trace = copy.deepcopy(trace)
            if getattr(copied_trace, "name", None) == "Nodes":
                copied_trace.showscale = index == 0
            montage.add_trace(copied_trace, row=row + 1, col=column + 1)
        scene_name = "scene" if index == 0 else f"scene{index + 1}"
        resolved_scene_layout = dict(scene_layout)
        resolved_scene_layout.pop("domain", None)
        montage.layout[scene_name].update(resolved_scene_layout)

    source_layout = figure.layout.to_plotly_json()
    metadata = dict(source_layout.get("meta") or {})
    metadata["static_views"] = [
        view if isinstance(view, str) else list(view) for view in views
    ]
    montage.update_layout(
        paper_bgcolor=source_layout.get("paper_bgcolor", "white"),
        plot_bgcolor=source_layout.get("plot_bgcolor", "white"),
        font=source_layout.get("font", {}),
        margin={"l": 0, "r": 80, "b": 0, "t": 55, "pad": 0},
        meta=metadata,
        showlegend=False,
    )
    return montage


def plot_surface_plotly(
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    *,
    style: str = "soft",
    view: str | tuple[float, float] = "dorsal",
    surface_values: Mapping[str, Any] | None = None,
    node_overlay: OverlayMode | None = None,
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    surface_cmap: str | None = None,
    node_cmap: str | None = None,
    node_size_range: tuple[float, float] = (6.0, 16.0),
    edge_cmap: str | None = None,
    edge_width_range: tuple[float, float] | None = None,
    edge_vmin: float | None = None,
    edge_vmax: float | None = None,
    cortex_alpha: float | None = None,
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
    include_plotlyjs: bool | str = True,
    static_views: Sequence[StaticView] | None = DEFAULT_STATIC_VIEWS,
    image_width: int | None = None,
    image_height: int | None = None,
    image_scale: float | None = None,
    output: str | Path | Sequence[str | Path] | None = None,
    show: bool = False,
) -> PlotResult:
    """Render an interactive surface and optional static multi-view montage."""

    _require_plotly()
    paths = _output_paths(output)
    _check_static_export_dependencies(paths)
    resolved_static_views = _resolve_static_views(static_views)
    if tuple(geometry.node_names) != tuple(prepared.node_names):
        raise ValueError("geometry node_names must match prepared node_names in order")
    visual = get_style(style)
    resolved_cortex_alpha = resolve_cortex_alpha(
        cortex_alpha, default=visual["cortex_alpha"]
    )
    node_count = len(prepared.node_names)

    def node_array(values: Any, fallback: np.ndarray, name: str) -> np.ndarray:
        if values is None:
            return np.asarray(fallback, dtype=np.float64)
        array = np.array(values, dtype=np.float64, copy=True)
        if array.shape != (node_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain {node_count} finite node values")
        return array

    base_values = node_array(node_values, prepared.node_strength, "node_values")
    color_values = node_array(node_color_values, base_values, "node_color_values")
    size_values = node_array(node_size_values, base_values, "node_size_values")
    selected_overlay = cast(
        OverlayMode,
        "vertex" if surface_values is not None else (
            visual["default_overlay"] if node_overlay is None else node_overlay
        ),
    )
    overlay = build_surface_overlay(
        selected_overlay,
        geometry,
        node_values=base_values,
        surface_values=surface_values,
        overlay_sigma_mm=overlay_sigma_mm,
        overlay_radius_mm=overlay_radius_mm,
        overlay_distance=overlay_distance,
        overlay_reduce=overlay_reduce,
    )
    display = surface_display_coordinates(
        geometry,
        "both",
        node_offset_mm=node_offset_mm,
        center=False,
    )
    resolved_surface_cmap = (
        visual["surface_cmap"] if surface_cmap is None else surface_cmap
    )
    wrapper = _build_base_surface(
        geometry,
        overlay.values,
        surface_cmap=resolved_surface_cmap,
        cortex_color=visual["cortex_color"],
        sulc_contrast=visual["sulc_contrast"],
        hemi=display.hemi,
        view=view,
    )
    figure = get_public_plotly_figure(wrapper)
    _set_surface_opacity(figure, resolved_cortex_alpha)
    graph_objects = importlib.import_module("plotly.graph_objects")
    positions = display.node_positions
    normals = node_surface_normals(geometry)
    center = display.brain_center
    node_diameters = scale_values(size_values, node_size_range)
    node_cmap_name = visual["node_cmap"] if node_cmap is None else node_cmap
    node_hover = [
        "<br>".join(
            (
                f"name={name}",
                f"hemi={hemi}",
                f"group={geometry.groups[index] if geometry.groups is not None else 'n/a'}",
                f"strength={prepared.node_strength[index]:.6g}",
            )
        )
        for index, (name, hemi) in enumerate(
            zip(prepared.node_names, geometry.hemispheres, strict=True)
        )
    ]
    node_mesh, node_intensities, node_mesh_hover, node_vertex_counts = _node_mesh_data(
        positions,
        node_diameters,
        color_values,
        node_hover,
    )
    warn_if_many_edges(len(prepared.edges))
    weights = np.array([edge.weight for edge in prepared.edges], dtype=float)
    edge_parts: list[TriangleMesh] = []
    edge_vertex_colors: list[str] = []
    edge_mesh_hover: list[str] = []
    edge_vertex_counts: list[int] = []
    if len(weights):
        normalization, automatic_cmap = edge_color_norm(
            weights,
            vmin=edge_vmin,
            vmax=edge_vmax,
        )
        edge_cmap_name = automatic_cmap if edge_cmap in {None, "auto"} else edge_cmap
        edge_colormap = colormaps[edge_cmap_name]
        edge_diameters = scale_values(
            np.abs(weights),
            tuple(
                visual["edge_width_range"]
                if edge_width_range is None
                else edge_width_range
            ),
        )
    else:
        normalization = None
        edge_colormap = None
        edge_diameters = np.empty(0, dtype=np.float64)
    endpoint_x: list[float] = []
    endpoint_y: list[float] = []
    endpoint_z: list[float] = []
    endpoint_u: list[float] = []
    endpoint_v: list[float] = []
    endpoint_w: list[float] = []
    endpoint_hover: list[str] = []
    for edge, diameter in zip(prepared.edges, edge_diameters, strict=True):
        cross_hemi = (
            geometry.hemispheres[edge.source] != geometry.hemispheres[edge.target]
        )
        curve = quadratic_bezier(
            positions[edge.source],
            positions[edge.target],
            normals[edge.source],
            normals[edge.target],
            cross_hemisphere=cross_hemi,
            brain_center=center,
            arc_height=edge_arc_height,
            arc_min_mm=edge_arc_min_mm,
            arc_max_mm=edge_arc_max_mm,
            samples=edge_samples,
        )
        direction_name = (
            f"{prepared.node_names[edge.source]} → {prepared.node_names[edge.target]}"
        )
        distance_text = (
            "n/a" if edge.distance_mm is None else f"{edge.distance_mm:.6g} mm"
        )
        hover = f"{direction_name}<br>weight={edge.weight:.6g}<br>distance={distance_text}"
        part = tube_mesh(curve, float(diameter) / 2.0, sides=_TUBE_SIDES)
        edge_parts.append(part)
        edge_vertex_counts.append(len(part.vertices))
        assert normalization is not None and edge_colormap is not None
        color = to_hex(edge_colormap(normalization(edge.weight)))
        edge_vertex_colors.extend([color] * len(part.vertices))
        edge_mesh_hover.extend([hover] * len(part.vertices))
        if show_arrows and prepared.directed:
            tangent = curve[-1] - curve[-2]
            tangent /= np.linalg.norm(tangent)
            tip = curve[-1] - tangent * (node_diameters[edge.target] / 2.0)
            endpoint_x.append(float(tip[0]))
            endpoint_y.append(float(tip[1]))
            endpoint_z.append(float(tip[2]))
            endpoint_u.append(float(tangent[0]))
            endpoint_v.append(float(tangent[1]))
            endpoint_w.append(float(tangent[2]))
            endpoint_hover.append(hover)
    edge_mesh = merge_meshes(edge_parts)
    if edge_parts:
        figure.add_trace(
            graph_objects.Mesh3d(
                **_mesh_coordinates(edge_mesh),
                name="Edges",
                vertexcolor=edge_vertex_colors,
                flatshading=False,
                lighting=_EDGE_LIGHTING,
                lightposition=_NETWORK_LIGHT_POSITION,
                opacity=0.96,
                hovertext=edge_mesh_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                hoverinfo="text",
                meta={
                    "diameters": edge_diameters.tolist(),
                    "vertex_counts": edge_vertex_counts,
                },
                showlegend=False,
            )
        )
    node_cmin, node_cmax = _color_limits(color_values)
    figure.add_trace(
        graph_objects.Mesh3d(
            **_mesh_coordinates(node_mesh),
            name="Nodes",
            intensity=node_intensities,
            intensitymode="vertex",
            colorscale=node_cmap_name,
            cmin=node_cmin,
            cmax=node_cmax,
            showscale=True,
            colorbar={"title": "Node value"},
            flatshading=False,
            lighting=_NODE_LIGHTING,
            lightposition=_NETWORK_LIGHT_POSITION,
            hovertext=node_mesh_hover,
            hovertemplate="%{hovertext}<extra></extra>",
            hoverinfo="text",
            meta={
                "diameters": node_diameters.tolist(),
                "vertex_counts": node_vertex_counts,
            },
            showlegend=False,
        )
    )
    if endpoint_x:
        figure.add_trace(
            graph_objects.Cone(
                x=endpoint_x,
                y=endpoint_y,
                z=endpoint_z,
                u=endpoint_u,
                v=endpoint_v,
                w=endpoint_w,
                name="Directions",
                anchor="tip",
                sizemode="absolute",
                sizeref=4.0,
                colorscale=((0.0, "#16181d"), (1.0, "#16181d")),
                showscale=False,
                lighting=_EDGE_LIGHTING,
                lightposition=_NETWORK_LIGHT_POSITION,
                hovertext=endpoint_hover,
                hovertemplate="%{hovertext}<extra></extra>",
                hoverinfo="text",
                showlegend=False,
            )
        )
    figure.update_layout(
        paper_bgcolor=visual["background"],
        plot_bgcolor=visual["background"],
        meta={
            "pyconnviz_version": "0.1.0",
            "render_mode": "ball-and-stick",
            "edge_count": len(prepared.edges),
            "directed": prepared.directed,
            "network_vertices": len(node_mesh.vertices) + len(edge_mesh.vertices),
            "network_triangles": len(node_mesh.faces) + len(edge_mesh.faces),
        },
        showlegend=False,
    )
    static_figure = None
    if resolved_static_views is not None and any(
        path.suffix.lower() in {".png", ".svg", ".pdf"} for path in paths
    ):
        scene_layouts = [
            (
                dict(figure.layout.scene.to_plotly_json())
                if static_view == view
                else _nilearn_scene_layout(
                    geometry,
                    overlay.values,
                    surface_cmap=resolved_surface_cmap,
                    cortex_color=visual["cortex_color"],
                    sulc_contrast=visual["sulc_contrast"],
                    hemi=display.hemi,
                    view=static_view,
                )
            )
            for static_view in resolved_static_views
        ]
        static_figure = _build_static_montage(
            figure,
            resolved_static_views,
            scene_layouts,
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".html":
            figure.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
        elif path.suffix.lower() in {".png", ".svg", ".pdf"}:
            image_options = {
                name: value
                for name, value in {
                    "width": image_width,
                    "height": image_height,
                    "scale": image_scale,
                }.items()
                if value is not None
            }
            export_figure = figure if static_figure is None else static_figure
            export_figure.write_image(path, **image_options)
        else:
            raise ValueError("Plotly output must use .html, .png, .svg, or .pdf")
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Plotly did not create a non-empty output file: {path}")
    if show:
        figure.show()
    return PlotResult(
        backend="surface",
        engine="plotly",
        artist=figure,
        prepared=prepared,
        output_files=paths,
        panel_edges={"both-interactive": prepared.edges},
    )
