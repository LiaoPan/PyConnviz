"""Direct Nilearn ``plot_img_on_surf`` montage with network overlays."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import nibabel as nib
import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..geometry import geometry_from_arrays, node_surface_normals
from ..models import (
    ConnectomeGeometry,
    GeometryError,
    HemisphereMesh,
    PlotResult,
    PreparedConnectome,
    ViewSpec,
)
from ..styles import get_style
from ._surface_common import (
    edge_color_norm,
    panel_key,
    quadratic_bezier,
    resolve_cortex_alpha,
    scale_values,
    select_panel_edges,
    surface_display_coordinates,
    transparent_surface_cmap,
)

_NATIVE_VIEW_PRESETS = {
    "lateral": ("lateral",),
    "four": ("lateral", "medial"),
    "paper": ("lateral", "medial", "dorsal"),
    "whole": ("dorsal", "anterior", "posterior"),
    "single": ("lateral",),
}
_NATIVE_VIEWS = {
    "lateral",
    "medial",
    "dorsal",
    "ventral",
    "anterior",
    "posterior",
}


@dataclass(frozen=True)
class NativeSurfaceMesh:
    """Nilearn inputs and validated PyConnviz meshes for one montage."""

    nilearn_mesh: Mapping[str, Any]
    pial_meshes: Mapping[str, HemisphereMesh]
    display_meshes: Mapping[str, HemisphereMesh]


def resolve_native_views(views: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve an ordered native Nilearn view sequence."""

    if isinstance(views, str):
        try:
            return _NATIVE_VIEW_PRESETS[views]
        except KeyError as error:
            raise ValueError(
                "views must be a native preset or a sequence of Nilearn view strings"
            ) from error
    resolved = tuple(views)
    if (
        not resolved
        or any(not isinstance(view, str) or view not in _NATIVE_VIEWS for view in resolved)
        or len(set(resolved)) != len(resolved)
    ):
        raise ValueError(
            "views must contain unique supported Nilearn view strings"
        )
    return resolved


def freesurfer_surface_paths(subject_dir: str | Path) -> dict[str, Path]:
    """Return a validated Nilearn mesh mapping for one FreeSurfer subject."""

    root = Path(subject_dir)
    surf = root / "surf"
    paths = {
        "pial_left": surf / "lh.pial",
        "pial_right": surf / "rh.pial",
        "infl_left": surf / "lh.inflated",
        "infl_right": surf / "rh.inflated",
        "sulc_left": surf / "lh.sulc",
        "sulc_right": surf / "rh.sulc",
        "curv_left": surf / "lh.curv",
        "curv_right": surf / "rh.curv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise GeometryError(f"Required FreeSurfer surface file does not exist: {missing[0]}")
    return paths


def _hemisphere_mesh(mesh: Any, sulc: Any, *, name: str) -> HemisphereMesh:
    return HemisphereMesh(
        coordinates=np.asarray(mesh.coordinates, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        sulc=np.asarray(sulc, dtype=np.float64),
        name=name,
    )


def resolve_native_surface_mesh(
    geometry: ConnectomeGeometry,
    surf_mesh: str | Path | Mapping[str, Any] | None,
    *,
    inflate: bool,
) -> NativeSurfaceMesh:
    """Resolve native projection and display meshes without downloading data."""

    missing_hemispheres = sorted({"left", "right"} - set(geometry.meshes))
    if missing_hemispheres:
        raise GeometryError(
            "The native Nilearn montage requires left and right surface meshes; "
            f"missing {missing_hemispheres}"
        )
    surface = importlib.import_module("nilearn.surface")
    if surf_mesh is None:
        mapping: dict[str, Any] = {}
        pial_meshes: dict[str, HemisphereMesh] = {}
        display_meshes: dict[str, HemisphereMesh] = {}
        for hemi in ("left", "right"):
            source = geometry.meshes[hemi]
            nilearn_part = surface.InMemoryMesh(
                np.array(source.coordinates, copy=True),
                np.array(source.faces, copy=True),
            )
            background = (
                np.zeros(len(source.coordinates), dtype=np.float64)
                if source.sulc is None
                else np.array(source.sulc, copy=True)
            )
            mapping[f"pial_{hemi}"] = nilearn_part
            mapping[f"infl_{hemi}"] = nilearn_part
            mapping[f"sulc_{hemi}"] = background
            mapping[f"curv_{hemi}"] = background
            pial_meshes[hemi] = source
            display_meshes[hemi] = source
        return NativeSurfaceMesh(mapping, pial_meshes, display_meshes)

    if geometry.node_vertices is None:
        raise GeometryError(
            "geometry.node_vertices is required when surf_mesh supplies an external template"
        )
    if isinstance(surf_mesh, (str, Path)):
        mapping = freesurfer_surface_paths(surf_mesh)
    elif isinstance(surf_mesh, Mapping):
        mapping = dict(surf_mesh)
    else:
        raise TypeError("surf_mesh must be a FreeSurfer directory, mapping, or None")
    required = {
        "pial_left",
        "pial_right",
        "infl_left",
        "infl_right",
        "sulc_left",
        "sulc_right",
    }
    if inflate:
        required.update({"curv_left", "curv_right"})
    missing = sorted(required - set(mapping))
    if missing:
        raise GeometryError(f"surf_mesh is missing required Nilearn keys: {missing}")

    pial_meshes = {}
    display_meshes = {}
    display_prefix = "infl" if inflate else "pial"
    for hemi in ("left", "right"):
        sulc = surface.load_surf_data(mapping[f"sulc_{hemi}"])
        pial = surface.load_surf_mesh(mapping[f"pial_{hemi}"])
        display = surface.load_surf_mesh(mapping[f"{display_prefix}_{hemi}"])
        pial_meshes[hemi] = _hemisphere_mesh(
            pial,
            sulc,
            name=f"native:{hemi}:pial",
        )
        display_meshes[hemi] = _hemisphere_mesh(
            display,
            sulc,
            name=f"native:{hemi}:{display_prefix}",
        )
        expected = len(geometry.meshes[hemi].coordinates)
        actual = len(display_meshes[hemi].coordinates)
        if actual != expected:
            raise GeometryError(
                f"surf_mesh {hemi} vertex count {actual} does not match geometry {expected}"
            )
    return NativeSurfaceMesh(mapping, pial_meshes, display_meshes)


def _neutral_stat_map(
    pial_meshes: Mapping[str, HemisphereMesh],
) -> nib.Nifti1Image:
    """Create a zero-valued Niimg covering the pial mesh bounds."""

    coordinates = np.concatenate(
        [pial_meshes[hemi].coordinates for hemi in ("left", "right")],
        axis=0,
    )
    voxel_size = 4.0
    margin = 10.0
    origin = np.floor((np.min(coordinates, axis=0) - margin) / voxel_size) * voxel_size
    upper = np.ceil((np.max(coordinates, axis=0) + margin - origin) / voxel_size)
    shape = tuple(int(max(value + 1, 2)) for value in upper)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] *= voxel_size
    affine[:3, 3] = origin
    return nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)


def _resolve_hemispheres(hemispheres: str | Sequence[str]) -> tuple[str, ...]:
    if hemispheres == "both":
        return ("left", "right")
    resolved = (
        (hemispheres,) if isinstance(hemispheres, str) else tuple(hemispheres)
    )
    if (
        not resolved
        or any(hemi not in {"left", "right"} for hemi in resolved)
        or len(set(resolved)) != len(resolved)
    ):
        raise ValueError("hemispheres must contain unique 'left' or 'right' values")
    return resolved


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


def _as_output_paths(
    output: str | Path | Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if output is None:
        return ()
    if isinstance(output, (str, Path)):
        return (Path(output),)
    paths = tuple(Path(path) for path in output)
    if not paths:
        raise ValueError("output sequence must not be empty")
    return paths


def _display_geometry(
    geometry: ConnectomeGeometry,
    display_meshes: Mapping[str, HemisphereMesh],
    *,
    surface_name: str,
) -> ConnectomeGeometry:
    if geometry.node_vertices is None:
        raise GeometryError(
            "geometry.node_vertices is required to align native surface nodes"
        )
    surface_coords = np.asarray(
        [
            display_meshes[hemi].coordinates[int(vertex)]
            for hemi, vertex in zip(
                geometry.hemispheres,
                geometry.node_vertices,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    return geometry_from_arrays(
        geometry.node_names,
        surface_coords,
        geometry.hemispheres,
        display_meshes,
        mni_coords=geometry.mni_coords,
        groups=geometry.groups,
        node_vertices=geometry.node_vertices,
        roi_vertices=geometry.roi_vertices,
        subject=geometry.subject,
        surface_name=surface_name,
    )


def plot_surface_nilearn(
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    *,
    stat_map: Any = None,
    surf_mesh: str | Path | Mapping[str, Any] | None = None,
    style: str = "paper",
    views: str | Sequence[str] = "paper",
    hemispheres: str | Sequence[str] = ("left", "right"),
    surface_values: Mapping[str, Any] | None = None,
    node_overlay: str | None = None,
    surface_cmap: str | None = None,
    threshold: float | str | None = None,
    bg_on_data: bool = True,
    symmetric_cmap: bool | None = None,
    symmetric_cbar: bool | str = "auto",
    inflate: bool = False,
    surface_vmin: float | None = None,
    surface_vmax: float | None = None,
    cortex_alpha: float | None = None,
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    node_cmap: str | None = None,
    node_size_range: tuple[float, float] | None = None,
    node_offset_mm: float = 1.5,
    edge_cmap: str | None = None,
    edge_width_range: tuple[float, float] | None = None,
    edge_alpha: float | None = None,
    edge_vmin: float | None = None,
    edge_vmax: float | None = None,
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
    """Render a direct native Nilearn surface montage."""

    if tuple(geometry.node_names) != tuple(prepared.node_names):
        raise ValueError("geometry node_names must match prepared node_names in order")
    if geometry.node_vertices is None:
        raise GeometryError(
            "geometry.node_vertices is required to align native surface nodes"
        )
    if surface_values is not None or node_overlay not in {None, "none"}:
        raise TypeError(
            "surface_values and functional node_overlay modes require the Matplotlib "
            "or Plotly surface engine"
        )
    resolved_views = resolve_native_views(views)
    resolved_hemispheres = _resolve_hemispheres(hemispheres)
    resolved_mesh = resolve_native_surface_mesh(
        geometry,
        surf_mesh,
        inflate=inflate,
    )
    visual = get_style(style)
    resolved_alpha = resolve_cortex_alpha(
        cortex_alpha,
        default=visual["cortex_alpha"],
    )
    neutral = stat_map is None
    resolved_stat_map = (
        _neutral_stat_map(resolved_mesh.pial_meshes) if neutral else stat_map
    )
    resolved_cmap: Any = (
        transparent_surface_cmap()
        if neutral
        else visual["surface_cmap"] if surface_cmap is None else surface_cmap
    )
    resolved_threshold = np.nextafter(0.0, 1.0) if neutral else threshold
    plotting = importlib.import_module("nilearn.plotting")
    rendered = plotting.plot_img_on_surf(
        stat_map=resolved_stat_map,
        surf_mesh=resolved_mesh.nilearn_mesh,
        views=list(resolved_views),
        hemispheres=list(resolved_hemispheres),
        cmap=resolved_cmap,
        colorbar=bool(colorbar and not neutral),
        threshold=resolved_threshold,
        bg_on_data=bg_on_data,
        inflate=inflate,
        vmin=surface_vmin,
        vmax=surface_vmax,
        symmetric_cbar=symmetric_cbar,
        symmetric_cmap=symmetric_cmap,
        title=title,
        alpha=resolved_alpha,
    )
    if rendered is None:
        raise RuntimeError("Nilearn did not return a surface figure")
    figure, axes = rendered
    if figsize is not None:
        figure.set_size_inches(*figsize)
    figure.set_dpi(dpi)
    figure.set_facecolor(visual["background"])

    node_count = len(prepared.node_names)
    base_node_values = (
        prepared.node_strength
        if node_values is None
        else _finite_node_values(node_values, node_count, name="node_values")
    )
    color_values = (
        base_node_values
        if node_color_values is None
        else _finite_node_values(
            node_color_values,
            node_count,
            name="node_color_values",
        )
    )
    size_values = (
        base_node_values
        if node_size_values is None
        else _finite_node_values(
            node_size_values,
            node_count,
            name="node_size_values",
        )
    )
    node_cmap_name = visual["node_cmap"] if node_cmap is None else node_cmap
    node_colormap = colormaps[node_cmap_name]
    node_normalization = _node_norm(color_values)
    node_sizes = scale_values(
        size_values,
        tuple(
            visual["node_size_range"]
            if node_size_range is None
            else node_size_range
        ),
    )
    all_weights = np.asarray(
        [edge.weight for edge in prepared.edges],
        dtype=np.float64,
    )
    edge_normalization, automatic_edge_cmap = edge_color_norm(
        all_weights,
        vmin=edge_vmin,
        vmax=edge_vmax,
    )
    edge_cmap_name = (
        automatic_edge_cmap if edge_cmap in {None, "auto"} else edge_cmap
    )
    edge_colormap = colormaps[edge_cmap_name]
    widths = scale_values(
        np.abs(all_weights),
        tuple(
            visual["edge_width_range"]
            if edge_width_range is None
            else edge_width_range
        ),
    )
    edge_width_by_pair = {
        (edge.source, edge.target): float(width)
        for edge, width in zip(prepared.edges, widths, strict=True)
    }
    display_geometry = _display_geometry(
        geometry,
        resolved_mesh.display_meshes,
        surface_name="inflated" if inflate else "pial",
    )
    normals = node_surface_normals(display_geometry)
    expected_axes = len(resolved_views) * len(resolved_hemispheres)
    panel_axes = list(axes[:expected_axes])
    if len(panel_axes) != expected_axes:
        raise RuntimeError(
            f"Nilearn returned {len(panel_axes)} cortical axes; expected {expected_axes}"
        )
    panel_edges: dict[str, tuple] = {}
    for axis, (view, hemi) in zip(
        panel_axes,
        itertools.product(resolved_views, resolved_hemispheres),
        strict=True,
    ):
        axis.computed_zorder = False
        axis.set_facecolor(visual["background"])
        for collection in axis.collections:
            collection.set_zorder(0)
            if rasterize_surface:
                collection.set_rasterized(True)
        display = surface_display_coordinates(
            display_geometry,
            hemi,
            node_offset_mm=node_offset_mm,
            center=True,
        )
        visible_edges = select_panel_edges(prepared.edges, display_geometry, hemi)
        panel_hemi = cast(Literal["left", "right"], hemi)
        key = panel_key(ViewSpec(panel_hemi, view))
        panel_edges[key] = visible_edges
        curves = []
        colors = []
        line_widths = []
        for edge in visible_edges:
            curve = quadratic_bezier(
                display.node_positions[edge.source],
                display.node_positions[edge.target],
                normals[edge.source],
                normals[edge.target],
                brain_center=display.brain_center,
                arc_height=edge_arc_height,
                arc_min_mm=edge_arc_min_mm,
                arc_max_mm=edge_arc_max_mm,
                samples=edge_samples,
            )
            curves.append(curve)
            colors.append(edge_colormap(edge_normalization(edge.weight)))
            line_widths.append(edge_width_by_pair[(edge.source, edge.target)])
        if curves:
            edge_collection = Line3DCollection(
                curves,
                colors=colors,
                linewidths=line_widths,
                alpha=(
                    visual["edge_alpha"] if edge_alpha is None else edge_alpha
                ),
            )
            edge_collection.set_zorder(10)
            axis.add_collection3d(edge_collection)
            if show_arrows and prepared.directed:
                for curve, color in zip(curves, colors, strict=True):
                    direction = curve[-1] - curve[-2]
                    arrow = axis.quiver(
                        *curve[-2],
                        *direction,
                        color=color,
                        arrow_length_ratio=0.5,
                        linewidth=0.8,
                        normalize=False,
                    )
                    arrow.set_zorder(11)
        node_indices = [
            index
            for index, node_hemi in enumerate(display_geometry.hemispheres)
            if node_hemi == hemi
        ]
        if node_indices:
            points = display.node_positions[node_indices]
            node_collection = axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=node_sizes[node_indices],
                c=color_values[node_indices],
                cmap=node_colormap,
                norm=node_normalization,
                edgecolors=visual["node_edgecolor"],
                linewidths=0.6,
                depthshade=True,
            )
            node_collection.set_zorder(12)
        axis.set_title(
            f"{hemi.title()} {view}",
            color="white" if style == "dark" else "black",
        )

    if colorbar:
        for axis in figure.axes:
            position = axis.get_position()
            axis.set_position(
                [
                    position.x0 * 0.82,
                    position.y0,
                    position.width * 0.82,
                    position.height,
                ]
            )
        if len(prepared.edges):
            edge_axis = figure.add_axes((0.84, 0.27, 0.018, 0.46))
            edge_mappable = ScalarMappable(
                norm=edge_normalization,
                cmap=edge_colormap,
            )
            edge_bar = figure.colorbar(edge_mappable, cax=edge_axis)
            edge_bar.set_label("Edge weight")
            node_axis_left = 0.93
        else:
            node_axis_left = 0.89
        node_axis = figure.add_axes((node_axis_left, 0.27, 0.018, 0.46))
        node_mappable = ScalarMappable(
            norm=node_normalization,
            cmap=node_colormap,
        )
        node_bar = figure.colorbar(node_mappable, cax=node_axis)
        node_bar.set_label("Node value")

    output_paths = _as_output_paths(output)
    for path in output_paths:
        if path.suffix.lower() not in {".png", ".svg", ".pdf"}:
            raise ValueError("Nilearn surface output must use .png, .svg, or .pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=dpi,
            transparent=transparent,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Nilearn did not create a non-empty output file: {path}")
    if show:
        pyplot = importlib.import_module("matplotlib.pyplot")
        pyplot.show()
    return PlotResult(
        backend="surface",
        engine="nilearn",
        artist=figure,
        prepared=prepared,
        output_files=output_paths,
        panel_edges=panel_edges,
    )
