"""Backend-neutral helpers shared by static and interactive surface renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np
from matplotlib.colors import (
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
    to_rgb,
    to_rgba_array,
)
from numpy.typing import NDArray

from ..geometry import offset_node_coordinates
from ..models import ConnectomeGeometry, Edge, ViewSpec

SurfaceHemisphere = Literal["left", "right", "both"]


def resolve_cortex_alpha(value: object, *, default: object) -> float:
    """Resolve a style or caller opacity as one finite unit-interval value."""

    selected = default if value is None else value
    if isinstance(selected, (bool, np.bool_)) or not isinstance(selected, Real):
        raise TypeError("cortex_alpha must be a real number between 0 and 1")
    alpha = float(selected)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("cortex_alpha must be finite and between 0 and 1")
    return alpha


@dataclass(frozen=True)
class SurfaceDisplayCoordinates:
    """Surface, node, and arc coordinates after one Nilearn display transform."""

    hemi: SurfaceHemisphere
    node_positions: NDArray[np.float64]
    surface_coordinates: NDArray[np.float64]
    brain_center: NDArray[np.float64]


def _finite_points_3d(values: Any, *, name: str) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _unit_interval(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number between 0 and 1")
    resolved = float(value)
    if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return resolved


def _projected_depths(
    points: NDArray[np.float64], projection: NDArray[np.float64]
) -> NDArray[np.float64]:
    if projection.shape != (4, 4) or not np.all(np.isfinite(projection)):
        raise ValueError("projection must be a finite 4-by-4 matrix")
    if not len(points):
        return np.empty(0, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    clip = homogeneous @ projection.T
    if np.any(np.isclose(clip[:, 3], 0.0)):
        raise ValueError("projection maps a point to an invalid homogeneous depth")
    depths = clip[:, 2] / clip[:, 3]
    if not np.all(np.isfinite(depths)):
        raise ValueError("projection produced a non-finite depth")
    return depths


def projected_depth_factors(
    points: Any,
    reference_points: Any,
    projection: Any,
    *,
    minimum: float,
) -> NDArray[np.float64]:
    """Return view-depth opacity factors without changing data encodings."""

    selected = _finite_points_3d(points, name="points")
    reference = _finite_points_3d(reference_points, name="reference_points")
    if not len(reference):
        raise ValueError("reference_points must contain at least one coordinate")
    matrix = np.asarray(projection, dtype=np.float64)
    floor = _unit_interval(minimum, name="minimum")
    depths = _projected_depths(selected, matrix)
    reference_depths = _projected_depths(reference, matrix)
    depth_min = float(np.min(reference_depths))
    depth_range = float(np.ptp(reference_depths))
    if depth_range <= np.finfo(np.float64).eps:
        return np.ones(len(selected), dtype=np.float64)
    normalized = np.clip((depths - depth_min) / depth_range, 0.0, 1.0)
    return 1.0 - normalized * (1.0 - floor)


def depth_cued_line_data(
    curves: Sequence[Any],
    colors: Any,
    widths: Any,
    *,
    reference_points: Any,
    projection: Any,
    minimum: float,
    alpha: float,
    enabled: bool,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """Prepare batched 3D line paths with optional per-segment depth alpha."""

    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("enabled must be a bool")
    base_alpha = _unit_interval(alpha, name="alpha")
    floor = _unit_interval(minimum, name="minimum")
    resolved_curves = [
        _finite_points_3d(curve, name=f"curves[{index}]")
        for index, curve in enumerate(curves)
    ]
    if any(len(curve) < 2 for curve in resolved_curves):
        raise ValueError("each curve must contain at least two points")
    resolved_colors = to_rgba_array(colors)
    resolved_widths = np.asarray(widths, dtype=np.float64)
    expected = len(resolved_curves)
    if resolved_colors.shape != (expected, 4):
        raise ValueError("colors must contain one color per curve")
    if resolved_widths.shape != (expected,):
        raise ValueError("widths must contain one value per curve")
    if not np.all(np.isfinite(resolved_widths)) or np.any(resolved_widths < 0.0):
        raise ValueError("widths must contain finite non-negative values")
    if not enabled:
        uniform_colors = np.array(resolved_colors, copy=True)
        uniform_colors[:, 3] *= base_alpha
        return resolved_curves, uniform_colors, np.array(resolved_widths, copy=True)

    paths: list[NDArray[np.float64]] = []
    segment_colors: list[NDArray[np.float64]] = []
    segment_widths: list[NDArray[np.float64]] = []
    for curve, color, width in zip(
        resolved_curves,
        resolved_colors,
        resolved_widths,
        strict=True,
    ):
        segments = np.stack((curve[:-1], curve[1:]), axis=1)
        midpoints = np.mean(segments, axis=1)
        factors = projected_depth_factors(
            midpoints,
            reference_points,
            projection,
            minimum=floor,
        )
        repeated_colors = np.repeat(color[np.newaxis, :], len(segments), axis=0)
        repeated_colors[:, 3] *= base_alpha * factors
        paths.extend(segments)
        segment_colors.append(repeated_colors)
        segment_widths.append(np.full(len(segments), width, dtype=np.float64))
    return (
        paths,
        np.concatenate(segment_colors, axis=0),
        np.concatenate(segment_widths),
    )

_VIEW_PRESETS: dict[str, tuple[ViewSpec, ...]] = {
    "lateral": (
        ViewSpec("left", "lateral", "Left lateral"),
        ViewSpec("right", "lateral", "Right lateral"),
    ),
    "four": (
        ViewSpec("left", "lateral", "Left lateral"),
        ViewSpec("right", "lateral", "Right lateral"),
        ViewSpec("left", "medial", "Left medial"),
        ViewSpec("right", "medial", "Right medial"),
    ),
    "paper": (
        ViewSpec("left", "lateral", "Left lateral"),
        ViewSpec("right", "lateral", "Right lateral"),
        ViewSpec("both", "dorsal", "Whole-brain dorsal"),
    ),
    "whole": (
        ViewSpec("both", "dorsal", "Dorsal"),
        ViewSpec("both", "anterior", "Anterior"),
        ViewSpec("both", "posterior", "Posterior"),
    ),
    "single": (ViewSpec("both", (20.0, -90.0), "Whole brain"),),
}


def resolve_views(views: str | Sequence[ViewSpec]) -> tuple[ViewSpec, ...]:
    """Expand one named view preset or validate an explicit view sequence."""

    if isinstance(views, str):
        try:
            return _VIEW_PRESETS[views]
        except KeyError as error:
            raise ValueError(
                "views must be lateral, four, paper, whole, single, or a ViewSpec sequence"
            ) from error
    resolved = tuple(views)
    if not resolved or any(not isinstance(view, ViewSpec) for view in resolved):
        raise ValueError("views sequence must contain at least one ViewSpec")
    return resolved


def panel_key(view: ViewSpec) -> str:
    """Return a deterministic audit key for one surface panel."""

    view_name = (
        view.view
        if isinstance(view.view, str)
        else f"{view.view[0]:g}-{view.view[1]:g}"
    )
    return f"{view.hemi}-{view_name}"


def resolve_surface_hemi(
    geometry: ConnectomeGeometry, requested_hemi: str
) -> SurfaceHemisphere | None:
    """Resolve one renderable hemisphere without inventing a missing mesh."""

    if requested_hemi not in {"left", "right", "both"}:
        raise ValueError("requested_hemi must be 'left', 'right', or 'both'")
    if requested_hemi == "left":
        return "left" if "left" in geometry.meshes else None
    if requested_hemi == "right":
        return "right" if "right" in geometry.meshes else None
    present = tuple(hemi for hemi in ("left", "right") if hemi in geometry.meshes)
    if len(present) == 2:
        return "both"
    if len(present) == 1:
        return "left" if present[0] == "left" else "right"
    return None


def surface_display_coordinates(
    geometry: ConnectomeGeometry,
    hemi: str,
    *,
    node_offset_mm: float = 1.5,
    center: bool,
) -> SurfaceDisplayCoordinates:
    """Mirror Nilearn's public hemisphere combination and engine centring."""

    effective_hemi = resolve_surface_hemi(geometry, hemi)
    if effective_hemi is None:
        raise ValueError(f"No surface mesh is available for hemisphere {hemi!r}")
    selected = (
        ("left", "right") if effective_hemi == "both" else (effective_hemi,)
    )
    translations = {name: np.zeros(3, dtype=np.float64) for name in selected}
    if effective_hemi == "both":
        left = geometry.meshes["left"].coordinates
        right = geometry.meshes["right"].coordinates
        translations["right"][0] = float(
            np.max(left[:, 0]) - np.min(right[:, 0]) + 1.0
        )

    surface_parts = [
        np.asarray(geometry.meshes[name].coordinates, dtype=np.float64)
        + translations[name]
        for name in selected
    ]
    surface_coordinates = np.concatenate(surface_parts, axis=0)
    centering = (
        np.mean(surface_coordinates, axis=0)
        if center
        else np.zeros(3, dtype=np.float64)
    )
    surface_coordinates = surface_coordinates - centering

    node_positions = np.array(
        offset_node_coordinates(geometry, offset_mm=node_offset_mm), copy=True
    )
    for index, node_hemi in enumerate(geometry.hemispheres):
        translation = translations.get(node_hemi)
        if translation is not None:
            node_positions[index] += translation
    node_positions -= centering

    return SurfaceDisplayCoordinates(
        hemi=effective_hemi,
        node_positions=node_positions,
        surface_coordinates=surface_coordinates,
        brain_center=np.mean(surface_coordinates, axis=0),
    )


def surface_background_values(
    geometry: ConnectomeGeometry,
    *,
    cortex_color: Any,
    sulc_contrast: float,
) -> dict[str, NDArray[np.float64]]:
    """Create restrained Nilearn grayscale indices from sulcal curvature."""

    contrast = float(sulc_contrast)
    if not np.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError("sulc_contrast must be finite and between 0 and 1")
    base = float(np.clip(1.0 - np.mean(to_rgb(cortex_color)), 0.0, 1.0))
    amplitude = 0.32 * contrast
    result: dict[str, NDArray[np.float64]] = {}
    for hemi, mesh in geometry.meshes.items():
        if mesh.sulc is None:
            result[hemi] = np.full(len(mesh.coordinates), base, dtype=np.float64)
            continue
        values = np.asarray(mesh.sulc, dtype=np.float64)
        centered = values - float(np.median(values))
        scale = float(np.percentile(np.abs(centered), 95.0))
        if scale <= np.finfo(float).eps:
            normalized = np.zeros_like(centered)
        else:
            normalized = np.clip(centered / scale, -1.0, 1.0)
        result[hemi] = np.clip(base + amplitude * normalized, 0.0, 1.0)
    return result


def transparent_surface_cmap() -> ListedColormap:
    """Return an invisible map that activates Nilearn's background pipeline."""

    return ListedColormap(
        np.zeros((2, 4), dtype=np.float64), name="pyconnviz-transparent"
    )


def has_visible_surface_values(values: Mapping[str, np.ndarray] | None) -> bool:
    """Return whether an overlay contains any nonzero display value."""

    return values is not None and any(np.any(np.asarray(part) != 0) for part in values.values())


def select_panel_edges(
    edges: Sequence[Edge], geometry: ConnectomeGeometry, hemi: str
) -> tuple[Edge, ...]:
    """Apply display-only hemisphere scope without changing the full edge set."""

    if hemi == "both":
        return tuple(edges)
    if hemi not in {"left", "right"}:
        raise ValueError("panel hemi must be 'left', 'right', or 'both'")
    return tuple(
        edge
        for edge in edges
        if geometry.hemispheres[edge.source] == hemi
        and geometry.hemispheres[edge.target] == hemi
    )


def quadratic_bezier(
    start: NDArray[np.float64],
    end: NDArray[np.float64],
    start_normal: NDArray[np.float64],
    end_normal: NDArray[np.float64],
    *,
    cross_hemisphere: bool = False,
    brain_center: NDArray[np.float64] | None = None,
    arc_height: float = 0.16,
    arc_min_mm: float = 4.0,
    arc_max_mm: float = 35.0,
    samples: int = 40,
) -> NDArray[np.float64]:
    """Generate an outward quadratic Bezier arc with exact endpoints."""

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    start_normal = np.asarray(start_normal, dtype=np.float64)
    end_normal = np.asarray(end_normal, dtype=np.float64)
    if any(value.shape != (3,) for value in (start, end, start_normal, end_normal)):
        raise ValueError("Bezier endpoints and normals must each have shape (3,)")
    if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    distance = float(np.linalg.norm(end - start))
    height = float(np.clip(distance * arc_height, arc_min_mm, arc_max_mm))
    midpoint = (start + end) / 2.0
    if cross_hemisphere:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = start_normal + end_normal
        length = float(np.linalg.norm(direction))
        if length <= np.finfo(float).eps:
            center = np.zeros(3) if brain_center is None else np.asarray(brain_center, dtype=float)
            direction = midpoint - center
            length = float(np.linalg.norm(direction))
        if length <= np.finfo(float).eps:
            direction = np.array([0.0, 0.0, 1.0])
        else:
            direction = direction / length
    control = midpoint + direction * height
    time = np.linspace(0.0, 1.0, int(samples))[:, np.newaxis]
    return (1 - time) ** 2 * start + 2 * (1 - time) * time * control + time**2 * end


def scale_values(
    values: NDArray[np.float64], output_range: tuple[float, float]
) -> NDArray[np.float64]:
    """Scale finite values robustly while keeping constants and NaNs visible."""

    array = np.asarray(values, dtype=np.float64)
    low, high = (float(output_range[0]), float(output_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError("output_range must contain finite increasing values")
    if not array.size:
        return np.array([], dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.full(array.shape, (low + high) / 2.0)
    minimum = float(np.min(array[finite]))
    maximum = float(np.max(array[finite]))
    if np.isclose(minimum, maximum):
        return np.full(array.shape, (low + high) / 2.0)
    safe = np.where(finite, array, minimum)
    return low + (safe - minimum) / (maximum - minimum) * (high - low)


def edge_color_norm(
    weights: NDArray[np.float64],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Normalize, str]:
    """Choose signed or one-sided color semantics without inferring science."""

    values = np.asarray(weights, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return Normalize(vmin=0.0, vmax=1.0), "viridis"
    minimum = float(np.min(finite) if vmin is None else vmin)
    maximum = float(np.max(finite) if vmax is None else vmax)
    has_negative = np.any(finite < 0)
    has_positive = np.any(finite > 0)
    if has_negative and has_positive:
        limit = max(abs(minimum), abs(maximum))
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), "RdBu_r"
    if has_negative:
        minimum = min(minimum, -np.finfo(float).eps)
        return Normalize(vmin=minimum, vmax=0.0), "Blues_r"
    maximum = max(maximum, np.finfo(float).eps)
    return Normalize(vmin=0.0, vmax=maximum), "viridis"
