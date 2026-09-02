"""Pure data algorithms for mapping node or vertex values onto cortical meshes.

Gaussian overlays are visual interpolation only. They do not represent source
reconstruction, cortical propagation, or statistical significance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .models import ConnectomeGeometry, GeometryError, HemisphereMesh


@dataclass(frozen=True)
class SurfaceOverlay:
    """Validated per-hemisphere overlay arrays and provenance metadata."""

    values: Mapping[str, NDArray[np.float64]] | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.values is not None:
            object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _node_values(geometry: ConnectomeGeometry, values: Any) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    expected = (len(geometry.node_names),)
    if array.shape != expected:
        raise GeometryError(f"node_values must have shape {expected}; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GeometryError("node_values must contain only finite real values")
    return array


def _read_only(array: Any) -> NDArray[np.float64]:
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def validate_surface_values(
    geometry: ConnectomeGeometry, surface_values: Mapping[str, Any]
) -> dict[str, NDArray[np.float64]]:
    """Validate and copy real vertex values for every geometry mesh."""

    missing = sorted(set(geometry.meshes) - set(surface_values))
    extra = sorted(set(surface_values) - set(geometry.meshes))
    if missing or extra:
        raise GeometryError(
            f"surface_values hemisphere keys must match geometry; missing={missing}, extra={extra}"
        )
    result: dict[str, NDArray[np.float64]] = {}
    for hemi, mesh in geometry.meshes.items():
        values = np.array(surface_values[hemi], dtype=np.float64, copy=True)
        expected = (len(mesh.coordinates),)
        if values.shape != expected:
            raise GeometryError(
                f"surface_values[{hemi!r}] must have shape {expected}; got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise GeometryError(f"surface_values[{hemi!r}] must contain finite values")
        values.setflags(write=False)
        result[hemi] = values
    return result


def mesh_adjacency(mesh: HemisphereMesh) -> csr_matrix:
    """Build a symmetric sparse mesh graph weighted by Euclidean edge length."""

    edges: dict[tuple[int, int], float] = {}
    for face in mesh.faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            source, target = sorted((int(first), int(second)))
            if source == target:
                continue
            length = float(np.linalg.norm(mesh.coordinates[source] - mesh.coordinates[target]))
            if not np.isfinite(length) or length <= 0:
                continue
            key = (source, target)
            edges[key] = min(edges.get(key, length), length)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for (source, target), length in edges.items():
        rows.extend((source, target))
        columns.extend((target, source))
        data.extend((length, length))
    vertex_count = len(mesh.coordinates)
    return csr_matrix((data, (rows, columns)), shape=(vertex_count, vertex_count))


def truncated_geodesic_distances(
    mesh: HemisphereMesh, source_vertex: int, *, radius_mm: float
) -> NDArray[np.float64]:
    """Find shortest mesh paths from one vertex, limited to ``radius_mm``."""

    if isinstance(source_vertex, bool) or not isinstance(source_vertex, (int, np.integer)):
        raise GeometryError("source_vertex must be an integer mesh index")
    source = int(source_vertex)
    if source < 0 or source >= len(mesh.coordinates):
        raise GeometryError(f"source_vertex {source} is outside the mesh")
    radius = float(radius_mm)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius_mm must be finite and positive")
    distances = dijkstra(
        mesh_adjacency(mesh),
        directed=False,
        indices=source,
        limit=radius,
        return_predecessors=False,
    )
    return np.asarray(distances, dtype=np.float64)


def make_roi_overlay(
    geometry: ConnectomeGeometry,
    node_values: Any,
    *,
    overlap_reducer: Literal["raise", "sum", "max", "mean"] = "raise",
) -> dict[str, NDArray[np.float64]]:
    """Assign each node value to its explicitly recorded ROI vertices."""

    if geometry.roi_vertices is None:
        raise GeometryError("roi overlay requires geometry.roi_vertices")
    if overlap_reducer not in {"raise", "sum", "max", "mean"}:
        raise ValueError("overlap_reducer must be raise, sum, max, or mean")
    values = _node_values(geometry, node_values)
    output = {
        hemi: np.zeros(len(mesh.coordinates), dtype=np.float64)
        for hemi, mesh in geometry.meshes.items()
    }
    counts = {hemi: np.zeros(len(array), dtype=np.int64) for hemi, array in output.items()}
    for index, (hemi, vertices) in enumerate(
        zip(geometry.hemispheres, geometry.roi_vertices, strict=True)
    ):
        existing = counts[hemi][vertices] > 0
        if overlap_reducer == "raise" and np.any(existing):
            conflict = int(vertices[np.flatnonzero(existing)[0]])
            raise GeometryError(
                f"ROI overlap at {hemi} vertex {conflict}; choose an explicit overlap_reducer"
            )
        if overlap_reducer in {"raise", "sum", "mean"}:
            output[hemi][vertices] += values[index]
        elif overlap_reducer == "max":
            unseen = counts[hemi][vertices] == 0
            current = output[hemi][vertices]
            output[hemi][vertices] = np.where(
                unseen, values[index], np.maximum(current, values[index])
            )
        counts[hemi][vertices] += 1
    if overlap_reducer == "mean":
        for hemi in output:
            assigned = counts[hemi] > 0
            output[hemi][assigned] /= counts[hemi][assigned]
    return {hemi: _read_only(values) for hemi, values in output.items()}


def make_gaussian_overlay(
    geometry: ConnectomeGeometry,
    node_values: Any,
    *,
    sigma_mm: float = 12.0,
    radius_mm: float = 30.0,
    distance: Literal["geodesic", "euclidean"] = "geodesic",
    reducer: Literal["sum", "max"] = "sum",
) -> dict[str, NDArray[np.float64]]:
    """Visually interpolate node values over nearby same-hemisphere vertices."""

    sigma = float(sigma_mm)
    radius = float(radius_mm)
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("sigma_mm and radius_mm must be finite and positive")
    if distance not in {"geodesic", "euclidean"}:
        raise ValueError("distance must be 'geodesic' or 'euclidean'")
    if reducer not in {"sum", "max"}:
        raise ValueError("reducer must be 'sum' or 'max'")
    if geometry.node_vertices is None:
        raise GeometryError("gaussian overlay requires geometry.node_vertices")
    values = _node_values(geometry, node_values)
    output = {
        hemi: np.zeros(len(mesh.coordinates), dtype=np.float64)
        for hemi, mesh in geometry.meshes.items()
    }
    assigned = {
        hemi: np.zeros(len(mesh.coordinates), dtype=bool) for hemi, mesh in geometry.meshes.items()
    }
    adjacency = (
        {hemi: mesh_adjacency(mesh) for hemi, mesh in geometry.meshes.items()}
        if distance == "geodesic"
        else {}
    )
    for index, (hemi, source) in enumerate(
        zip(geometry.hemispheres, geometry.node_vertices, strict=True)
    ):
        mesh = geometry.meshes[hemi]
        if distance == "geodesic":
            distances = np.asarray(
                dijkstra(
                    adjacency[hemi],
                    directed=False,
                    indices=int(source),
                    limit=radius,
                    return_predecessors=False,
                ),
                dtype=np.float64,
            )
        else:
            distances = np.linalg.norm(
                mesh.coordinates - mesh.coordinates[int(source)], axis=1
            )
        inside = np.isfinite(distances) & (distances <= radius)
        contribution = values[index] * np.exp(-0.5 * (distances[inside] / sigma) ** 2)
        if reducer == "sum":
            output[hemi][inside] += contribution
        else:
            current = output[hemi][inside]
            previously_assigned = assigned[hemi][inside]
            output[hemi][inside] = np.where(
                previously_assigned, np.maximum(current, contribution), contribution
            )
            assigned[hemi][inside] = True
    return {hemi: _read_only(values) for hemi, values in output.items()}


def overlay_limits(
    values: Mapping[str, NDArray[np.float64]], *, robust: bool = False
) -> tuple[float, float]:
    """Return stable color limits for signed, constant, and empty overlays."""

    finite_parts = [np.asarray(part)[np.isfinite(part)] for part in values.values()]
    combined = np.concatenate([part for part in finite_parts if part.size]) if any(
        part.size for part in finite_parts
    ) else np.array([], dtype=float)
    if not len(combined) or np.all(combined == 0):
        return 0.0, 1.0
    if robust and len(combined) > 2:
        low, high = np.percentile(combined, [2.0, 98.0])
    else:
        low, high = float(np.min(combined)), float(np.max(combined))
    if low < 0 < high:
        limit = float(max(abs(low), abs(high)))
        return -limit, limit
    if high <= 0:
        return float(low), 0.0
    return 0.0, float(high)


def build_surface_overlay(
    mode: Literal["none", "roi", "gaussian", "vertex"],
    geometry: ConnectomeGeometry,
    *,
    node_values: Any = None,
    surface_values: Mapping[str, Any] | None = None,
    overlap_reducer: Literal["raise", "sum", "max", "mean"] = "raise",
    overlay_sigma_mm: float = 12.0,
    overlay_radius_mm: float = 30.0,
    overlay_distance: Literal["geodesic", "euclidean"] = "geodesic",
    overlay_reduce: Literal["sum", "max"] = "sum",
) -> SurfaceOverlay:
    """Build one validated overlay without changing connectivity data."""

    if mode == "none":
        return SurfaceOverlay(None, {"mode": "none"})
    if mode == "vertex":
        if surface_values is None:
            raise GeometryError("vertex overlay requires surface_values")
        values = validate_surface_values(geometry, surface_values)
        return SurfaceOverlay(values, {"mode": "vertex", "source": "user_vertex_values"})
    if node_values is None:
        raise GeometryError(f"{mode} overlay requires node_values")
    if mode == "roi":
        values = make_roi_overlay(
            geometry, node_values, overlap_reducer=overlap_reducer
        )
        return SurfaceOverlay(
            values, {"mode": "roi", "overlap_reducer": overlap_reducer}
        )
    if mode == "gaussian":
        values = make_gaussian_overlay(
            geometry,
            node_values,
            sigma_mm=overlay_sigma_mm,
            radius_mm=overlay_radius_mm,
            distance=overlay_distance,
            reducer=overlay_reduce,
        )
        return SurfaceOverlay(
            values,
            {
                "mode": "gaussian",
                "distance": overlay_distance,
                "sigma_mm": float(overlay_sigma_mm),
                "radius_mm": float(overlay_radius_mm),
                "reducer": overlay_reduce,
                "visual_interpolation": True,
            },
        )
    raise ValueError("mode must be 'none', 'roi', 'gaussian', or 'vertex'")
