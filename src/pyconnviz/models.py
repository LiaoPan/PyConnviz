"""Validated, backend-neutral data models used throughout PyConnviz."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


class PyConnvizError(Exception):
    """Base class for all public PyConnviz errors."""


class GeometryError(PyConnvizError, ValueError):
    """Raised when mesh or coordinate geometry is invalid."""


class ConnectivityShapeError(PyConnvizError, ValueError):
    """Raised when connectivity cannot become a dense node-by-node matrix."""


class CoordinateSpaceError(PyConnvizError, ValueError):
    """Raised when a backend is missing coordinates in its required space."""


class OptionalDependencyError(PyConnvizError, ImportError):
    """Raised when an explicitly requested optional feature is unavailable."""


def _float_array(value: Any, *, name: str, ndim: int | None = None) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise GeometryError(f"{name} must have {ndim} dimensions; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GeometryError(f"{name} must contain only finite real values")
    array.setflags(write=False)
    return array


def _integer_array(value: Any, *, name: str, ndim: int | None = None) -> NDArray[np.int64]:
    original = np.asarray(value)
    if not np.issubdtype(original.dtype, np.integer):
        raise GeometryError(f"{name} must contain integer indices")
    array = np.array(original, dtype=np.int64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise GeometryError(f"{name} must have {ndim} dimensions; got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HemisphereMesh:
    """Triangular surface mesh in millimetre display coordinates."""

    coordinates: NDArray[np.float64]
    faces: NDArray[np.int64]
    sulc: NDArray[np.float64] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        coordinates = _float_array(self.coordinates, name="coordinates", ndim=2)
        if coordinates.shape[1:] != (3,):
            raise GeometryError(
                f"coordinates must have shape (n_vertices, 3); got {coordinates.shape}"
            )
        faces = _integer_array(self.faces, name="faces", ndim=2)
        if faces.shape[1:] != (3,):
            raise GeometryError(f"faces must have shape (n_faces, 3); got {faces.shape}")
        if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= len(coordinates)):
            raise GeometryError(
                "faces contain an index outside the valid vertex range "
                f"[0, {max(len(coordinates) - 1, 0)}]"
            )
        sulc = None
        if self.sulc is not None:
            sulc = _float_array(self.sulc, name="sulc", ndim=1)
            if len(sulc) != len(coordinates):
                raise GeometryError(
                    "sulc must have one value per vertex; "
                    f"expected {len(coordinates)}, got {len(sulc)}"
                )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "sulc", sulc)


@dataclass(frozen=True)
class ConnectomeGeometry:
    """Node coordinates and hemisphere meshes with explicit coordinate spaces."""

    node_names: tuple[str, ...]
    surface_coords: NDArray[np.float64]
    hemispheres: tuple[str, ...]
    meshes: Mapping[str, HemisphereMesh]
    mni_coords: NDArray[np.float64] | None = None
    groups: tuple[str, ...] | None = None
    node_vertices: NDArray[np.int64] | None = None
    roi_vertices: tuple[NDArray[np.int64], ...] | None = None
    subject: str | None = None
    surface_name: str | None = None
    coordinate_units: str = "mm"

    def __post_init__(self) -> None:
        node_names = tuple(str(name) for name in self.node_names)
        hemispheres = tuple(str(hemi) for hemi in self.hemispheres)
        node_count = len(node_names)
        surface_coords = _float_array(self.surface_coords, name="surface_coords", ndim=2)
        if surface_coords.shape != (node_count, 3):
            raise GeometryError(
                f"surface_coords must have shape ({node_count}, 3); got {surface_coords.shape}"
            )
        if len(hemispheres) != node_count or any(
            hemi not in {"left", "right"} for hemi in hemispheres
        ):
            raise GeometryError(
                "hemispheres must contain exactly one 'left' or 'right' value per node"
            )
        meshes = dict(self.meshes)
        if any(not isinstance(mesh, HemisphereMesh) for mesh in meshes.values()):
            raise GeometryError("meshes values must be HemisphereMesh instances")
        missing_meshes = sorted(set(hemispheres) - set(meshes))
        if missing_meshes:
            raise GeometryError(f"meshes are missing required hemispheres: {missing_meshes}")
        if self.coordinate_units != "mm":
            raise GeometryError("coordinate_units must be 'mm' (millimetres)")

        mni_coords = None
        if self.mni_coords is not None:
            mni_coords = _float_array(self.mni_coords, name="mni_coords", ndim=2)
            if mni_coords.shape != (node_count, 3):
                raise GeometryError(
                    f"mni_coords must have shape ({node_count}, 3); got {mni_coords.shape}"
                )
        groups = None if self.groups is None else tuple(str(group) for group in self.groups)
        if groups is not None and len(groups) != node_count:
            raise GeometryError(
                f"groups must have length {node_count}; got {len(groups)}"
            )
        node_vertices = None
        if self.node_vertices is not None:
            node_vertices = _integer_array(self.node_vertices, name="node_vertices", ndim=1)
            if node_vertices.shape != (node_count,):
                raise GeometryError(
                    f"node_vertices must have shape ({node_count},); got {node_vertices.shape}"
                )
            for index, (vertex, hemi) in enumerate(zip(node_vertices, hemispheres, strict=True)):
                if vertex < 0 or vertex >= len(meshes[hemi].coordinates):
                    raise GeometryError(
                        f"node_vertices[{index}]={vertex} is outside the {hemi} mesh vertex range"
                    )
        roi_vertices = None
        if self.roi_vertices is not None:
            if len(self.roi_vertices) != node_count:
                raise GeometryError(
                    f"roi_vertices must have length {node_count}; got {len(self.roi_vertices)}"
                )
            converted = []
            for index, (vertices, hemi) in enumerate(
                zip(self.roi_vertices, hemispheres, strict=True)
            ):
                values = _integer_array(vertices, name=f"roi_vertices[{index}]", ndim=1)
                if values.size and (
                    int(values.min()) < 0 or int(values.max()) >= len(meshes[hemi].coordinates)
                ):
                    raise GeometryError(
                        f"roi_vertices[{index}] contains an index outside the {hemi} mesh"
                    )
                converted.append(values)
            roi_vertices = tuple(converted)

        object.__setattr__(self, "node_names", node_names)
        object.__setattr__(self, "surface_coords", surface_coords)
        object.__setattr__(self, "hemispheres", hemispheres)
        object.__setattr__(self, "meshes", MappingProxyType(meshes))
        object.__setattr__(self, "mni_coords", mni_coords)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "node_vertices", node_vertices)
        object.__setattr__(self, "roi_vertices", roi_vertices)


@dataclass(frozen=True)
class Edge:
    """One visible connection."""

    source: int
    target: int
    weight: float
    distance_mm: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.source, bool) or not isinstance(self.source, (int, np.integer)):
            raise ValueError("edge source must be an integer index")
        if isinstance(self.target, bool) or not isinstance(self.target, (int, np.integer)):
            raise ValueError("edge target must be an integer index")
        if self.source < 0 or self.target < 0:
            raise ValueError("edge indices must be non-negative")
        if self.source == self.target:
            raise ValueError("self-edge records are not allowed")
        if not np.isfinite(self.weight):
            raise ValueError("edge weight must be finite")
        if self.distance_mm is not None and (
            not np.isfinite(self.distance_mm) or self.distance_mm < 0
        ):
            raise ValueError("edge distance_mm must be finite and non-negative")
        object.__setattr__(self, "source", int(self.source))
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(self, "weight", float(self.weight))
        if self.distance_mm is not None:
            object.__setattr__(self, "distance_mm", float(self.distance_mm))


@dataclass(frozen=True)
class PreparedConnectome:
    """Scientifically prepared matrix and the final auditable edge set."""

    matrix: NDArray[np.float64]
    visible_matrix: NDArray[np.float64]
    edges: tuple[Edge, ...]
    node_strength: NDArray[np.float64]
    directed: bool
    node_names: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.array(self.matrix, dtype=np.float64, copy=True)
        visible = np.array(self.visible_matrix, dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"matrix must be square; got shape {matrix.shape}")
        if visible.shape != matrix.shape:
            raise ValueError(
                f"visible_matrix must match matrix shape {matrix.shape}; got {visible.shape}"
            )
        if not np.all(np.isfinite(visible)):
            raise ValueError("visible_matrix must contain only finite values")
        node_count = matrix.shape[0]
        strength = np.array(self.node_strength, dtype=np.float64, copy=True)
        if strength.shape != (node_count,):
            raise ValueError(
                f"node_strength must have shape ({node_count},); got {strength.shape}"
            )
        if not np.all(np.isfinite(strength)) or np.any(strength < 0):
            raise ValueError("node_strength must contain finite non-negative values")
        names = tuple(str(name) for name in self.node_names)
        if len(names) != node_count:
            raise ValueError(f"node_names must have length {node_count}; got {len(names)}")
        edges = tuple(self.edges)
        if any(not isinstance(edge, Edge) for edge in edges):
            raise ValueError("edges must contain only Edge instances")
        for edge in edges:
            if edge.source >= node_count or edge.target >= node_count:
                raise ValueError(
                    f"edge index ({edge.source}, {edge.target}) exceeds matrix size {node_count}"
                )
        if np.any(np.diag(visible) != 0):
            raise ValueError("visible_matrix diagonal must be zero")
        directed = bool(self.directed)
        if not directed and not np.array_equal(visible, visible.T):
            raise ValueError("visible_matrix must be symmetric when directed=False")
        reconstructed = np.zeros_like(visible)
        seen_edges: set[tuple[int, int]] = set()
        for edge in edges:
            key = (
                (edge.source, edge.target)
                if directed
                else (
                    min(edge.source, edge.target),
                    max(edge.source, edge.target),
                )
            )
            if key in seen_edges:
                raise ValueError(f"edges contain a duplicate connection: {key}")
            seen_edges.add(key)
            reconstructed[edge.source, edge.target] = edge.weight
            if not directed:
                reconstructed[edge.target, edge.source] = edge.weight
        if not np.array_equal(reconstructed, visible):
            raise ValueError(
                "edges must exactly match every nonzero entry and weight in visible_matrix"
            )
        matrix.setflags(write=False)
        visible.setflags(write=False)
        strength.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "visible_matrix", visible)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "node_strength", strength)
        object.__setattr__(self, "directed", directed)
        object.__setattr__(self, "node_names", names)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ViewSpec:
    """One surface panel view."""

    hemi: Literal["left", "right", "both"]
    view: str | tuple[float, float]
    title: str | None = None

    def __post_init__(self) -> None:
        if self.hemi not in {"left", "right", "both"}:
            raise ValueError("ViewSpec hemi must be 'left', 'right', or 'both'")
        if not isinstance(self.view, str):
            if len(self.view) != 2 or not all(np.isfinite(value) for value in self.view):
                raise ValueError("ViewSpec numeric view must contain finite (elevation, azimuth)")
            object.__setattr__(self, "view", (float(self.view[0]), float(self.view[1])))


@dataclass
class PlotResult:
    """A backend result together with the exact data used to create it."""

    backend: str
    engine: str | None
    artist: Any
    prepared: PreparedConnectome
    output_files: tuple[Path, ...] = ()
    panel_edges: Mapping[str, tuple[Edge, ...]] | None = None

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be a non-empty string")
        if not isinstance(self.prepared, PreparedConnectome):
            raise TypeError("prepared must be a PreparedConnectome")
        self.output_files = tuple(Path(path) for path in self.output_files)
        if self.panel_edges is not None:
            self.panel_edges = MappingProxyType(
                {str(name): tuple(edges) for name, edges in self.panel_edges.items()}
            )
