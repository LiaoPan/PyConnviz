"""Deterministic triangular primitives shared by true-3D renderers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TriangleMesh:
    """A validated collection of vertices and triangular faces."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        faces = np.array(self.faces, dtype=np.int64, copy=True)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("vertices must have shape (n, 3)")
        if faces.ndim != 2 or faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (m, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must contain only finite values")
        if len(faces) and (
            len(vertices) == 0 or np.min(faces) < 0 or np.max(faces) >= len(vertices)
        ):
            raise ValueError("faces contain an out-of-range vertex index")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


def _positive_finite(value: float, *, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return resolved


def sphere_mesh(
    center: ArrayLike,
    radius: float,
    *,
    latitude_steps: int,
    longitude_steps: int,
) -> TriangleMesh:
    """Build a closed UV sphere with unique poles and outward faces."""

    resolved_center = np.asarray(center, dtype=np.float64)
    if resolved_center.shape != (3,) or not np.all(np.isfinite(resolved_center)):
        raise ValueError("center must contain three finite coordinates")
    resolved_radius = _positive_finite(radius, name="radius")
    if latitude_steps < 3:
        raise ValueError("latitude_steps must be at least 3")
    if longitude_steps < 3:
        raise ValueError("longitude_steps must be at least 3")

    vertices: list[NDArray[np.float64]] = [
        resolved_center + np.array([0.0, 0.0, resolved_radius])
    ]
    for latitude in range(1, latitude_steps):
        phi = np.pi * latitude / latitude_steps
        ring_radius = resolved_radius * np.sin(phi)
        z = resolved_radius * np.cos(phi)
        for longitude in range(longitude_steps):
            theta = 2.0 * np.pi * longitude / longitude_steps
            vertices.append(
                resolved_center
                + np.array(
                    [ring_radius * np.cos(theta), ring_radius * np.sin(theta), z]
                )
            )
    south_index = len(vertices)
    vertices.append(resolved_center + np.array([0.0, 0.0, -resolved_radius]))

    faces: list[tuple[int, int, int]] = []
    first_ring = 1
    for longitude in range(longitude_steps):
        current = first_ring + longitude
        next_vertex = first_ring + (longitude + 1) % longitude_steps
        faces.append((0, current, next_vertex))

    for latitude in range(latitude_steps - 2):
        upper = first_ring + latitude * longitude_steps
        lower = upper + longitude_steps
        for longitude in range(longitude_steps):
            next_longitude = (longitude + 1) % longitude_steps
            a = upper + longitude
            b = lower + longitude
            c = lower + next_longitude
            d = upper + next_longitude
            faces.extend(((a, b, c), (a, c, d)))

    last_ring = south_index - longitude_steps
    for longitude in range(longitude_steps):
        current = last_ring + longitude
        next_vertex = last_ring + (longitude + 1) % longitude_steps
        faces.append((current, south_index, next_vertex))

    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _unit(vector: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have non-zero finite length")
    return vector / length


def tube_mesh(points: ArrayLike, radius: float, *, sides: int) -> TriangleMesh:
    """Build an open, smoothly framed tube around a polyline."""

    resolved_points = np.asarray(points, dtype=np.float64)
    if resolved_points.ndim != 2 or resolved_points.shape[1:] != (3,):
        raise ValueError("points must have shape (n, 3)")
    if len(resolved_points) < 2:
        raise ValueError("points must contain at least two coordinates")
    if not np.all(np.isfinite(resolved_points)):
        raise ValueError("points must contain only finite coordinates")
    resolved_radius = _positive_finite(radius, name="radius")
    if sides < 3:
        raise ValueError("sides must be at least 3")

    segments = np.diff(resolved_points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= np.finfo(np.float64).eps):
        raise ValueError("consecutive points must be distinct")
    unit_segments = segments / segment_lengths[:, None]
    tangents = np.empty_like(resolved_points)
    tangents[0] = unit_segments[0]
    tangents[-1] = unit_segments[-1]
    for index in range(1, len(resolved_points) - 1):
        candidate = unit_segments[index - 1] + unit_segments[index]
        if np.linalg.norm(candidate) <= np.finfo(np.float64).eps:
            candidate = unit_segments[index]
        tangents[index] = _unit(candidate, name="polyline tangent")

    axes = np.eye(3, dtype=np.float64)
    reference = axes[int(np.argmin(np.abs(tangents[0])))]
    normal = _unit(np.cross(tangents[0], reference), name="initial tube normal")
    normals = np.empty_like(resolved_points)
    binormals = np.empty_like(resolved_points)
    for index, tangent in enumerate(tangents):
        if index:
            projected = normal - np.dot(normal, tangent) * tangent
            if np.linalg.norm(projected) <= np.finfo(np.float64).eps:
                reference = axes[int(np.argmin(np.abs(tangent)))]
                projected = np.cross(tangent, reference)
            normal = _unit(projected, name="transported tube normal")
        normals[index] = normal
        binormals[index] = _unit(
            np.cross(tangent, normal),
            name="transported tube binormal",
        )

    angles = 2.0 * np.pi * np.arange(sides, dtype=np.float64) / sides
    rings = (
        resolved_points[:, None, :]
        + resolved_radius * np.cos(angles)[None, :, None] * normals[:, None, :]
        + resolved_radius * np.sin(angles)[None, :, None] * binormals[:, None, :]
    )

    faces: list[tuple[int, int, int]] = []
    for ring in range(len(resolved_points) - 1):
        current_ring = ring * sides
        next_ring = (ring + 1) * sides
        for side in range(sides):
            next_side = (side + 1) % sides
            a = current_ring + side
            b = next_ring + side
            c = next_ring + next_side
            d = current_ring + next_side
            faces.extend(((a, d, c), (a, c, b)))

    return TriangleMesh(
        vertices=rings.reshape(-1, 3),
        faces=np.asarray(faces, dtype=np.int64),
    )


def cone_mesh(
    base_center: ArrayLike,
    tip: ArrayLike,
    radius: float,
    *,
    sides: int,
) -> TriangleMesh:
    """Build a closed cone with an outward-facing base and side surface."""

    resolved_base = np.asarray(base_center, dtype=np.float64)
    resolved_tip = np.asarray(tip, dtype=np.float64)
    if resolved_base.shape != (3,) or not np.all(np.isfinite(resolved_base)):
        raise ValueError("base_center must contain three finite coordinates")
    if resolved_tip.shape != (3,) or not np.all(np.isfinite(resolved_tip)):
        raise ValueError("tip must contain three finite coordinates")
    axis = resolved_tip - resolved_base
    direction = _unit(axis, name="base_center and tip must be distinct")
    resolved_radius = _positive_finite(radius, name="radius")
    if sides < 3:
        raise ValueError("sides must be at least 3")

    coordinate_axes = np.eye(3, dtype=np.float64)
    reference = coordinate_axes[int(np.argmin(np.abs(direction)))]
    normal = _unit(np.cross(direction, reference), name="cone base normal")
    binormal = _unit(np.cross(direction, normal), name="cone base binormal")
    angles = 2.0 * np.pi * np.arange(sides, dtype=np.float64) / sides
    rim = (
        resolved_base[None, :]
        + resolved_radius * np.cos(angles)[:, None] * normal[None, :]
        + resolved_radius * np.sin(angles)[:, None] * binormal[None, :]
    )
    vertices = np.vstack((resolved_base, rim, resolved_tip))
    tip_index = sides + 1
    faces: list[tuple[int, int, int]] = []
    for side in range(sides):
        current = 1 + side
        next_vertex = 1 + (side + 1) % sides
        faces.extend(
            (
                (tip_index, current, next_vertex),
                (0, next_vertex, current),
            )
        )
    return TriangleMesh(vertices, np.asarray(faces, dtype=np.int64))


def merge_meshes(meshes: Sequence[TriangleMesh]) -> TriangleMesh:
    """Merge disconnected meshes while preserving face topology."""

    if not meshes:
        return TriangleMesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
        )
    vertices: list[NDArray[np.float64]] = []
    faces: list[NDArray[np.int64]] = []
    offset = 0
    for mesh in meshes:
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return TriangleMesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
    )
