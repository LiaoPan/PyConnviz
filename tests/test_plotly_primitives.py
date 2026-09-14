from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest

MODULE_NAME = "pyconnviz.plotting._plotly_primitives"


def _primitives():
    assert find_spec(MODULE_NAME) is not None, (
        "true-3D Plotly primitives module is missing"
    )
    return import_module(MODULE_NAME)


def test_sphere_mesh_is_closed_non_degenerate_and_has_requested_radius() -> None:
    primitives = _primitives()
    center = np.array([1.0, 2.0, 3.0])
    latitude_steps = 6
    longitude_steps = 8

    mesh = primitives.sphere_mesh(
        center,
        2.0,
        latitude_steps=latitude_steps,
        longitude_steps=longitude_steps,
    )

    assert mesh.vertices.shape == (
        2 + (latitude_steps - 1) * longitude_steps,
        3,
    )
    assert mesh.faces.shape == (
        2 * longitude_steps * (latitude_steps - 1),
        3,
    )
    np.testing.assert_allclose(
        np.linalg.norm(mesh.vertices - center, axis=1),
        2.0,
    )
    assert np.min(mesh.faces) == 0
    assert np.max(mesh.faces) == len(mesh.vertices) - 1
    triangles = mesh.vertices[mesh.faces]
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    assert np.all(doubled_areas > 1e-12)


def test_merge_meshes_offsets_later_face_indices() -> None:
    primitives = _primitives()
    first = primitives.TriangleMesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    second = primitives.TriangleMesh(
        vertices=np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )

    merged = primitives.merge_meshes((first, second))

    np.testing.assert_array_equal(merged.vertices, np.vstack((first.vertices, second.vertices)))
    np.testing.assert_array_equal(merged.faces, np.array([[0, 1, 2], [3, 4, 5]]))


def test_merge_meshes_accepts_an_empty_sequence() -> None:
    primitives = _primitives()

    merged = primitives.merge_meshes(())

    assert merged.vertices.shape == (0, 3)
    assert merged.vertices.dtype == np.float64
    assert merged.faces.shape == (0, 3)
    assert merged.faces.dtype == np.int64


def test_tube_mesh_has_physical_radius_and_outward_faces() -> None:
    primitives = _primitives()
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 5.0]])
    radius = 0.75
    sides = 8

    mesh = primitives.tube_mesh(points, radius, sides=sides)

    rings = mesh.vertices.reshape(len(points), sides, 3)
    np.testing.assert_allclose(np.mean(rings, axis=1), points, atol=1e-12)
    radial = rings - points[:, None, :]
    np.testing.assert_allclose(np.linalg.norm(radial, axis=2), radius)
    np.testing.assert_allclose(radial[:, :, 2], 0.0, atol=1e-12)
    assert mesh.faces.shape == (2 * (len(points) - 1) * sides, 3)
    assert np.min(mesh.faces) == 0
    assert np.max(mesh.faces) == len(mesh.vertices) - 1

    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    face_centers = np.mean(triangles, axis=1)
    face_radial = face_centers.copy()
    face_radial[:, 2] = 0.0
    assert np.all(np.einsum("ij,ij->i", normals, face_radial) > 0.0)


@pytest.mark.parametrize(
    ("points", "radius", "sides", "message"),
    (
        (np.array([[0.0, 0.0, 0.0]]), 1.0, 8, "at least two"),
        (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            1.0,
            8,
            "consecutive",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            0.0,
            8,
            "positive",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            1.0,
            2,
            "at least 3",
        ),
    ),
)
def test_tube_mesh_rejects_invalid_geometry(
    points: np.ndarray,
    radius: float,
    sides: int,
    message: str,
) -> None:
    primitives = _primitives()

    with pytest.raises(ValueError, match=message):
        primitives.tube_mesh(points, radius, sides=sides)
