from __future__ import annotations

import numpy as np
import pytest

from pyconnviz.geometry import geometry_from_arrays
from pyconnviz.models import GeometryError, HemisphereMesh
from pyconnviz.surface_overlay import (
    build_surface_overlay,
    make_gaussian_overlay,
    make_roi_overlay,
    mesh_adjacency,
    overlay_limits,
    truncated_geodesic_distances,
    validate_surface_values,
)


def chain_like_mesh(offset: float = 0.0) -> HemisphereMesh:
    return HemisphereMesh(
        np.array(
            [[offset + 0, 0, 0], [offset + 1, 0, 0], [offset + 2, 0, 0]], float
        ),
        np.array([[0, 1, 2]], int),
    )


def bilateral_geometry(*, overlap: bool = False):
    left_roi_2 = np.array([1, 2]) if overlap else np.array([2])
    return geometry_from_arrays(
        ("L1", "L2", "R1"),
        np.array([[0, 0, 0], [2, 0, 0], [10, 0, 0]], float),
        ("left", "left", "right"),
        {"left": chain_like_mesh(), "right": chain_like_mesh(10)},
        node_vertices=np.array([0, 2, 0]),
        roi_vertices=(np.array([0, 1]), left_roi_2, np.array([0, 1])),
    )


def test_mesh_adjacency_uses_physical_edge_lengths() -> None:
    adjacency = mesh_adjacency(chain_like_mesh())
    assert adjacency.shape == (3, 3)
    assert adjacency[0, 1] == pytest.approx(1.0)
    assert adjacency[0, 2] == pytest.approx(2.0)
    np.testing.assert_array_equal(adjacency.toarray(), adjacency.toarray().T)


def test_truncated_geodesic_marks_values_beyond_radius_infinite() -> None:
    distances = truncated_geodesic_distances(chain_like_mesh(), 0, radius_mm=1.1)
    assert distances[0] == 0
    assert distances[1] == pytest.approx(1.0)
    assert np.isinf(distances[2])


def test_roi_overlay_assigns_each_hemisphere() -> None:
    result = make_roi_overlay(bilateral_geometry(), np.array([1.0, 3.0, -2.0]))
    np.testing.assert_allclose(result["left"], [1.0, 1.0, 3.0])
    np.testing.assert_allclose(result["right"], [-2.0, -2.0, 0.0])


@pytest.mark.parametrize(
    ("reducer", "expected"),
    [("sum", 4.0), ("max", 3.0), ("mean", 2.0)],
)
def test_roi_overlap_requires_explicit_reducer(reducer: str, expected: float) -> None:
    geom = bilateral_geometry(overlap=True)
    with pytest.raises(GeometryError, match="overlap"):
        make_roi_overlay(geom, np.array([1.0, 3.0, -2.0]))
    result = make_roi_overlay(
        geom, np.array([1.0, 3.0, -2.0]), overlap_reducer=reducer
    )
    assert result["left"][1] == pytest.approx(expected)


def disconnected_folded_mesh() -> HemisphereMesh:
    return HemisphereMesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.05, 0.0],
                [1.05, 0.05, 0.0],
                [0.05, 1.05, 0.0],
            ]
        ),
        np.array([[0, 1, 2], [3, 4, 5]], int),
    )


def test_geodesic_does_not_jump_across_close_disconnected_fold() -> None:
    mesh = disconnected_folded_mesh()
    geom = geometry_from_arrays(
        ("node",),
        np.array([[0, 0, 0]], float),
        ("left",),
        {"left": mesh},
        node_vertices=np.array([0]),
    )
    geodesic = make_gaussian_overlay(
        geom,
        np.array([1.0]),
        sigma_mm=1.0,
        radius_mm=0.2,
        distance="geodesic",
    )
    euclidean = make_gaussian_overlay(
        geom,
        np.array([1.0]),
        sigma_mm=1.0,
        radius_mm=0.2,
        distance="euclidean",
    )
    assert geodesic["left"][3] == 0
    assert euclidean["left"][3] > 0


def test_gaussian_overlay_is_hemisphere_local_and_radius_truncated() -> None:
    result = make_gaussian_overlay(
        bilateral_geometry(),
        np.array([2.0, 0.0, 5.0]),
        sigma_mm=1.0,
        radius_mm=1.1,
    )
    assert result["left"][0] == pytest.approx(2.0)
    assert result["left"][1] > 0
    assert result["left"][2] == 0
    assert result["right"][0] == pytest.approx(5.0)


@pytest.mark.parametrize(("sigma", "radius"), [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_gaussian_parameters_must_be_positive(sigma: float, radius: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        make_gaussian_overlay(
            bilateral_geometry(), np.ones(3), sigma_mm=sigma, radius_mm=radius
        )


def test_signed_and_zero_gaussian_values_stay_finite() -> None:
    signed = make_gaussian_overlay(
        bilateral_geometry(), np.array([-2.0, 1.0, 0.0]), sigma_mm=2, radius_mm=4
    )
    zeros = make_gaussian_overlay(
        bilateral_geometry(), np.zeros(3), sigma_mm=2, radius_mm=4
    )
    assert np.all(np.isfinite(signed["left"]))
    assert np.all(zeros["left"] == 0)


def test_vertex_values_are_validated_copied_and_read_only() -> None:
    geom = bilateral_geometry()
    values = {"left": np.array([1.0, 2.0, 3.0]), "right": np.zeros(3)}
    result = validate_surface_values(geom, values)
    values["left"][:] = 99
    np.testing.assert_allclose(result["left"], [1, 2, 3])
    assert not result["left"].flags.writeable
    with pytest.raises(GeometryError, match="right"):
        validate_surface_values(geom, {"left": np.zeros(3)})
    with pytest.raises(GeometryError, match="shape"):
        validate_surface_values(geom, {"left": np.zeros(2), "right": np.zeros(3)})


def test_build_surface_overlay_modes_and_metadata() -> None:
    geom = bilateral_geometry()
    none = build_surface_overlay("none", geom)
    assert none.values is None
    assert none.metadata == {"mode": "none"}

    gaussian = build_surface_overlay(
        "gaussian",
        geom,
        node_values=np.ones(3),
        overlay_sigma_mm=2,
        overlay_radius_mm=4,
    )
    assert gaussian.values is not None
    assert gaussian.metadata["distance"] == "geodesic"
    assert gaussian.metadata["visual_interpolation"] is True

    vertex = build_surface_overlay(
        "vertex",
        geom,
        surface_values={"left": np.ones(3), "right": np.ones(3)},
    )
    assert vertex.metadata == {"mode": "vertex", "source": "user_vertex_values"}


def test_overlay_limits_handle_signed_constant_zero_and_robust_values() -> None:
    assert overlay_limits({"left": np.array([-2.0, 1.0])}) == (-2.0, 2.0)
    assert overlay_limits({"left": np.zeros(3)}) == (0.0, 1.0)
    assert overlay_limits({"left": np.full(3, 4.0)}) == (0.0, 4.0)
    low, high = overlay_limits(
        {"left": np.array([0.0, 1.0, 2.0, 1000.0])}, robust=True
    )
    assert low == 0
    assert high < 1000


def test_node_values_are_not_modified() -> None:
    values = np.array([1.0, 2.0, 3.0])
    original = values.copy()
    make_roi_overlay(bilateral_geometry(), values)
    np.testing.assert_array_equal(values, original)
