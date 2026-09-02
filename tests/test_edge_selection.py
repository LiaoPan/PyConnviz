from __future__ import annotations

import numpy as np
import pytest

from pyconnviz.connectivity import prepare_connectome
from pyconnviz.models import ConnectivityShapeError, ConnectomeGeometry, HemisphereMesh


def geometry(coords: np.ndarray) -> ConnectomeGeometry:
    mesh = HemisphereMesh(
        coordinates=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float),
        faces=np.array([[0, 1, 2], [1, 2, 3]], int),
    )
    return ConnectomeGeometry(
        tuple(f"n{i}" for i in range(len(coords))),
        np.asarray(coords, float),
        tuple("left" for _ in coords),
        {"left": mesh},
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("mean", 2.0),
        ("maxabs", 3.0),
        ("lower", 3.0),
        ("upper", 1.0),
    ],
)
def test_explicit_symmetrization_modes(mode: str, expected: float) -> None:
    result = prepare_connectome(
        np.array([[0.0, 1.0], [3.0, 0.0]]), symmetrize=mode, directed=False
    )
    assert result.matrix[0, 1] == pytest.approx(expected)
    assert result.matrix[1, 0] == pytest.approx(expected)
    assert result.directed is False


def test_numpy_triangle_is_directed_in_auto_mode() -> None:
    lower = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    result = prepare_connectome(lower)
    assert result.directed is True
    assert {(edge.source, edge.target) for edge in result.edges} == {(1, 0), (2, 0), (2, 1)}


def test_forcing_undirected_asymmetric_matrix_requires_symmetrize() -> None:
    with pytest.raises(ValueError, match="symmetrize"):
        prepare_connectome(np.array([[0.0, 1.0], [0.0, 0.0]]), directed=False)


def test_directed_symmetric_matrix_keeps_both_directions() -> None:
    result = prepare_connectome(
        np.array([[0.0, 2.0], [2.0, 0.0]]), directed=True
    )
    assert [(edge.source, edge.target) for edge in result.edges] == [(0, 1), (1, 0)]


def test_invalid_diagonal_nan_and_inf_never_become_edges() -> None:
    values = np.array(
        [[99.0, np.nan, 0.5], [np.nan, 88.0, np.inf], [0.5, np.inf, 77.0]]
    )
    result = prepare_connectome(values)
    assert [(edge.source, edge.target, edge.weight) for edge in result.edges] == [
        (0, 2, 0.5)
    ]
    assert np.all(np.diag(result.visible_matrix) == 0)


def test_undirected_mask_must_be_symmetric_and_directed_mask_need_not_be() -> None:
    values = np.array([[0.0, 1.0], [1.0, 0.0]])
    mask = np.array([[False, True], [False, False]])
    with pytest.raises(ValueError, match="symmetric"):
        prepare_connectome(values, edge_mask=mask)
    result = prepare_connectome(values, directed=True, edge_mask=mask)
    assert [(edge.source, edge.target) for edge in result.edges] == [(0, 1)]


@pytest.mark.parametrize(
    ("keep_sign", "weights"),
    [
        ("both", [-2.0, 1.0]),
        ("positive", [1.0]),
        ("negative", [-2.0]),
    ],
)
def test_sign_filtering(keep_sign: str, weights: list[float]) -> None:
    values = np.array([[0, 1, -2], [1, 0, 0], [-2, 0, 0]], float)
    result = prepare_connectome(values, keep_sign=keep_sign)
    assert [edge.weight for edge in result.edges] == weights


def test_distance_filter_requires_geometry_and_records_distance() -> None:
    values = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]], float)
    with pytest.raises(ValueError, match="geometry"):
        prepare_connectome(values, min_distance_mm=5)
    geom = geometry(np.array([[0, 0, 0], [2, 0, 0], [10, 0, 0]], float))
    result = prepare_connectome(values, geometry=geom, min_distance_mm=5)
    assert {(edge.source, edge.target) for edge in result.edges} == {(0, 2), (1, 2)}
    assert [edge.distance_mm for edge in result.edges] == pytest.approx([8.0, 10.0])
    assert result.metadata["distance_coordinate"] == "surface_coords"


def test_numeric_threshold_is_inclusive() -> None:
    values = np.array([[0, 0.5, 0.49], [0.5, 0, 0], [0.49, 0, 0]], float)
    result = prepare_connectome(values, edge_threshold=0.5)
    assert [edge.weight for edge in result.edges] == [0.5]


def test_percentile_threshold_is_strictly_above_the_resolved_boundary() -> None:
    values = np.array(
        [[0.0, 0.1, 0.9], [0.1, 0.0, 0.5], [0.9, 0.5, 0.0]],
        dtype=float,
    )

    result = prepare_connectome(values, edge_threshold="50%")

    assert result.metadata["resolved_edge_threshold"] == pytest.approx(0.5)
    assert [(edge.source, edge.target, edge.weight) for edge in result.edges] == [
        (0, 2, 0.9)
    ]


def test_percentile_population_includes_finite_zero_pairs_like_nilearn() -> None:
    values = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 2.0, 0.0, 4.0],
            [1.0, 3.0, 4.0, 0.0],
        ]
    )

    result = prepare_connectome(values, edge_threshold="50%")

    assert result.metadata["resolved_edge_threshold"] == pytest.approx(1.5)
    assert [edge.weight for edge in result.edges] == [4.0, 3.0, 2.0]


def test_percentile_uses_only_candidates_after_mask_sign_and_distance() -> None:
    values = np.array(
        [
            [0, 100, 1, 0],
            [100, 0, 2, 0],
            [1, 2, 0, 4],
            [0, 0, 4, 0],
        ],
        float,
    )
    mask = np.ones((4, 4), bool)
    mask[0, 1] = mask[1, 0] = False
    geom = geometry(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [20, 0, 0]], float))
    result = prepare_connectome(
        values,
        geometry=geom,
        edge_mask=mask,
        keep_sign="positive",
        min_distance_mm=2,
        edge_threshold="50%",
    )
    assert [(edge.source, edge.target, edge.weight) for edge in result.edges] == [
        (2, 3, 4.0)
    ]
    assert result.metadata["resolved_edge_threshold"] == pytest.approx(2.5)


def test_empty_percentile_candidate_set_is_valid() -> None:
    result = prepare_connectome(np.zeros((3, 3)), edge_threshold="95%")
    assert result.edges == ()
    assert np.count_nonzero(result.visible_matrix) == 0


def test_top_k_is_stable_by_absolute_weight_source_target() -> None:
    values = np.array(
        [[0, -1, 1, 1], [-1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], float
    )
    result = prepare_connectome(values, max_edges=2)
    assert [(edge.source, edge.target) for edge in result.edges] == [(0, 1), (0, 2)]


def test_visible_matrix_and_edges_reconstruct_each_other() -> None:
    values = np.array([[0, 1, -3], [1, 0, 2], [-3, 2, 0]], float)
    result = prepare_connectome(values, max_edges=2)
    rebuilt = np.zeros_like(values)
    for edge in result.edges:
        rebuilt[edge.source, edge.target] = edge.weight
        rebuilt[edge.target, edge.source] = edge.weight
    np.testing.assert_array_equal(result.visible_matrix, rebuilt)


def test_strength_comes_from_visible_matrix() -> None:
    values = np.array([[0, 1, 9], [1, 0, 2], [9, 2, 0]], float)
    undirected = prepare_connectome(values, max_edges=1)
    np.testing.assert_allclose(undirected.node_strength, [9, 0, 9])

    directed_values = np.array([[0, 2, 0], [0, 0, -3], [4, 0, 0]], float)
    total = prepare_connectome(directed_values, strength_mode="total")
    incoming = prepare_connectome(directed_values, strength_mode="in")
    outgoing = prepare_connectome(directed_values, strength_mode="out")
    np.testing.assert_allclose(incoming.node_strength, [4, 2, 3])
    np.testing.assert_allclose(outgoing.node_strength, [2, 3, 4])
    np.testing.assert_allclose(total.node_strength, [6, 5, 7])


@pytest.mark.parametrize("bad_threshold", ["95", "-2%", "101%", -0.1])
def test_invalid_thresholds_are_rejected(bad_threshold) -> None:
    with pytest.raises(ValueError, match="edge_threshold"):
        prepare_connectome(np.zeros((2, 2)), edge_threshold=bad_threshold)


@pytest.mark.parametrize("bad_max", [-1, 1.2, True])
def test_invalid_max_edges_are_rejected(bad_max) -> None:
    with pytest.raises(ValueError, match="max_edges"):
        prepare_connectome(np.zeros((2, 2)), max_edges=bad_max)


def test_declared_symmetric_mne_triangles_are_recovered_but_mixed_data_are_not() -> None:
    class FakeConnectivity:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data
            self.n_nodes = 2
            self.names = ("A", "B")
            self.dims = ("node_in", "node_out")
            self.coords = {}
            self.attrs = {"symmetric": True}

        def get_data(self, output: str = "dense") -> np.ndarray:
            assert output == "dense"
            return self.data.copy()

    upper = FakeConnectivity(np.array([[0.0, 2.0], [0.0, 0.0]]))
    lower = FakeConnectivity(np.array([[0.0, 0.0], [3.0, 0.0]]))
    np.testing.assert_allclose(prepare_connectome(upper).matrix, [[0, 2], [2, 0]])
    np.testing.assert_allclose(prepare_connectome(lower).matrix, [[0, 3], [3, 0]])

    mixed = FakeConnectivity(np.array([[0.0, 2.0], [1.0, 0.0]]))
    assert prepare_connectome(mixed).directed is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"complex_mode": "phase"}, "complex_mode"),
        ({"symmetrize": "guess"}, "symmetrize"),
        ({"directed": "yes"}, "directed"),
        ({"keep_sign": "zero"}, "keep_sign"),
        ({"min_distance_mm": True}, "min_distance_mm"),
        ({"min_distance_mm": -1}, "min_distance_mm"),
        ({"strength_mode": "net"}, "strength_mode"),
    ],
)
def test_invalid_scientific_options_are_rejected(kwargs, message: str) -> None:
    values = np.array([[0.0, 1.0], [1.0, 0.0]])
    if "min_distance_mm" in kwargs:
        kwargs["geometry"] = geometry(np.array([[0, 0, 0], [1, 0, 0]], float))
    with pytest.raises(ValueError, match=message):
        prepare_connectome(values, **kwargs)


def test_shape_and_input_specific_option_errors() -> None:
    geom = geometry(np.array([[0, 0, 0], [1, 0, 0]], float))
    with pytest.raises(ConnectivityShapeError, match="geometry has"):
        prepare_connectome(np.zeros((3, 3)), geometry=geom)
    with pytest.raises(ValueError, match=r"edge_mask.*shape"):
        prepare_connectome(np.zeros((2, 2)), edge_mask=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="selectors require"):
        prepare_connectome(np.zeros((2, 2)), freq=10)
