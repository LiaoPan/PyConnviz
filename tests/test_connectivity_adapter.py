from __future__ import annotations

import numpy as np
import pytest

from pyconnviz.connectivity import prepare_connectome
from pyconnviz.models import ConnectivityShapeError


class FakeConnectivity:
    def __init__(
        self,
        data: np.ndarray,
        *,
        dims: tuple[str, ...],
        coords: dict[str, np.ndarray] | None = None,
        names: tuple[str, ...] = ("A", "B"),
        symmetric: bool = False,
    ) -> None:
        self._data = np.asarray(data)
        self.dims = dims
        self.coords = coords or {}
        self.names = names
        self.n_nodes = len(names)
        self.attrs = {"symmetric": symmetric}

    def get_data(self, output: str = "dense") -> np.ndarray:
        assert output == "dense"
        return self._data.copy()


def test_two_dimensional_numpy_is_copied_and_named() -> None:
    original = np.array([[9.0, 0.3], [0.1, 8.0]])
    prepared = prepare_connectome(original)
    original[:] = 0
    assert prepared.matrix[0, 1] == pytest.approx(0.3)
    assert prepared.node_names == ("0", "1")
    assert prepared.directed is True
    assert prepared.metadata["source_type"] == "numpy"


def test_numpy_matrix_must_be_square() -> None:
    with pytest.raises(ConnectivityShapeError, match=r"square.*\(2, 3\)"):
        prepare_connectome(np.zeros((2, 3)))


def test_high_dimensional_numpy_requires_explicit_axes_and_reduces() -> None:
    values = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 3.0], [3.0, 0.0]]),
        ],
        axis=0,
    )
    with pytest.raises(ConnectivityShapeError, match="node_axes"):
        prepare_connectome(values)
    with pytest.raises(ConnectivityShapeError, match="reduce_axes"):
        prepare_connectome(values, node_axes=(1, 2))

    mean = prepare_connectome(
        values,
        node_axes=(1, 2),
        reduce_axes=(0,),
        reduction="mean",
    )
    median = prepare_connectome(
        values,
        node_axes=(1, 2),
        reduce_axes=(0,),
        reduction="median",
    )
    assert mean.matrix[0, 1] == pytest.approx(2.0)
    assert median.matrix[0, 1] == pytest.approx(2.0)


def test_high_dimensional_axes_must_cover_every_non_node_axis() -> None:
    values = np.zeros((2, 2, 3, 4))
    with pytest.raises(ConnectivityShapeError, match="all non-node axes"):
        prepare_connectome(values, node_axes=(0, 1), reduce_axes=(2,))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("magnitude", np.sqrt(2.0)),
        ("real", 1.0),
        ("imag", 1.0),
    ],
)
def test_complex_modes_are_explicit(mode: str, expected: float) -> None:
    values = np.array([[0, 1 + 1j], [1 - 1j, 0]])
    result = prepare_connectome(values, complex_mode=mode)
    assert result.matrix[0, 1] == pytest.approx(expected)
    assert result.metadata["complex_mode"] == mode


def test_complex_defaults_to_error() -> None:
    with pytest.raises(ValueError, match="complex_mode"):
        prepare_connectome(np.array([[0, 1j], [-1j, 0]]))


def test_fake_mne_frequency_range_uses_public_dense_data() -> None:
    data = np.zeros((2, 2, 3))
    data[0, 1] = [1.0, 3.0, 9.0]
    data[1, 0] = [1.0, 3.0, 9.0]
    con = FakeConnectivity(
        data,
        dims=("node_in", "node_out", "freqs"),
        coords={"freqs": np.array([8.0, 10.0, 20.0])},
        symmetric=True,
    )
    result = prepare_connectome(con, freq=(8.0, 12.0), reduction="mean")
    assert result.matrix[0, 1] == pytest.approx(2.0)
    assert result.node_names == ("A", "B")
    assert result.metadata["freq"] == (8.0, 12.0)
    assert result.metadata["reduction"] == "mean"


def test_fake_mne_requires_selector_for_non_singleton_extra_dimension() -> None:
    con = FakeConnectivity(
        np.zeros((2, 2, 2)),
        dims=("node_in", "node_out", "times"),
        coords={"times": np.array([0.1, 0.2])},
    )
    with pytest.raises(ConnectivityShapeError, match="time selector"):
        prepare_connectome(con)


def test_fake_mne_scalar_time_and_epoch_selectors() -> None:
    data = np.zeros((2, 2, 2, 3))
    data[0, 1, :, :] = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    data[1, 0] = data[0, 1]
    con = FakeConnectivity(
        data,
        dims=("node_in", "node_out", "epochs", "times"),
        coords={"epochs": np.array([0, 1]), "times": np.array([0.1, 0.2, 0.3])},
        symmetric=True,
    )
    result = prepare_connectome(con, epoch=1, time=0.21)
    assert result.matrix[0, 1] == pytest.approx(20.0)


def test_multivariate_component_dimension_is_not_silently_reduced() -> None:
    con = FakeConnectivity(
        np.zeros((2, 2, 3)),
        dims=("node_in", "node_out", "components"),
    )
    with pytest.raises(ConnectivityShapeError, match=r"multivariate.*component"):
        prepare_connectome(con)


def test_invalid_reduction_and_selector_range_are_clear() -> None:
    with pytest.raises(ValueError, match=r"mean.*median"):
        prepare_connectome(
            np.zeros((2, 2, 2)),
            node_axes=(0, 1),
            reduce_axes=(2,),
            reduction="sum",
        )
    con = FakeConnectivity(
        np.zeros((2, 2, 2)),
        dims=("node_in", "node_out", "freqs"),
        coords={"freqs": np.array([8.0, 10.0])},
    )
    with pytest.raises(ValueError, match="does not select any"):
        prepare_connectome(con, freq=(20.0, 30.0))


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        (np.zeros(2), {}, "at least two"),
        (np.zeros((2, 2)), {"node_axes": (0, 1)}, "only valid"),
        (
            np.zeros((2, 2, 3)),
            {"node_axes": (0,), "reduce_axes": (1, 2)},
            "exactly two",
        ),
        (
            np.zeros((2, 2, 3)),
            {"node_axes": (0, 0), "reduce_axes": (1, 2)},
            "distinct",
        ),
        (
            np.zeros((2, 2, 3)),
            {"node_axes": (0, "x"), "reduce_axes": (1, 2)},
            "integer axes",
        ),
        (
            np.zeros((2, 2, 3)),
            {"node_axes": (0, 9), "reduce_axes": (1, 2)},
            "out of range",
        ),
        (
            np.zeros((2, 3, 4)),
            {"node_axes": (0, 1), "reduce_axes": (2,)},
            "square",
        ),
    ],
)
def test_numpy_axis_validation_branches(values, kwargs, message: str) -> None:
    with pytest.raises(ConnectivityShapeError, match=message):
        prepare_connectome(values, **kwargs)


def test_negative_numpy_axes_are_normalized() -> None:
    values = np.stack((np.eye(2), np.eye(2) * 3), axis=0)
    result = prepare_connectome(
        values, node_axes=(-2, -1), reduce_axes=(-3,), reduction="mean"
    )
    np.testing.assert_allclose(result.matrix, np.eye(2) * 2)


def test_mne_dimension_fallback_singletons_and_selector_errors() -> None:
    fallback = FakeConnectivity(
        np.ones((2, 2, 2)),
        dims=("connections", "mystery"),
        coords={"times": np.array([0.0, 1.0])},
    )
    assert prepare_connectome(fallback, time=0.1).matrix.shape == (2, 2)

    singleton = FakeConnectivity(
        np.ones((2, 2, 1, 1)),
        dims=("node_in", "node_out", "freqs", "components"),
        coords={"freqs": np.array([10.0])},
    )
    assert prepare_connectome(singleton).matrix.shape == (2, 2)

    unknown = FakeConnectivity(
        np.ones((2, 2, 2)), dims=("node_in", "node_out", "mystery")
    )
    with pytest.raises(ConnectivityShapeError, match="Could not map"):
        prepare_connectome(unknown)

    wrong_coord = FakeConnectivity(
        np.ones((2, 2, 2)),
        dims=("node_in", "node_out", "times"),
        coords={"times": np.array([0.0])},
    )
    with pytest.raises(ConnectivityShapeError, match="coordinate length"):
        prepare_connectome(wrong_coord, time=0.0)


def test_selector_validation_for_ranges_epochs_and_categorical_coords() -> None:
    con = FakeConnectivity(
        np.ones((2, 2, 2)),
        dims=("node_in", "node_out", "freqs"),
        coords={"freqs": np.array([8.0, 10.0])},
    )
    with pytest.raises(ValueError, match="two-value"):
        prepare_connectome(con, freq=(8.0, 9.0, 10.0))
    with pytest.raises(ValueError, match="lower bound"):
        prepare_connectome(con, freq=(10.0, 8.0))

    epoch_con = FakeConnectivity(
        np.ones((2, 2, 2)),
        dims=("node_in", "node_out", "epochs"),
        coords={"epochs": np.array(["first", "second"])},
    )
    assert prepare_connectome(epoch_con, epoch=-1).matrix.shape == (2, 2)
    with pytest.raises(ValueError, match="out of range"):
        prepare_connectome(epoch_con, epoch=4)
    with pytest.raises(ValueError, match="not available"):
        prepare_connectome(epoch_con, epoch="third")


def test_mne_public_contract_validation_and_real_spectral_object() -> None:
    invalid_nodes = FakeConnectivity(np.zeros((2, 2)), dims=("node_in", "node_out"))
    invalid_nodes.n_nodes = 0
    with pytest.raises(ConnectivityShapeError, match="positive public n_nodes"):
        prepare_connectome(invalid_nodes)

    invalid_dense = FakeConnectivity(np.zeros((3, 3)), dims=("node_in", "node_out"))
    with pytest.raises(ConnectivityShapeError, match="dense data"):
        prepare_connectome(invalid_dense)

    invalid_names = FakeConnectivity(
        np.zeros((2, 2)), dims=("node_in", "node_out"), names=("A", "B")
    )
    invalid_names.names = ("only-one",)
    with pytest.raises(ConnectivityShapeError, match="names has length"):
        prepare_connectome(invalid_names)

    from mne_connectivity import SpectralConnectivity

    real = SpectralConnectivity(
        np.arange(12.0).reshape(4, 3),
        freqs=np.array([8.0, 10.0, 20.0]),
        n_nodes=2,
        names=["A", "B"],
    )
    result = prepare_connectome(real, freq=(8.0, 10.0))
    np.testing.assert_allclose(result.matrix, [[0.5, 3.5], [6.5, 9.5]])


def _real_all_to_all_spectral(method: str):
    from mne_connectivity import spectral_connectivity_epochs

    rng = np.random.default_rng(20260901)
    epochs = rng.standard_normal((8, 4, 256))
    return spectral_connectivity_epochs(
        epochs,
        names=["A", "B", "C", "D"],
        method=method,
        mode="multitaper",
        sfreq=128.0,
        fmin=8.0,
        fmax=13.0,
        faverage=True,
        n_jobs=1,
        verbose=False,
    )


@pytest.mark.parametrize("method", ["coh", "plv"])
def test_real_mne_symmetric_all_to_all_metric_recovers_dense_triangle(method: str) -> None:
    connectivity = _real_all_to_all_spectral(method)
    raw = np.asarray(connectivity.get_data(output="dense"), dtype=np.float64)[:, :, 0]
    lower = np.tril(raw, -1)
    expected = lower + lower.T + np.diag(np.diag(raw))

    prepared = prepare_connectome(connectivity)

    assert prepared.directed is False
    np.testing.assert_allclose(prepared.matrix, expected, rtol=0.0, atol=1e-12)
    assert prepared.metadata["symmetry_basis"] == f"mne:{method}:all-to-all"


def test_real_mne_directed_metric_is_not_auto_mirrored() -> None:
    connectivity = _real_all_to_all_spectral("dpli")
    raw = np.asarray(connectivity.get_data(output="dense"), dtype=np.float64)[:, :, 0]

    prepared = prepare_connectome(connectivity)

    assert prepared.directed is True
    np.testing.assert_array_equal(prepared.matrix, raw)
    assert prepared.metadata["symmetry_basis"] is None
