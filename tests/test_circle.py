from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyconnviz.connectivity import prepare_connectome
from pyconnviz.geometry import geometry_from_arrays
from pyconnviz.models import HemisphereMesh
from pyconnviz.plotting import circle
from pyconnviz.plotting.circle import plot_circle_connectome


def geometry():
    mesh = HemisphereMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
        np.array([[0, 1, 2]], int),
    )
    return geometry_from_arrays(
        ("L", "R", "R2"),
        np.zeros((3, 3)),
        ("left", "right", "right"),
        {"left": mesh, "right": mesh},
        groups=("A", "A", "B"),
    )


class FakeFigure:
    def savefig(self, path, **kwargs) -> None:
        Path(path).write_bytes(b"circle")


def test_circle_forwards_exact_matrix_names_and_disables_second_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    fake_figure = FakeFigure()
    fake_axis = object()

    def fake_circle(matrix, names, **kwargs):
        calls.append({"matrix": matrix.copy(), "names": list(names), **kwargs})
        return fake_figure, fake_axis

    monkeypatch.setattr(circle, "_get_circle_function", lambda: fake_circle)
    prepared = prepare_connectome(
        np.array([[0, 1, 0], [0, 0, -2], [3, 0, 0]], float),
        geometry=geometry(),
        directed=True,
    )
    output = tmp_path / "circle.png"
    result = plot_circle_connectome(prepared, geometry(), output=output, show=False)
    np.testing.assert_array_equal(calls[0]["matrix"], prepared.visible_matrix)
    assert calls[0]["names"] == ["L", "R", "R2"]
    assert calls[0]["n_lines"] is None
    assert calls[0]["show"] is False
    assert result.artist == (fake_figure, fake_axis)
    assert result.prepared is prepared
    assert output.stat().st_size > 0


def test_explicit_circle_order_reorders_matrix_and_names_only_for_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_circle(matrix, names, **kwargs):
        calls.append({"matrix": matrix.copy(), "names": list(names), **kwargs})
        return FakeFigure(), object()

    monkeypatch.setattr(circle, "_get_circle_function", lambda: fake_circle)
    prepared = prepare_connectome(
        np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], float), geometry=geometry()
    )
    result = plot_circle_connectome(prepared, geometry(), node_order=(2, 0, 1))
    assert calls[0]["names"] == ["R2", "L", "R"]
    np.testing.assert_array_equal(
        calls[0]["matrix"], prepared.visible_matrix[np.ix_([2, 0, 1], [2, 0, 1])]
    )
    assert result.prepared is prepared


@pytest.mark.parametrize("bad_order", [(0, 0, 1), (0, 1), (0, 1, 3)])
def test_circle_order_must_be_a_permutation(bad_order) -> None:
    prepared = prepare_connectome(np.zeros((3, 3)), geometry=geometry())
    with pytest.raises(ValueError, match="permutation"):
        plot_circle_connectome(prepared, geometry(), node_order=bad_order)
