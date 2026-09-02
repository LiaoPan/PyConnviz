from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyconnviz.connectivity import prepare_connectome
from pyconnviz.geometry import geometry_from_arrays
from pyconnviz.models import CoordinateSpaceError, HemisphereMesh
from pyconnviz.plotting import nilearn_connectome
from pyconnviz.plotting.nilearn_connectome import (
    plot_glass_connectome,
    plot_html_connectome,
)


def mesh() -> HemisphereMesh:
    return HemisphereMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
        np.array([[0, 1, 2]], int),
    )


def geometry(*, with_mni: bool = True):
    return geometry_from_arrays(
        ("A", "B"),
        np.array([[100, 100, 100], [200, 200, 200]], float),
        ("left", "right"),
        {"left": mesh(), "right": mesh()},
        mni_coords=np.array([[-20, 0, 40], [20, 0, 40]], float) if with_mni else None,
    )


def directed_network():
    return prepare_connectome(
        np.array([[0.0, 0.8], [-0.2, 0.0]]), geometry=geometry(), directed=True
    )


class FakeDisplay:
    pass


class FakeHtmlView:
    def save_as_html(self, path) -> None:
        Path(path).write_text("<html><body>connectome</body></html>", encoding="utf-8")


def test_glass_forwards_exact_visible_matrix_mni_and_no_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_plot_connectome(matrix, coords, **kwargs):
        calls.append({"matrix": matrix.copy(), "coords": coords.copy(), **kwargs})
        Path(kwargs["output_file"]).write_bytes(b"svg")
        return FakeDisplay()

    monkeypatch.setattr(
        nilearn_connectome,
        "_get_nilearn_plotting",
        lambda: SimpleNamespace(plot_connectome=fake_plot_connectome),
    )
    output = tmp_path / "glass.svg"
    prepared = directed_network()
    result = plot_glass_connectome(prepared, geometry(), output=output, show=False)
    np.testing.assert_array_equal(calls[0]["matrix"], prepared.visible_matrix)
    np.testing.assert_array_equal(calls[0]["coords"], geometry().mni_coords)
    assert not np.array_equal(calls[0]["coords"], geometry().surface_coords)
    assert calls[0]["edge_threshold"] is None
    assert calls[0]["display_mode"] == "lyrz"
    assert result.prepared is prepared
    assert result.output_files == (output,)


def test_html_forwards_exact_matrix_labels_and_mni_without_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def fake_view_connectome(matrix, coords, **kwargs):
        calls.append({"matrix": matrix.copy(), "coords": coords.copy(), **kwargs})
        return FakeHtmlView()

    monkeypatch.setattr(
        nilearn_connectome,
        "_get_nilearn_plotting",
        lambda: SimpleNamespace(view_connectome=fake_view_connectome),
    )
    output = tmp_path / "connectome.html"
    prepared = directed_network()
    result = plot_html_connectome(prepared, geometry(), output=output)
    np.testing.assert_array_equal(calls[0]["matrix"], prepared.visible_matrix)
    np.testing.assert_array_equal(calls[0]["coords"], geometry().mni_coords)
    assert calls[0]["edge_threshold"] is None
    assert calls[0]["node_labels"] == ["A", "B"]
    assert output.stat().st_size > 0
    assert result.artist.__class__ is FakeHtmlView
    assert result.prepared is prepared


@pytest.mark.parametrize("backend", [plot_glass_connectome, plot_html_connectome])
def test_nilearn_connectomes_fail_before_render_without_mni(backend) -> None:
    geom = geometry(with_mni=False)
    prepared = prepare_connectome(np.array([[0, 1], [1, 0]], float), geometry=geom)
    with pytest.raises(CoordinateSpaceError, match="mni_coords"):
        backend(prepared, geom)


def test_glass_rejects_html_extension_and_html_rejects_static_extension(tmp_path: Path) -> None:
    prepared = directed_network()
    with pytest.raises(ValueError, match=r"PNG.*SVG.*PDF"):
        plot_glass_connectome(prepared, geometry(), output=tmp_path / "bad.html")
    with pytest.raises(ValueError, match="HTML"):
        plot_html_connectome(prepared, geometry(), output=tmp_path / "bad.png")
