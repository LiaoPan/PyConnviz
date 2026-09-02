from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pyconnviz
from pyconnviz import (
    ConnectomeGeometry,
    HemisphereMesh,
    PlotResult,
    PreparedConnectome,
    api,
    geometry_from_arrays,
    plot_connectome,
    prepare_connectome,
)


def geometry() -> ConnectomeGeometry:
    mesh = HemisphereMesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float),
        np.array([[0, 1, 2]], int),
    )
    return geometry_from_arrays(
        ("L", "R"),
        np.array([[0, 0, 0], [2, 0, 0]], float),
        ("left", "right"),
        {"left": mesh, "right": mesh},
        mni_coords=np.array([[-20, 0, 40], [20, 0, 40]], float),
        groups=("visual", "auditory"),
        node_vertices=np.array([0, 0]),
    )


def prepared() -> PreparedConnectome:
    return prepare_connectome(np.array([[0, 1], [1, 0]], float), geometry=geometry())


def fake_backend(name: str, calls: list[tuple]):
    def render(network, geom, **kwargs):
        calls.append((name, network, geom, kwargs))
        return PlotResult(name, kwargs.get("engine"), object(), network)

    return render


@pytest.mark.parametrize(
    ("backend", "engine", "loader_name", "expected"),
    [
        ("surface", "matplotlib", "_load_surface_matplotlib", "surface-mpl"),
        ("surface", "nilearn", "_load_surface_nilearn", "surface-nilearn"),
        ("surface", "plotly", "_load_surface_plotly", "surface-plotly"),
        ("glass", "matplotlib", "_load_glass", "glass"),
        ("html", "matplotlib", "_load_html", "html"),
        ("circle", "matplotlib", "_load_circle", "circle"),
    ],
)
def test_dispatches_every_backend(
    backend: str,
    engine: str,
    loader_name: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(api, loader_name, lambda: fake_backend(expected, calls))
    result = plot_connectome(prepared(), geometry(), backend=backend, engine=engine)
    assert calls[0][0] == expected
    assert result.prepared is calls[0][1]


def test_raw_connectivity_is_prepared_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_calls: list[dict] = []
    real_prepare = api.prepare_connectome

    def counted_prepare(connectivity, **kwargs):
        prepare_calls.append(kwargs)
        return real_prepare(connectivity, **kwargs)

    backend_calls: list[tuple] = []
    monkeypatch.setattr(api, "prepare_connectome", counted_prepare)
    monkeypatch.setattr(
        api, "_load_surface_matplotlib", lambda: fake_backend("surface", backend_calls)
    )
    matrix = np.array([[0, 0.4], [0.4, 0]], float)
    result = plot_connectome(
        matrix,
        geometry(),
        edge_threshold=0.3,
        max_edges=1,
        node_overlay="none",
    )
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["edge_threshold"] == 0.3
    assert prepare_calls[0]["max_edges"] == 1
    assert result.prepared.edges[0].weight == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("engine", "loader_name"),
    [
        ("matplotlib", "_load_surface_matplotlib"),
        ("plotly", "_load_surface_plotly"),
    ],
)
def test_surface_cmap_routes_only_to_surface_engines(
    engine: str, loader_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(api, loader_name, lambda: fake_backend("surface", calls))

    plot_connectome(
        prepared(),
        geometry(),
        backend="surface",
        engine=engine,
        surface_cmap="RdBu_r",
    )

    assert calls[0][3]["surface_cmap"] == "RdBu_r"
    with pytest.raises(TypeError, match="surface_cmap"):
        plot_connectome(
            prepared(), geometry(), backend="glass", surface_cmap="RdBu_r"
        )


def test_cortex_alpha_routes_to_matplotlib_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        api, "_load_surface_matplotlib", lambda: fake_backend("surface", calls)
    )

    plot_connectome(prepared(), geometry(), cortex_alpha=0.7)

    assert calls[0][3]["cortex_alpha"] == pytest.approx(0.7)


def test_cortex_alpha_routes_to_plotly_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(api, "_load_surface_plotly", lambda: fake_backend("surface", calls))

    plot_connectome(
        prepared(), geometry(), engine="plotly", cortex_alpha=0.7
    )

    assert calls[0][3]["cortex_alpha"] == pytest.approx(0.7)


def test_depth_cue_routes_only_to_static_surface_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib_calls: list[tuple] = []
    monkeypatch.setattr(
        api,
        "_load_surface_matplotlib",
        lambda: fake_backend("surface", matplotlib_calls),
    )

    plot_connectome(prepared(), geometry(), depth_cue=False)

    assert matplotlib_calls[0][3]["depth_cue"] is False
    nilearn_calls: list[tuple] = []
    monkeypatch.setattr(
        api,
        "_load_surface_nilearn",
        lambda: fake_backend("surface-nilearn", nilearn_calls),
    )
    plot_connectome(
        prepared(),
        geometry(),
        engine="nilearn",
        depth_cue=False,
    )
    assert nilearn_calls[0][3]["depth_cue"] is False
    with pytest.raises(TypeError, match="depth_cue"):
        plot_connectome(prepared(), geometry(), engine="plotly", depth_cue=False)
    with pytest.raises(TypeError, match="depth_cue"):
        plot_connectome(prepared(), geometry(), backend="glass", depth_cue=False)


def test_static_image_dimensions_route_only_to_plotly_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(api, "_load_surface_plotly", lambda: fake_backend("surface", calls))

    plot_connectome(
        prepared(),
        geometry(),
        engine="plotly",
        image_width=1200,
        image_height=900,
        image_scale=1.5,
    )

    assert calls[0][3]["image_width"] == 1200
    assert calls[0][3]["image_height"] == 900
    assert calls[0][3]["image_scale"] == pytest.approx(1.5)
    with pytest.raises(TypeError, match="image_width"):
        plot_connectome(prepared(), geometry(), engine="matplotlib", image_width=1200)


def test_static_views_route_only_to_plotly_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(api, "_load_surface_plotly", lambda: fake_backend("surface", calls))
    views = ("left", "right", "dorsal", "ventral")

    plot_connectome(
        prepared(),
        geometry(),
        engine="plotly",
        static_views=views,
    )

    assert calls[0][3]["static_views"] == views
    with pytest.raises(TypeError, match="static_views"):
        plot_connectome(
            prepared(),
            geometry(),
            engine="matplotlib",
            static_views=views,
        )


def test_native_nilearn_options_route_only_to_the_native_surface_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        api,
        "_load_surface_nilearn",
        lambda: fake_backend("surface-nilearn", calls),
    )
    stat_map = object()
    surface_path = Path("data/fsaverage")

    plot_connectome(
        prepared(),
        geometry(),
        backend="surface",
        engine="nilearn",
        stat_map=stat_map,
        surf_mesh=surface_path,
        hemispheres=["left", "right"],
        bg_on_data=True,
        symmetric_cmap=None,
        symmetric_cbar="auto",
        inflate=False,
        threshold=1.2,
        surface_vmin=-3.0,
        surface_vmax=3.0,
    )

    options = calls[0][3]
    assert options["stat_map"] is stat_map
    assert options["surf_mesh"] == surface_path
    assert options["hemispheres"] == ["left", "right"]
    assert options["bg_on_data"] is True
    assert options["symmetric_cmap"] is None
    assert options["symmetric_cbar"] == "auto"
    assert options["threshold"] == pytest.approx(1.2)
    assert options["surface_vmin"] == pytest.approx(-3.0)
    assert options["surface_vmax"] == pytest.approx(3.0)

    with pytest.raises(TypeError, match="stat_map"):
        plot_connectome(
            prepared(),
            geometry(),
            backend="surface",
            engine="matplotlib",
            stat_map=stat_map,
        )
    with pytest.raises(TypeError, match="surf_mesh"):
        plot_connectome(
            prepared(),
            geometry(),
            backend="glass",
            surf_mesh=surface_path,
        )


@pytest.mark.parametrize(
    "scientific_option",
    [
        {"edge_threshold": 0.5},
        {"max_edges": 2},
        {"edge_mask": np.ones((2, 2), bool)},
        {"freq": (8, 13)},
        {"directed": False},
    ],
)
def test_prepared_input_rejects_repeated_scientific_options(scientific_option) -> None:
    with pytest.raises(ValueError, match="already prepared"):
        plot_connectome(prepared(), geometry(), **scientific_option)


def test_unknown_and_backend_incompatible_options_fail_immediately() -> None:
    with pytest.raises(TypeError, match="unknown_option"):
        plot_connectome(prepared(), geometry(), unknown_option=True)
    with pytest.raises(TypeError, match="display_mode"):
        plot_connectome(
            prepared(), geometry(), backend="surface", display_mode="lyrz"
        )
    with pytest.raises(ValueError, match="engine"):
        plot_connectome(prepared(), geometry(), backend="surface", engine="vtk")
    with pytest.raises(ValueError, match="backend"):
        plot_connectome(prepared(), geometry(), backend="volume")


@pytest.mark.parametrize(
    ("color_by", "expected"),
    [
        ("hemisphere", np.array([0.0, 1.0])),
        ("group", np.array([0.0, 1.0])),
        ("custom", np.array([3.0, 7.0])),
    ],
)
def test_public_node_encoding_options_are_resolved_for_every_backend(
    color_by: str,
    expected: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(api, "_load_glass", lambda: fake_backend("glass", calls))

    plot_connectome(
        prepared(),
        geometry(),
        backend="glass",
        node_values=np.array([3.0, 7.0]),
        node_color_by=color_by,
        node_size_by="custom",
        edge_cmap="coolwarm",
    )

    options = calls[0][3]
    np.testing.assert_array_equal(options["node_color_values"], expected)
    np.testing.assert_array_equal(options["node_size_values"], [3.0, 7.0])
    assert options["edge_cmap"] == "coolwarm"


def test_group_encoding_requires_geometry_groups() -> None:
    geom = geometry_from_arrays(
        geometry().node_names,
        geometry().surface_coords,
        geometry().hemispheres,
        geometry().meshes,
        mni_coords=geometry().mni_coords,
        node_vertices=geometry().node_vertices,
    )
    with pytest.raises(ValueError, match=r"geometry\.groups"):
        plot_connectome(prepared(), geom, node_color_by="group")


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("node_color_by", "network", "node_color_by"),
        ("node_size_by", "hemisphere", "node_size_by"),
        ("node_size_by", "custom", "node_values"),
    ],
)
def test_invalid_public_node_encoding_options_fail(
    option: str, value: str, message: str
) -> None:
    kwargs = {option: value}
    with pytest.raises(ValueError, match=message):
        plot_connectome(prepared(), geometry(), **kwargs)


def test_public_exports_and_version_are_complete() -> None:
    expected = {
        "ConnectomeGeometry",
        "HemisphereMesh",
        "PlotResult",
        "PreparedConnectome",
        "ViewSpec",
        "geometry_from_arrays",
        "geometry_from_mne_labels",
        "get_style",
        "plot_connectome",
        "prepare_connectome",
    }
    assert expected.issubset(set(pyconnviz.__all__))
    assert pyconnviz.__version__ == "0.1.0"


def test_readme_covers_required_scientific_contracts() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "surface-RAS",
        "MNI",
        "edge_mask",
        "edge_threshold",
        "max_edges",
        "Gaussian",
        "visual interpolation",
        "directed",
        "does not download",
    ):
        assert phrase in readme


def test_top_level_import_does_not_eagerly_load_plotting_backends() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys, pyconnviz; "
        "assert 'matplotlib.pyplot' not in sys.modules; "
        "assert 'plotly' not in sys.modules; "
        "assert 'mne_connectivity' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr
