from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyconnviz._compat import get_public_plotly_figure
from pyconnviz.connectivity import prepare_connectome
from pyconnviz.geometry import geometry_from_arrays
from pyconnviz.models import HemisphereMesh, OptionalDependencyError
from pyconnviz.plotting import surface_plotly
from pyconnviz.plotting.surface_plotly import plot_surface_plotly, warn_if_many_edges


def tetra_mesh(center_x: float) -> HemisphereMesh:
    coordinates = np.array(
        [
            [center_x - 1, -1, -1],
            [center_x + 1, -1, 1],
            [center_x - 1, 1, 1],
            [center_x + 1, 1, -1],
        ],
        float,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], int)
    return HemisphereMesh(coordinates, faces, np.linspace(-1, 1, 4))


def geometry():
    return geometry_from_arrays(
        ("L0", "L1", "R0", "R1"),
        np.array([[-4, -1, -1], [-2, 1, -1], [2, -1, -1], [4, 1, -1]], float),
        ("left", "left", "right", "right"),
        {"left": tetra_mesh(-3), "right": tetra_mesh(3)},
        groups=("A", "B", "A", "B"),
        node_vertices=np.array([0, 3, 0, 3]),
    )


def network(*, directed: bool = False):
    if directed:
        matrix = np.array(
            [[0, 0.5, 0, 0], [0, 0, -0.8, 0], [0.2, 0, 0, 1], [0, 0, 0, 0]],
            float,
        )
        return prepare_connectome(matrix, geometry=geometry(), directed=True)
    matrix = np.array(
        [[0, 0.5, -0.8, 0], [0.5, 0, 0.6, 0], [-0.8, 0.6, 0, 1], [0, 0, 1, 0]],
        float,
    )
    return prepare_connectome(matrix, geometry=geometry())


def test_public_plotly_figure_access_uses_figure_attribute() -> None:
    sentinel = object()
    assert get_public_plotly_figure(SimpleNamespace(figure=sentinel)) is sentinel
    with pytest.raises(TypeError, match=r"public.*figure"):
        get_public_plotly_figure(SimpleNamespace(_figure=sentinel))


def test_missing_plotly_has_actionable_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_find_spec = surface_plotly.importlib.util.find_spec
    monkeypatch.setattr(
        surface_plotly.importlib.util,
        "find_spec",
        lambda name: None if name == "plotly" else real_find_spec(name),
    )
    with pytest.raises(OptionalDependencyError, match=r"pyconnviz\[interactive\]"):
        plot_surface_plotly(network(), geometry())


def test_warning_does_not_change_large_edge_count() -> None:
    with pytest.warns(RuntimeWarning, match="201"):
        count = warn_if_many_edges(201)
    assert count == 201


def test_plotly_nodes_share_nilearn_bilateral_mesh_coordinates() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = plot_surface_plotly(
            network(),
            geometry(),
            node_overlay="none",
            node_offset_mm=0.0,
            show=False,
        )

    surface = result.artist.data[0]
    nodes = next(trace for trace in result.artist.data if trace.name == "Nodes")
    node_vertices = (0, 3, 4, 7)
    np.testing.assert_allclose(nodes.x, np.asarray(surface.x)[list(node_vertices)])
    np.testing.assert_allclose(nodes.y, np.asarray(surface.y)[list(node_vertices)])
    np.testing.assert_allclose(nodes.z, np.asarray(surface.z)[list(node_vertices)])
    np.testing.assert_allclose(nodes.x, [-4.0, -2.0, -1.0, 1.0])
    assert not any("threshold" in str(item.message).lower() for item in caught)


def test_plotly_surface_uses_style_and_explicit_cortex_alpha() -> None:
    default = plot_surface_plotly(
        network(), geometry(), node_overlay="none", show=False
    )
    overridden = plot_surface_plotly(
        network(),
        geometry(),
        node_overlay="none",
        cortex_alpha=0.72,
        show=False,
    )

    assert default.artist.data[0].type == "mesh3d"
    assert default.artist.data[0].opacity == pytest.approx(0.35)
    assert overridden.artist.data[0].opacity == pytest.approx(0.72)


def test_plotly_surface_opacity_requires_nilearn_mesh_trace() -> None:
    with pytest.raises(RuntimeError, match="Mesh3d"):
        surface_plotly._set_surface_opacity(SimpleNamespace(data=[]), 0.4)


def test_plotly_surface_cmap_is_independent_from_node_cmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    real_build = surface_plotly._build_base_surface

    def capture_build(*args, **kwargs):
        calls.append(kwargs)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(surface_plotly, "_build_base_surface", capture_build)
    result = plot_surface_plotly(
        network(),
        geometry(),
        node_overlay="gaussian",
        node_cmap="viridis",
        surface_cmap="RdBu_r",
        overlay_sigma_mm=2.0,
        overlay_radius_mm=4.0,
        show=False,
    )

    assert calls[0]["surface_cmap"] == "RdBu_r"
    nodes = next(trace for trace in result.artist.data if trace.name == "Nodes")
    assert nodes.marker.colorscale != result.artist.data[0].colorscale


def test_plotly_traces_hover_direction_and_offline_html(tmp_path: Path) -> None:
    output = tmp_path / "interactive.html"
    prepared = network(directed=True)
    result = plot_surface_plotly(
        prepared,
        geometry(),
        node_overlay="gaussian",
        overlay_sigma_mm=2.0,
        overlay_radius_mm=4.0,
        output=output,
        include_plotlyjs=True,
        show=False,
    )
    figure = result.artist
    assert result.prepared is prepared
    assert result.output_files == (output,)
    assert output.stat().st_size > 1000
    node_traces = [trace for trace in figure.data if trace.name == "Nodes"]
    edge_traces = [trace for trace in figure.data if str(trace.name).startswith("Edge ")]
    assert len(node_traces) == 1
    assert len(node_traces[0].x) == 4
    assert len(edge_traces) == len(prepared.edges)
    assert "L0" in node_traces[0].hovertext[0]
    assert "group=A" in node_traces[0].hovertext[0]
    expected = [
        f"{prepared.node_names[edge.source]} → {prepared.node_names[edge.target]}"
        for edge in prepared.edges
    ]
    assert [trace.name.removeprefix("Edge ") for trace in edge_traces] == expected
    html = output.read_text(encoding="utf-8")
    assert "plotly.js" in html.lower()
    assert "L0" in html
    assert "weight=" in html
    assert "pyconnviz_version" in html


def test_show_false_never_calls_plotly_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plotly.graph_objects as go

    calls: list[bool] = []
    monkeypatch.setattr(go.Figure, "show", lambda self: calls.append(True))
    plot_surface_plotly(
        network(), geometry(), output=tmp_path / "no-show.html", show=False
    )
    assert calls == []


def test_style_does_not_change_interactive_edge_set(tmp_path: Path) -> None:
    prepared = network()
    before = prepared.edges
    result = plot_surface_plotly(
        prepared,
        geometry(),
        style="dark",
        output=tmp_path / "dark.html",
        show=False,
    )
    assert result.prepared.edges == before


def test_static_export_checks_kaleido_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_find_spec = surface_plotly.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "kaleido":
            return None
        return real_find_spec(name)

    monkeypatch.setattr(surface_plotly.importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(OptionalDependencyError, match=r"pyconnviz\[export\]"):
        plot_surface_plotly(
            network(), geometry(), output=tmp_path / "static.png", show=False
        )


def test_static_export_forwards_explicit_pixel_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plotly.graph_objects as go

    calls: list[tuple[Path, dict[str, object]]] = []

    def capture_write_image(self, path, **kwargs):
        destination = Path(path)
        destination.write_bytes(b"static-image")
        calls.append((destination, kwargs))

    monkeypatch.setattr(go.Figure, "write_image", capture_write_image)
    output = tmp_path / "static.png"

    result = plot_surface_plotly(
        network(),
        geometry(),
        output=output,
        image_width=1200,
        image_height=900,
        image_scale=1.5,
        show=False,
    )

    assert result.output_files == (output,)
    assert calls == [
        (
            output,
            {"width": 1200, "height": 900, "scale": pytest.approx(1.5)},
        )
    ]


def test_static_views_require_a_non_empty_sequence_not_a_string() -> None:
    with pytest.raises(TypeError, match="sequence"):
        plot_surface_plotly(network(), geometry(), static_views="left")
    with pytest.raises(ValueError, match="must not be empty"):
        plot_surface_plotly(network(), geometry(), static_views=())


def test_static_views_none_keeps_the_single_interactive_figure_for_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plotly.graph_objects as go

    exported: list[go.Figure] = []

    def capture_write_image(self, path, **kwargs):
        exported.append(self)
        Path(path).write_bytes(b"static-image")

    monkeypatch.setattr(go.Figure, "write_image", capture_write_image)
    result = plot_surface_plotly(
        network(),
        geometry(),
        node_overlay="none",
        static_views=None,
        output=tmp_path / "single.png",
        show=False,
    )

    assert exported == [result.artist]


def test_static_montage_has_four_native_cameras_and_complete_trace_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plotly.graph_objects as go

    exported: list[go.Figure] = []

    def capture_write_image(self, path, **kwargs):
        exported.append(self)
        Path(path).write_bytes(b"static-image")

    monkeypatch.setattr(go.Figure, "write_image", capture_write_image)
    views = ("left", "right", "dorsal", "ventral")
    result = plot_surface_plotly(
        network(),
        geometry(),
        node_overlay="none",
        static_views=views,
        output=tmp_path / "montage.png",
        show=False,
    )

    assert len(exported) == 1
    montage = exported[0]
    assert montage is not result.artist
    assert len(montage.data) == len(views) * len(result.artist.data)
    layout = montage.layout.to_plotly_json()
    scene_names = ("scene", "scene2", "scene3", "scene4")
    assert all(name in layout for name in scene_names)
    assert "scene2" not in result.artist.layout.to_plotly_json()
    assert [layout[name]["camera"]["eye"] for name in scene_names] == [
        {"x": -1.5, "y": 0, "z": 0},
        {"x": 1.5, "y": 0, "z": 0},
        {"x": 0, "y": 0, "z": 1.5},
        {"x": 0, "y": 0, "z": -1.5},
    ]
    assert [annotation.text for annotation in montage.layout.annotations] == [
        "Left",
        "Right",
        "Dorsal",
        "Ventral",
    ]
    for name in scene_names:
        assert sum(trace.scene == name for trace in montage.data) == len(
            result.artist.data
        )
    node_traces = [trace for trace in montage.data if trace.name == "Nodes"]
    assert [trace.marker.showscale for trace in node_traces] == [
        True,
        False,
        False,
        False,
    ]
    assert layout["meta"]["static_views"] == list(views)
