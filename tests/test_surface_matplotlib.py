from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
    Path3DCollection,
    Poly3DCollection,
)

from pyconnviz.connectivity import prepare_connectome
from pyconnviz.geometry import geometry_from_arrays
from pyconnviz.models import Edge, HemisphereMesh, PreparedConnectome, ViewSpec
from pyconnviz.plotting import _surface_common as surface_common
from pyconnviz.plotting._surface_common import (
    edge_color_norm,
    quadratic_bezier,
    resolve_cortex_alpha,
    resolve_views,
    scale_values,
    select_panel_edges,
    surface_display_coordinates,
)
from pyconnviz.plotting.surface_matplotlib import plot_surface_matplotlib

matplotlib.use("Agg", force=True)


@pytest.mark.parametrize("value", [True, -0.1, 1.1, np.nan, np.inf])
def test_cortex_alpha_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="cortex_alpha"):
        resolve_cortex_alpha(value, default=0.4)


def test_cortex_alpha_resolves_explicit_and_default_values() -> None:
    assert resolve_cortex_alpha(None, default=0.4) == pytest.approx(0.4)
    assert resolve_cortex_alpha(0.75, default=0.4) == pytest.approx(0.75)


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
    return HemisphereMesh(coordinates, faces, np.array([-1.0, -0.3, 0.3, 1.0]))


def geometry(node_count: int = 4):
    if node_count != 4:
        raise ValueError("test geometry has four nodes")
    return geometry_from_arrays(
        ("L0", "L1", "R0", "R1"),
        np.array([[-4, -1, -1], [-2, 1, -1], [2, -1, -1], [4, 1, -1]], float),
        ("left", "left", "right", "right"),
        {"left": tetra_mesh(-3), "right": tetra_mesh(3)},
        mni_coords=np.array([[-30, -10, 20], [-20, 10, 20], [20, -10, 20], [30, 10, 20]]),
        node_vertices=np.array([0, 3, 0, 3]),
    )


def prepared():
    matrix = np.array(
        [[0, 0.4, -0.8, 0], [0.4, 0, 0.6, 0], [-0.8, 0.6, 0, 1.0], [0, 0, 1.0, 0]],
        float,
    )
    return prepare_connectome(matrix, geometry=geometry())


def test_view_presets_and_custom_sequence() -> None:
    assert [(view.hemi, view.view) for view in resolve_views("lateral")] == [
        ("left", "lateral"),
        ("right", "lateral"),
    ]
    assert len(resolve_views("four")) == 4
    assert [view.hemi for view in resolve_views("paper")] == ["left", "right", "both"]
    assert [view.view for view in resolve_views("whole")] == [
        "dorsal",
        "anterior",
        "posterior",
    ]
    assert len(resolve_views("single")) == 1
    custom = (ViewSpec("both", (25.0, 30.0), "Custom"),)
    assert resolve_views(custom) == custom
    with pytest.raises(ValueError, match="views"):
        resolve_views("unknown")


def test_panel_edges_obey_hemisphere_scope() -> None:
    geom = geometry()
    edges = prepared().edges
    left = select_panel_edges(edges, geom, "left")
    right = select_panel_edges(edges, geom, "right")
    both = select_panel_edges(edges, geom, "both")
    assert {(edge.source, edge.target) for edge in left} == {(0, 1)}
    assert {(edge.source, edge.target) for edge in right} == {(2, 3)}
    assert len(both) == len(edges)


def test_bilateral_display_coordinates_match_nilearn_combination() -> None:
    geom = geometry()
    display = surface_display_coordinates(
        geom, "both", node_offset_mm=0.0, center=False
    )

    # Nilearn moves the right part to one millimetre beyond left_max_x.
    assert display.hemi == "both"
    np.testing.assert_allclose(display.node_positions[:, 0], [-4.0, -2.0, -1.0, 1.0])
    np.testing.assert_allclose(display.surface_coordinates[4], [-1.0, -1.0, -1.0])
    np.testing.assert_allclose(display.brain_center, [-1.5, 0.0, 0.0])


def test_matplotlib_display_coordinates_match_nilearn_centering() -> None:
    display = surface_display_coordinates(
        geometry(), "both", node_offset_mm=0.0, center=True
    )

    np.testing.assert_allclose(
        display.node_positions,
        [
            [-2.5, -1.0, -1.0],
            [-0.5, 1.0, -1.0],
            [0.5, -1.0, -1.0],
            [2.5, 1.0, -1.0],
        ],
    )
    np.testing.assert_allclose(np.mean(display.surface_coordinates, axis=0), 0.0)
    np.testing.assert_allclose(display.brain_center, 0.0)


def test_bezier_endpoints_and_outward_midpoint() -> None:
    curve = quadratic_bezier(
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        samples=11,
    )
    np.testing.assert_allclose(curve[0], [0, 0, 0])
    np.testing.assert_allclose(curve[-1], [10, 0, 0])
    assert curve[5, 2] > 0


def test_cross_hemisphere_bezier_bows_dorsally() -> None:
    curve = quadratic_bezier(
        np.array([-4.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        cross_hemisphere=True,
        samples=9,
    )
    assert curve[4, 2] > 0


def test_color_norm_distinguishes_signed_and_one_sided_data() -> None:
    signed_norm, signed_cmap = edge_color_norm(np.array([-2.0, 1.0]))
    positive_norm, positive_cmap = edge_color_norm(np.array([1.0, 2.0]))
    negative_norm, negative_cmap = edge_color_norm(np.array([-2.0, -1.0]))
    assert isinstance(signed_norm, TwoSlopeNorm)
    assert signed_norm.vmin == -2 and signed_norm.vmax == 2
    assert signed_cmap == "RdBu_r"
    assert isinstance(positive_norm, Normalize)
    assert positive_cmap != "RdBu_r"
    assert negative_cmap != "RdBu_r"
    assert negative_norm.vmax == 0


def test_scale_values_handles_empty_and_constant_arrays() -> None:
    np.testing.assert_array_equal(scale_values(np.array([]), (2, 8)), np.array([]))
    np.testing.assert_allclose(scale_values(np.zeros(3), (2, 8)), [5, 5, 5])
    scaled = scale_values(np.array([0.0, np.nan, 2.0]), (10, 20))
    assert np.all(np.isfinite(scaled))
    assert np.all((scaled >= 10) & (scaled <= 20))


def test_projected_depth_factors_are_bounded_and_near_to_far() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 10.0]]
    )

    factors = surface_common.projected_depth_factors(
        points,
        reference,
        np.eye(4),
        minimum=0.2,
    )

    np.testing.assert_allclose(factors, [1.0, 0.6, 0.2])


def test_projected_depth_factors_handle_flat_depth_and_reject_invalid_inputs() -> None:
    flat = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 3.0]])
    np.testing.assert_array_equal(
        surface_common.projected_depth_factors(
            flat,
            flat,
            np.eye(4),
            minimum=0.2,
        ),
        np.ones(2),
    )

    with pytest.raises(ValueError, match="shape"):
        surface_common.projected_depth_factors(
            np.zeros((2, 2)), flat, np.eye(4), minimum=0.2
        )
    with pytest.raises(ValueError, match="projection"):
        surface_common.projected_depth_factors(
            flat, flat, np.eye(3), minimum=0.2
        )
    with pytest.raises(ValueError, match="finite"):
        surface_common.projected_depth_factors(
            np.array([[0.0, 0.0, np.nan]]), flat, np.eye(4), minimum=0.2
        )
    with pytest.raises(ValueError, match="minimum"):
        surface_common.projected_depth_factors(
            flat, flat, np.eye(4), minimum=1.1
        )


def test_depth_cued_line_data_segments_curves_and_varies_only_alpha() -> None:
    curves = (
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 10.0]]),
        np.array([[1.0, 0.0, 2.0], [1.0, 0.0, 6.0], [1.0, 0.0, 8.0]]),
    )
    colors = np.array([[1.0, 0.2, 0.1, 1.0], [0.1, 0.2, 1.0, 1.0]])

    paths, resolved_colors, widths = surface_common.depth_cued_line_data(
        curves,
        colors,
        np.array([1.0, 3.0]),
        reference_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        projection=np.eye(4),
        minimum=0.2,
        alpha=0.8,
        enabled=True,
    )

    assert len(paths) == 4
    assert all(path.shape == (2, 3) for path in paths)
    np.testing.assert_allclose(
        resolved_colors[:2, :3], np.repeat(colors[[0], :3], 2, axis=0)
    )
    np.testing.assert_allclose(
        resolved_colors[2:, :3], np.repeat(colors[[1], :3], 2, axis=0)
    )
    assert np.ptp(resolved_colors[:, 3]) > 0.0
    assert np.max(resolved_colors[:, 3]) <= 0.8
    np.testing.assert_allclose(widths, [1.0, 1.0, 3.0, 3.0])


def test_disabled_depth_cue_keeps_one_uniform_path_per_curve() -> None:
    curves = (
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 10.0]]),
        np.array([[1.0, 0.0, 2.0], [1.0, 0.0, 6.0], [1.0, 0.0, 8.0]]),
    )
    colors = np.array([[1.0, 0.2, 0.1, 1.0], [0.1, 0.2, 1.0, 1.0]])

    paths, resolved_colors, widths = surface_common.depth_cued_line_data(
        curves,
        colors,
        np.array([1.0, 3.0]),
        reference_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]),
        projection=np.eye(4),
        minimum=0.2,
        alpha=0.8,
        enabled=False,
    )

    assert len(paths) == 2
    assert all(path.shape == (3, 3) for path in paths)
    np.testing.assert_allclose(resolved_colors[:, :3], colors[:, :3])
    np.testing.assert_allclose(resolved_colors[:, 3], 0.8)
    np.testing.assert_allclose(widths, [1.0, 3.0])


def test_select_panel_edges_handles_all_120_undirected_edges() -> None:
    node_count = 16
    edges = tuple(Edge(i, j, 1.0) for i in range(node_count) for j in range(i + 1, node_count))
    visible = np.ones((node_count, node_count)) - np.eye(node_count)
    network = PreparedConnectome(
        visible,
        visible,
        edges,
        np.full(node_count, node_count - 1, dtype=float),
        False,
        tuple(str(i) for i in range(node_count)),
        {},
    )
    mesh = HemisphereMesh(np.zeros((1, 3)), np.empty((0, 3), int))
    geom = geometry_from_arrays(
        network.node_names,
        np.zeros((node_count, 3)),
        tuple("left" for _ in range(node_count)),
        {"left": mesh},
    )
    assert len(select_panel_edges(network.edges, geom, "left")) == 120


def test_bilateral_panel_uses_one_polymesh_and_forces_background_colors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt
    import nilearn.plotting as nilearn_plotting

    calls: list[dict[str, object]] = []

    def capture_plot_surf(**kwargs):
        calls.append(kwargs)
        return kwargs["figure"]

    monkeypatch.setattr(nilearn_plotting, "plot_surf", capture_plot_surf)
    result = plot_surface_matplotlib(
        prepared(),
        geometry(),
        views=(ViewSpec("both", "dorsal", "Both"),),
        node_overlay="none",
        node_offset_mm=0.0,
        colorbar=False,
        show=False,
        dpi=60,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["hemi"] == "both"
    assert set(call["surf_mesh"].parts) == {"left", "right"}
    assert call["surf_map"] is not None
    assert call["bg_map"] is not None
    assert call["threshold"] is None
    assert call["cmap"].name == "pyconnviz-transparent"
    assert call["alpha"] == pytest.approx(0.08)
    assert call["vmin"] == -1.0
    assert call["vmax"] == 1.0
    plt.close(result.artist)


def test_soft_style_does_not_create_an_implicit_cortical_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt
    import nilearn.plotting as nilearn_plotting

    calls: list[dict[str, object]] = []

    def capture_plot_surf(**kwargs):
        calls.append(kwargs)
        return kwargs["figure"]

    monkeypatch.setattr(nilearn_plotting, "plot_surf", capture_plot_surf)
    result = plot_surface_matplotlib(
        prepared(),
        geometry(),
        style="soft",
        views=(ViewSpec("left", "lateral"),),
        colorbar=False,
        show=False,
        dpi=60,
    )

    assert calls[0]["cmap"].name == "pyconnviz-transparent"
    assert calls[0]["bg_on_data"] is False
    assert calls[0]["alpha"] == pytest.approx(0.08)
    plt.close(result.artist)


def test_matplotlib_static_defaults_keep_depth_cued_edges_legible() -> None:
    import matplotlib.pyplot as plt

    result = plot_surface_matplotlib(
        prepared(),
        geometry(),
        views=(ViewSpec("left", "lateral"),),
        node_overlay="none",
        colorbar=False,
        show=False,
        dpi=60,
    )

    axis = result.artist.axes[0]
    edges = next(
        item for item in axis.collections if isinstance(item, Line3DCollection)
    )
    edge_colors = np.asarray(edges.get_colors())
    assert np.min(edges.get_linewidths()) >= 1.4 - 1e-12
    assert np.ptp(edge_colors[:, 3]) > 0.0
    assert np.min(edge_colors[:, 3]) >= (0.95 * 0.70) - 1e-12
    assert np.max(edge_colors[:, 3]) <= 0.95 + 1e-12
    plt.close(result.artist)


def test_matplotlib_surface_uses_translucent_depth_cued_foreground() -> None:
    import matplotlib.pyplot as plt

    result = plot_surface_matplotlib(
        prepared(),
        geometry(),
        views=(ViewSpec("left", "lateral"),),
        node_overlay="none",
        cortex_alpha=0.65,
        colorbar=False,
        show=False,
        dpi=60,
    )

    axis = result.artist.axes[0]
    surface = next(
        item for item in axis.collections if isinstance(item, Poly3DCollection)
    )
    edges = next(
        item for item in axis.collections if isinstance(item, Line3DCollection)
    )
    nodes = next(
        item for item in axis.collections if isinstance(item, Path3DCollection)
    )
    assert axis.computed_zorder is False
    assert np.allclose(np.asarray(surface.get_facecolors())[:, 3], 0.65)
    assert surface.get_zorder() == 0
    assert edges.get_zorder() > surface.get_zorder()
    assert nodes.get_zorder() > edges.get_zorder()
    edge_colors = np.asarray(edges.get_colors())
    assert len(edge_colors) > len(result.panel_edges["left-lateral"])
    assert np.ptp(edge_colors[:, 3]) > 0.0
    assert nodes.get_depthshade() is True
    plt.close(result.artist)


def test_matplotlib_depth_cue_can_be_disabled_without_changing_edges() -> None:
    import matplotlib.pyplot as plt

    network = prepared()
    result = plot_surface_matplotlib(
        network,
        geometry(),
        views=(ViewSpec("left", "lateral"),),
        node_overlay="none",
        depth_cue=False,
        edge_alpha=0.64,
        colorbar=False,
        show=False,
        dpi=60,
    )

    axis = result.artist.axes[0]
    edges = next(
        item for item in axis.collections if isinstance(item, Line3DCollection)
    )
    nodes = next(
        item for item in axis.collections if isinstance(item, Path3DCollection)
    )
    assert result.panel_edges["left-lateral"] == select_panel_edges(
        network.edges, geometry(), "left"
    )
    assert len(edges.get_colors()) == len(result.panel_edges["left-lateral"])
    np.testing.assert_allclose(np.asarray(edges.get_colors())[:, 3], 0.64)
    assert nodes.get_depthshade() is False
    plt.close(result.artist)


def test_matplotlib_depth_cue_requires_a_bool() -> None:
    with pytest.raises(TypeError, match="depth_cue must be a bool"):
        plot_surface_matplotlib(
            prepared(),
            geometry(),
            views=(ViewSpec("left", "lateral"),),
            depth_cue=1,  # type: ignore[arg-type]
            colorbar=False,
            show=False,
            dpi=60,
        )


def test_surface_cmap_is_independent_from_node_cmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt
    import nilearn.plotting as nilearn_plotting

    calls: list[dict[str, object]] = []

    def capture_plot_surf(**kwargs):
        calls.append(kwargs)
        return kwargs["figure"]

    monkeypatch.setattr(nilearn_plotting, "plot_surf", capture_plot_surf)
    result = plot_surface_matplotlib(
        prepared(),
        geometry(),
        views=(ViewSpec("left", "lateral"),),
        node_overlay="gaussian",
        node_cmap="viridis",
        surface_cmap="RdBu_r",
        overlay_sigma_mm=2.0,
        overlay_radius_mm=4.0,
        colorbar=False,
        show=False,
        dpi=60,
    )

    assert calls[0]["cmap"] == "RdBu_r"
    assert 0.0 < calls[0]["threshold"] < 1e-100
    node_scatter = next(
        item for item in result.artist.axes[0].collections if isinstance(item, Path3DCollection)
    )
    assert node_scatter.get_cmap().name == "viridis"
    plt.close(result.artist)


def test_no_overlay_renders_nonuniform_sulcal_facecolors() -> None:
    import matplotlib.pyplot as plt

    mesh = HemisphereMesh(
        np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], float),
        np.array([[0, 1, 2], [0, 2, 3]], int),
        np.array([-1.0, -1.0, 1.0, 1.0]),
    )
    geom = geometry_from_arrays(
        ("L",),
        np.array([[0.0, 0.0, 0.0]]),
        ("left",),
        {"left": mesh},
        node_vertices=np.array([0]),
    )
    network = prepare_connectome(np.zeros((1, 1)), geometry=geom)
    result = plot_surface_matplotlib(
        network,
        geom,
        views=(ViewSpec("left", "dorsal"),),
        node_overlay="none",
        colorbar=False,
        show=False,
        dpi=60,
    )

    surface = next(
        item for item in result.artist.axes[0].collections if isinstance(item, Poly3DCollection)
    )
    colors = np.asarray(surface.get_facecolors())[:, :3]
    assert len(np.unique(np.round(colors, 5), axis=0)) > 1
    plt.close(result.artist)


def test_static_surface_renders_png_and_svg_with_batched_artists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    show_calls: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    png = tmp_path / "surface.png"
    svg = tmp_path / "surface.svg"
    network = prepared()
    result = plot_surface_matplotlib(
        network,
        geometry(),
        views="paper",
        node_overlay="gaussian",
        overlay_sigma_mm=2.0,
        overlay_radius_mm=4.0,
        output=(png, svg),
        show=False,
        dpi=100,
    )
    assert result.prepared is network
    assert result.output_files == (png, svg)
    assert all(path.stat().st_size > 100 for path in (png, svg))
    assert show_calls == []
    assert set(result.panel_edges) == {"left-lateral", "right-lateral", "both-dorsal"}
    for axis in result.artist.axes[:3]:
        assert sum(isinstance(item, Path3DCollection) for item in axis.collections) == 1
        assert sum(isinstance(item, Line3DCollection) for item in axis.collections) <= 1


def test_empty_connectome_renders_nodes_without_edge_colorbar(tmp_path: Path) -> None:
    network = prepare_connectome(np.zeros((4, 4)), geometry=geometry())
    result = plot_surface_matplotlib(
        network,
        geometry(),
        views=(ViewSpec("both", "dorsal", "Empty"),),
        output=tmp_path / "empty.png",
        show=False,
        colorbar=True,
        dpi=80,
    )
    assert result.panel_edges["both-dorsal"] == ()
    assert len(result.artist.axes) == 2  # panel plus node colorbar; no edge colorbar


def test_colorbars_are_reserved_to_the_right_of_all_surface_panels() -> None:
    result = plot_surface_matplotlib(
        prepared(),
        geometry(),
        views="paper",
        figsize=(14.0, 5.5),
        show=False,
        colorbar=True,
        dpi=60,
    )
    result.artist.canvas.draw()
    panel_axes = result.artist.axes[:3]
    colorbar_axes = result.artist.axes[3:]
    panel_right = max(axis.get_position().x1 for axis in panel_axes)

    assert len(colorbar_axes) == 2
    assert all(axis.get_position().x0 > panel_right for axis in colorbar_axes)
    assert min(axis.get_position().height for axis in panel_axes) > 0.6


def test_style_changes_do_not_mutate_prepared_edges(tmp_path: Path) -> None:
    network = prepared()
    before = network.edges
    plot_surface_matplotlib(
        network,
        geometry(),
        style="dark",
        views=(ViewSpec("left", "lateral"),),
        output=tmp_path / "dark.png",
        show=False,
        colorbar=False,
        dpi=60,
    )
    assert network.edges == before
