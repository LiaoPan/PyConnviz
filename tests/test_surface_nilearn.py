from __future__ import annotations

from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
from PIL import Image

from pyconnviz.connectivity import prepare_connectome
from pyconnviz.geometry import geometry_from_arrays
from pyconnviz.models import GeometryError, HemisphereMesh
from pyconnviz.plotting.surface_nilearn import (
    freesurfer_surface_paths,
    plot_surface_nilearn,
    resolve_native_views,
)

matplotlib.use("Agg", force=True)


def _tetra_mesh(center_x: float) -> HemisphereMesh:
    coordinates = np.array(
        [
            [center_x - 1, -1, -1],
            [center_x + 1, -1, 1],
            [center_x - 1, 1, 1],
            [center_x + 1, 1, -1],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return HemisphereMesh(
        coordinates,
        faces,
        np.array([-1.0, -0.3, 0.3, 1.0]),
    )


def _geometry():
    return geometry_from_arrays(
        ("L0", "L1", "R0", "R1"),
        np.array([[-4, -1, -1], [-2, 1, -1], [2, -1, -1], [4, 1, -1]]),
        ("left", "left", "right", "right"),
        {"left": _tetra_mesh(-3), "right": _tetra_mesh(3)},
        node_vertices=np.array([0, 3, 0, 3]),
    )


def _prepared():
    matrix = np.array(
        [
            [0, 0.4, -0.8, 0],
            [0.4, 0, 0.6, 0],
            [-0.8, 0.6, 0, 1.0],
            [0, 0, 1.0, 0],
        ],
        dtype=float,
    )
    return prepare_connectome(matrix, geometry=_geometry())


def _fake_native_axes(view_count: int, hemisphere_count: int):
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = [
        figure.add_subplot(
            view_count,
            hemisphere_count,
            index + 1,
            projection="3d",
        )
        for index in range(view_count * hemisphere_count)
    ]
    return figure, axes


def test_native_view_presets_and_explicit_order() -> None:
    assert resolve_native_views("paper") == ("lateral", "medial", "dorsal")
    assert resolve_native_views("four") == ("lateral", "medial")
    assert resolve_native_views("lateral") == ("lateral",)
    assert resolve_native_views("whole") == ("dorsal", "anterior", "posterior")
    assert resolve_native_views(["posterior", "dorsal"]) == (
        "posterior",
        "dorsal",
    )


@pytest.mark.parametrize(
    "views",
    [[], ["lateral", "lateral"], ["side"], [(20.0, 30.0)]],
)
def test_native_views_reject_ambiguous_or_unsupported_values(views: object) -> None:
    with pytest.raises((TypeError, ValueError), match="views"):
        resolve_native_views(views)  # type: ignore[arg-type]


def _make_freesurfer_tree(root: Path) -> dict[str, Path]:
    surf = root / "surf"
    surf.mkdir(parents=True)
    for hemi in ("lh", "rh"):
        for suffix in ("pial", "inflated", "sulc", "curv"):
            (surf / f"{hemi}.{suffix}").touch()
    return {
        "pial_left": surf / "lh.pial",
        "pial_right": surf / "rh.pial",
        "infl_left": surf / "lh.inflated",
        "infl_right": surf / "rh.inflated",
        "sulc_left": surf / "lh.sulc",
        "sulc_right": surf / "rh.sulc",
        "curv_left": surf / "lh.curv",
        "curv_right": surf / "rh.curv",
    }


def test_freesurfer_surface_paths_resolve_every_native_input(tmp_path: Path) -> None:
    subject = tmp_path / "fsaverage"
    expected = _make_freesurfer_tree(subject)

    assert freesurfer_surface_paths(subject) == expected


def test_freesurfer_surface_paths_name_the_exact_missing_file(tmp_path: Path) -> None:
    subject = tmp_path / "fsaverage"
    expected = _make_freesurfer_tree(subject)
    expected["curv_right"].unlink()

    with pytest.raises(GeometryError, match=r"rh\.curv"):
        freesurfer_surface_paths(subject)


def test_native_renderer_passes_real_volume_and_montage_options_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt
    import nilearn.plotting as nilearn_plotting

    calls: list[dict[str, object]] = []

    def capture_plot_img_on_surf(**kwargs):
        calls.append(kwargs)
        return _fake_native_axes(3, 2)

    monkeypatch.setattr(
        nilearn_plotting,
        "plot_img_on_surf",
        capture_plot_img_on_surf,
    )
    stat_img = nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4))
    result = plot_surface_nilearn(
        _prepared(),
        _geometry(),
        stat_map=stat_img,
        views="paper",
        hemispheres=("left", "right"),
        bg_on_data=True,
        symmetric_cmap=None,
        symmetric_cbar="auto",
        inflate=False,
        colorbar=True,
        show=False,
        dpi=72,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["stat_map"] is stat_img
    assert call["views"] == ["lateral", "medial", "dorsal"]
    assert call["hemispheres"] == ["left", "right"]
    assert call["bg_on_data"] is True
    assert call["symmetric_cmap"] is None
    assert call["symmetric_cbar"] == "auto"
    assert call["inflate"] is False
    assert call["colorbar"] is True
    assert "output_file" not in call
    assert result.backend == "surface"
    assert result.engine == "nilearn"
    plt.close(result.artist)


def test_native_renderer_uses_a_transparent_zero_carrier_without_stat_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt
    import nilearn.plotting as nilearn_plotting

    calls: list[dict[str, object]] = []

    def capture_plot_img_on_surf(**kwargs):
        calls.append(kwargs)
        return _fake_native_axes(1, 2)

    monkeypatch.setattr(
        nilearn_plotting,
        "plot_img_on_surf",
        capture_plot_img_on_surf,
    )
    result = plot_surface_nilearn(
        _prepared(),
        _geometry(),
        stat_map=None,
        views="lateral",
        colorbar=True,
        show=False,
        dpi=72,
    )

    call = calls[0]
    carrier = call["stat_map"]
    assert isinstance(carrier, nib.Nifti1Image)
    np.testing.assert_array_equal(np.asanyarray(carrier.dataobj), 0)
    assert call["cmap"].name == "pyconnviz-transparent"
    assert np.all(np.asarray(call["cmap"].colors)[:, 3] == 0)
    assert call["threshold"] == np.nextafter(0.0, 1.0)
    assert call["colorbar"] is False
    plt.close(result.artist)


def test_missing_node_vertices_fails_before_native_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nilearn.plotting as nilearn_plotting

    render_calls: list[bool] = []
    monkeypatch.setattr(
        nilearn_plotting,
        "plot_img_on_surf",
        lambda **_: render_calls.append(True),
    )
    source = _geometry()
    geometry_without_vertices = geometry_from_arrays(
        source.node_names,
        source.surface_coords,
        source.hemispheres,
        source.meshes,
    )
    network = prepare_connectome(
        _prepared().matrix,
        geometry=geometry_without_vertices,
    )

    with pytest.raises(GeometryError, match="node_vertices"):
        plot_surface_nilearn(
            network,
            geometry_without_vertices,
            views="lateral",
            colorbar=False,
        )

    assert render_calls == []


def test_native_renderer_aligns_nodes_and_scopes_edges_per_hemisphere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib.pyplot as plt
    import nilearn.plotting as nilearn_plotting

    monkeypatch.setattr(
        nilearn_plotting,
        "plot_img_on_surf",
        lambda **_: _fake_native_axes(3, 2),
    )
    network = _prepared()
    original_edges = network.edges
    result = plot_surface_nilearn(
        network,
        _geometry(),
        views="paper",
        node_offset_mm=0.0,
        colorbar=False,
        show=False,
        dpi=72,
    )

    assert result.prepared is network
    assert network.edges is original_edges
    assert tuple(result.panel_edges) == (
        "left-lateral",
        "right-lateral",
        "left-medial",
        "right-medial",
        "left-dorsal",
        "right-dorsal",
    )
    expected_pairs = {
        "left": {(0, 1)},
        "right": {(2, 3)},
    }
    expected_points = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, -1.0]])
    for index, axis in enumerate(result.artist.axes[:6]):
        hemi = "left" if index % 2 == 0 else "right"
        key = tuple(result.panel_edges)[index]
        assert {
            (edge.source, edge.target) for edge in result.panel_edges[key]
        } == expected_pairs[hemi]
        edge_artist = next(
            item for item in axis.collections if isinstance(item, Line3DCollection)
        )
        node_artist = next(
            item for item in axis.collections if isinstance(item, Path3DCollection)
        )
        points = np.column_stack(node_artist._offsets3d)
        np.testing.assert_allclose(points, expected_points)
        assert axis.computed_zorder is False
        assert edge_artist.get_zorder() == 10
        assert node_artist.get_zorder() == 12
    plt.close(result.artist)


def test_native_renderer_saves_a_real_nonempty_png(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    output = tmp_path / "native.png"
    result = plot_surface_nilearn(
        _prepared(),
        _geometry(),
        views="lateral",
        node_offset_mm=0.0,
        colorbar=False,
        figsize=(6.0, 3.0),
        output=output,
        show=False,
        dpi=80,
    )

    assert result.output_files == (output,)
    assert output.stat().st_size > 100
    with Image.open(output) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    assert np.var(pixels) > 1.0
    assert len(
        [axis for axis in result.artist.axes if getattr(axis, "name", None) == "3d"]
    ) == 2
    plt.close(result.artist)
