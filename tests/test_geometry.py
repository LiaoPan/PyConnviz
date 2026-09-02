from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyconnviz.geometry import (
    compute_vertex_normals,
    geometry_from_arrays,
    geometry_from_mne_labels,
    offset_node_coordinates,
    to_nilearn_polymesh,
)
from pyconnviz.models import GeometryError, HemisphereMesh


def plane_mesh(offset: float = 0.0) -> HemisphereMesh:
    return HemisphereMesh(
        np.array([[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]], float),
        np.array([[0, 1, 2]], int),
        np.array([-1, 0, 1], float),
    )


def test_geometry_from_arrays_preserves_both_coordinate_spaces() -> None:
    surface = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    mni = np.array([[-20.0, 0.0, 40.0], [20.0, 0.0, 40.0]])
    result = geometry_from_arrays(
        ("L", "R"),
        surface,
        ("left", "right"),
        {"left": plane_mesh(), "right": plane_mesh(5)},
        mni_coords=mni,
        node_vertices=np.array([0, 0]),
    )
    surface[:] = 99
    mni[:] = 99
    np.testing.assert_allclose(result.surface_coords[:, 0], [0, 5])
    np.testing.assert_allclose(result.mni_coords[:, 0], [-20, 20])


def test_area_weighted_plane_normals_are_unit_z() -> None:
    normals = compute_vertex_normals(plane_mesh())
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (3, 1)))


def test_degenerate_and_isolated_vertices_have_finite_unit_fallbacks() -> None:
    mesh = HemisphereMesh(
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], float),
        np.array([[0, 1, 2]], int),
    )
    normals = compute_vertex_normals(mesh)
    assert np.all(np.isfinite(normals))
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_node_offset_follows_vertex_normal_at_exact_distance() -> None:
    result = geometry_from_arrays(
        ("node",),
        np.array([[0.0, 0.0, 0.0]]),
        ("left",),
        {"left": plane_mesh()},
        node_vertices=np.array([0]),
    )
    offset = offset_node_coordinates(result, offset_mm=1.5)
    np.testing.assert_allclose(offset, [[0.0, 0.0, 1.5]])
    with pytest.raises(GeometryError, match="node_vertices"):
        offset_node_coordinates(
            geometry_from_arrays(
                ("node",), np.zeros((1, 3)), ("left",), {"left": plane_mesh()}
            )
        )


def test_nilearn_polymesh_has_matching_parts_and_vertices() -> None:
    result = geometry_from_arrays(
        ("L", "R"),
        np.array([[0, 0, 0], [5, 0, 0]], float),
        ("left", "right"),
        {"left": plane_mesh(), "right": plane_mesh(5)},
    )
    converted = to_nilearn_polymesh(result)
    assert set(converted.parts) == {"left", "right"}
    assert converted.n_vertices == 6
    np.testing.assert_array_equal(converted.parts["left"].faces, result.meshes["left"].faces)


class FakeLabel:
    def __init__(self, name: str, hemi: str, vertices: list[int], com: int) -> None:
        self.name = name
        self.hemi = hemi
        self.vertices = np.array(vertices, int)
        self.com = com
        self.calls: list[dict[str, object]] = []

    def center_of_mass(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return self.com


def test_mne_label_geometry_keeps_order_com_contract_and_mni_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surf_dir = tmp_path / "subject" / "surf"
    surf_dir.mkdir(parents=True)
    for hemi in ("lh", "rh"):
        for name in ("inflated", "sulc"):
            (surf_dir / f"{hemi}.{name}").touch()

    left_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    right_coords = left_coords + np.array([10, 0, 0])
    faces = np.array([[0, 1, 2]], int)
    vertex_to_mni_calls: list[tuple[object, object, object, object]] = []
    curvature_calls: list[Path] = []
    curvature_binary: list[bool] = []

    def read_surface(path):
        return (left_coords, faces) if Path(path).name.startswith("lh") else (right_coords, faces)

    def read_curvature(path, *, binary=True):
        curvature_calls.append(Path(path))
        curvature_binary.append(binary)
        return np.array([-1.0, 0.0, 1.0])

    def vertex_to_mni(vertices, hemis, subject, subjects_dir):
        vertex_to_mni_calls.append((vertices, hemis, subject, subjects_dir))
        return np.array([[-20.0, 0.0, 40.0], [20.0, 0.0, 42.0]])

    fake_mne = SimpleNamespace(
        read_surface=read_surface,
        surface=SimpleNamespace(read_curvature=read_curvature),
        vertex_to_mni=vertex_to_mni,
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    labels = [FakeLabel("right-first", "rh", [0, 2], 2), FakeLabel("left-second", "lh", [0, 1], 1)]
    source_space = object()

    result = geometry_from_mne_labels(
        labels,
        subject="subject",
        subjects_dir=tmp_path,
        src=source_space,
        surface="inflated",
        sphere_surface="sphere",
    )

    assert result.node_names == ("right-first", "left-second")
    assert result.hemispheres == ("right", "left")
    np.testing.assert_allclose(result.surface_coords, [right_coords[2], left_coords[1]])
    np.testing.assert_allclose(result.mni_coords[:, 0], [-20, 20])
    np.testing.assert_array_equal(result.roi_vertices[0], [0, 2])
    np.testing.assert_array_equal(result.meshes["left"].sulc, [-1.0, 0.0, 1.0])
    assert [path.name for path in curvature_calls] == ["lh.sulc", "rh.sulc"]
    assert curvature_binary == [False, False]
    for label in labels:
        assert label.calls == [
            {
                "subject": "subject",
                "restrict_vertices": source_space,
                "subjects_dir": tmp_path,
                "surf": "sphere",
            }
        ]
    assert vertex_to_mni_calls == [([2, 1], [1, 0], "subject", tmp_path)]


def test_mne_geometry_missing_surface_names_exact_path(tmp_path: Path) -> None:
    label = FakeLabel("L", "lh", [0], 0)
    with pytest.raises(GeometryError, match=r"lh\.inflated"):
        geometry_from_mne_labels(
            [label], subject="missing", subjects_dir=tmp_path, surface="inflated"
        )


def test_mne_geometry_missing_sulc_warns_but_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surf_dir = tmp_path / "subject" / "surf"
    surf_dir.mkdir(parents=True)
    (surf_dir / "lh.inflated").touch()
    fake_mne = SimpleNamespace(
        read_surface=lambda path: (plane_mesh().coordinates, plane_mesh().faces),
        surface=SimpleNamespace(read_curvature=lambda path: np.ones(3)),
        vertex_to_mni=lambda *args, **kwargs: np.array([[0.0, 0.0, 0.0]]),
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    with pytest.warns(RuntimeWarning, match="sulc"):
        result = geometry_from_mne_labels(
            [FakeLabel("L", "lh", [0], 0)],
            subject="subject",
            subjects_dir=tmp_path,
        )
    assert result.meshes["left"].sulc is None


def test_mne_geometry_rejects_bihemi_or_unknown_label_hemi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace())
    label = FakeLabel("both", "both", [0], 0)
    with pytest.raises(GeometryError, match=r"lh.*rh"):
        geometry_from_mne_labels([label], subject="subject", subjects_dir=tmp_path)
