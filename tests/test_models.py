from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyconnviz.models import (
    ConnectomeGeometry,
    Edge,
    GeometryError,
    HemisphereMesh,
    PlotResult,
    PreparedConnectome,
    ViewSpec,
)


def mesh(name: str = "left") -> HemisphereMesh:
    return HemisphereMesh(
        coordinates=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        sulc=np.array([-1.0, 0.0, 1.0]),
        name=name,
    )


def geometry() -> ConnectomeGeometry:
    left = mesh("left")
    right = HemisphereMesh(
        coordinates=left.coordinates + np.array([4.0, 0.0, 0.0]),
        faces=left.faces,
        sulc=left.sulc,
        name="right",
    )
    return ConnectomeGeometry(
        node_names=("L", "R"),
        surface_coords=np.array([[0.2, 0.2, 0.0], [4.2, 0.2, 0.0]]),
        hemispheres=("left", "right"),
        meshes={"left": left, "right": right},
        mni_coords=np.array([[-20.0, 0.0, 40.0], [20.0, 0.0, 40.0]]),
        groups=("visual", "visual"),
        node_vertices=np.array([0, 0]),
        roi_vertices=(np.array([0, 1]), np.array([0, 2])),
        subject="sample",
        surface_name="inflated",
    )


def test_mesh_validates_and_owns_read_only_arrays() -> None:
    coordinates = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    result = HemisphereMesh(coordinates, np.array([[0, 1, 2]]), name="lh")
    coordinates[0] = 99
    assert np.allclose(result.coordinates[0], 0)
    assert result.coordinates.dtype == np.float64
    assert result.faces.dtype == np.int64
    assert not result.coordinates.flags.writeable
    with pytest.raises(ValueError):
        result.faces[0, 0] = 2


@pytest.mark.parametrize(
    ("coordinates", "faces", "sulc", "message"),
    [
        (np.zeros((3, 2)), np.array([[0, 1, 2]]), None, "coordinates"),
        (np.array([[0, 0, np.nan]]), np.empty((0, 3), int), None, "finite"),
        (np.zeros((3, 3)), np.array([[0, 1]]), None, "faces"),
        (np.zeros((3, 3)), np.array([[0.0, 1.0, 2.0]]), None, "integer"),
        (np.zeros((3, 3)), np.array([[0, 1, 3]]), None, "range"),
        (np.zeros((3, 3)), np.array([[0, 1, 2]]), np.zeros(2), "sulc"),
    ],
)
def test_mesh_rejects_invalid_fields(coordinates, faces, sulc, message: str) -> None:
    with pytest.raises(GeometryError, match=message):
        HemisphereMesh(coordinates, faces, sulc)


def test_geometry_validates_node_fields_and_preserves_hemisphere_vertices() -> None:
    result = geometry()
    assert result.node_names == ("L", "R")
    assert result.node_vertices.tolist() == [0, 0]
    assert result.coordinate_units == "mm"
    assert set(result.meshes) == {"left", "right"}
    assert not result.surface_coords.flags.writeable
    assert not result.roi_vertices[0].flags.writeable


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"surface_coords": np.zeros((2, 2))}, "surface_coords"),
        ({"surface_coords": np.array([[0, 0, 0], [np.inf, 0, 0]])}, "finite"),
        ({"hemispheres": ("left", "middle")}, "hemispheres"),
        ({"mni_coords": np.zeros((1, 3))}, "mni_coords"),
        ({"groups": ("only-one",)}, "groups"),
        ({"node_vertices": np.array([0])}, "node_vertices"),
        ({"node_vertices": np.array([0.5, 0.0])}, "integer"),
        ({"node_vertices": np.array([99, 0])}, "node_vertices"),
        ({"roi_vertices": (np.array([0]),)}, "roi_vertices"),
        ({"coordinate_units": "m"}, "millimetres"),
    ],
)
def test_geometry_rejects_inconsistent_node_fields(changes, message: str) -> None:
    kwargs = dict(
        node_names=("L", "R"),
        surface_coords=np.array([[0.2, 0.2, 0.0], [4.2, 0.2, 0.0]]),
        hemispheres=("left", "right"),
        meshes={"left": mesh("left"), "right": mesh("right")},
        mni_coords=np.zeros((2, 3)),
        groups=("a", "b"),
        node_vertices=np.array([0, 0]),
        roi_vertices=(np.array([0]), np.array([0])),
    )
    kwargs.update(changes)
    with pytest.raises(GeometryError, match=message):
        ConnectomeGeometry(**kwargs)


def test_prepared_connectome_validates_edges_and_visible_matrix() -> None:
    matrix = np.array([[0.0, 0.7], [0.7, 0.0]])
    result = PreparedConnectome(
        matrix=matrix,
        visible_matrix=matrix,
        edges=(Edge(0, 1, 0.7, 4.0),),
        node_strength=np.array([0.7, 0.7]),
        directed=False,
        node_names=("L", "R"),
        metadata={"source": "numpy"},
    )
    matrix[0, 1] = 0.0
    assert result.visible_matrix[0, 1] == pytest.approx(0.7)
    assert not result.visible_matrix.flags.writeable
    assert result.metadata["source"] == "numpy"
    with pytest.raises(TypeError):
        result.metadata["source"] = "changed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"matrix": np.zeros((2, 3))}, "square"),
        ({"visible_matrix": np.zeros((3, 3))}, "visible_matrix"),
        ({"node_strength": np.zeros(1)}, "node_strength"),
        ({"node_names": ("only",)}, "node_names"),
        ({"edges": (Edge(0, 2, 1.0),)}, "edge index"),
    ],
)
def test_prepared_connectome_rejects_inconsistent_fields(kwargs, message: str) -> None:
    values = dict(
        matrix=np.zeros((2, 2)),
        visible_matrix=np.zeros((2, 2)),
        edges=(),
        node_strength=np.zeros(2),
        directed=False,
        node_names=("a", "b"),
        metadata={},
    )
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PreparedConnectome(**values)


@pytest.mark.parametrize(
    ("visible", "edges", "directed", "message"),
    [
        (np.array([[0.0, 0.5], [0.5, 0.0]]), (), False, "exactly match"),
        (np.zeros((2, 2)), (Edge(0, 1, 0.5),), False, "exactly match"),
        (
            np.array([[0.0, 0.5], [0.5, 0.0]]),
            (Edge(0, 1, 0.5), Edge(1, 0, 0.5)),
            False,
            "duplicate",
        ),
        (np.array([[0.0, 0.5], [0.0, 0.0]]), (Edge(0, 1, 0.5),), False, "symmetric"),
        (np.array([[0.0, 0.5], [0.0, 0.0]]), (Edge(0, 1, 0.4),), True, "exactly match"),
    ],
)
def test_prepared_connectome_requires_exact_edge_table_contract(
    visible: np.ndarray,
    edges: tuple[Edge, ...],
    directed: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PreparedConnectome(
            matrix=visible,
            visible_matrix=visible,
            edges=edges,
            node_strength=np.sum(np.abs(visible), axis=1),
            directed=directed,
            node_names=("a", "b"),
        )


def test_edge_view_and_plot_result_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="self-edge"):
        Edge(1, 1, 0.3)
    with pytest.raises(ValueError, match="finite"):
        Edge(0, 1, np.nan)
    assert ViewSpec("left", "lateral", "Left").hemi == "left"
    with pytest.raises(ValueError, match="hemi"):
        ViewSpec("middle", "lateral")

    prepared = PreparedConnectome(
        np.zeros((1, 1)), np.zeros((1, 1)), (), np.zeros(1), False, ("n",), {}
    )
    output = tmp_path / "plot.png"
    result = PlotResult("surface", "matplotlib", object(), prepared, (output,))
    assert result.output_files == (output,)


def test_import_is_lightweight_and_side_effect_free() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pyconnviz; "
            "assert pyconnviz.__version__ == '0.1.0'; "
            "assert 'matplotlib.pyplot' not in sys.modules; "
            "assert 'plotly' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
