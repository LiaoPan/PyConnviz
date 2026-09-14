from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg", force=True)

MODULE_NAME = "pyconnviz.plotting._matplotlib_ball_stick"


def _adapter():
    assert find_spec(MODULE_NAME) is not None, "Matplotlib ball-and-stick adapter is missing"
    return import_module(MODULE_NAME)


def test_spheres_are_one_auditable_shaded_poly_collection() -> None:
    adapter = _adapter()

    collection = adapter.sphere_collection(
        np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.array([2.0, 4.0]),
        ((0.2, 0.4, 0.6, 1.0), (0.8, 0.3, 0.1, 1.0)),
        depth_cue=True,
    )

    assert isinstance(collection, Poly3DCollection)
    assert collection.get_gid() == "pyconnviz-nodes"
    np.testing.assert_allclose(collection.diameters, [2.0, 4.0])
    assert collection.mesh_vertex_count == 2 * (2 + 7 * 12)
    assert collection.mesh_triangle_count == 2 * (2 * 12 * 7)
    assert collection.depth_cue is True
    assert collection.get_zorder() == 12
    np.testing.assert_allclose(
        collection.source_colors,
        ((0.2, 0.4, 0.6, 1.0), (0.8, 0.3, 0.1, 1.0)),
    )


def test_tubes_are_one_physical_collection_with_explicit_alpha() -> None:
    adapter = _adapter()
    curves = (
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
        np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 2.0], [2.0, 0.0, 4.0]]),
    )

    collection = adapter.tube_collection(
        curves,
        np.array([1.0, 2.0]),
        ((0.1, 0.2, 0.8, 1.0), (0.8, 0.2, 0.1, 1.0)),
        alpha=0.65,
        depth_cue=False,
    )

    assert isinstance(collection, Poly3DCollection)
    assert collection.get_gid() == "pyconnviz-edges"
    np.testing.assert_allclose(collection.diameters, [1.0, 2.0])
    assert collection.mesh_vertex_count == 2 * 3 * 8
    assert collection.mesh_triangle_count == 2 * 2 * (3 - 1) * 8
    assert collection.get_alpha() == 0.65
    assert collection.depth_cue is False
    assert collection.get_zorder() == 10


def test_empty_mesh_sequences_return_none() -> None:
    adapter = _adapter()

    assert adapter.sphere_collection(np.empty((0, 3)), np.empty(0), (), depth_cue=True) is None
    assert adapter.tube_collection((), np.empty(0), (), alpha=1.0, depth_cue=True) is None
