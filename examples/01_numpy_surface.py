"""Generate an offline static surface plot from deterministic simulated data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyconnviz import HemisphereMesh, geometry_from_arrays, prepare_connectome
from pyconnviz.plotting.surface_matplotlib import plot_surface_matplotlib


def _tetrahedron(center_x: float) -> HemisphereMesh:
    coordinates = np.array(
        [
            [center_x - 1, -1, -1],
            [center_x + 1, -1, 1],
            [center_x - 1, 1, 1],
            [center_x + 1, 1, -1],
        ],
        float,
    )
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return HemisphereMesh(coordinates, faces, np.linspace(-1, 1, 4))


def main(output: Path) -> None:
    """Write a clearly labelled simulated surface connectome."""

    geometry = geometry_from_arrays(
        ("L-a", "L-b", "R-a", "R-b"),
        np.array([[-4, -1, -1], [-2, 1, -1], [2, -1, -1], [4, 1, -1]], float),
        ("left", "left", "right", "right"),
        {"left": _tetrahedron(-3), "right": _tetrahedron(3)},
        node_vertices=np.array([0, 3, 0, 3]),
    )
    matrix = np.array(
        [[0, 0.4, -0.8, 0], [0.4, 0, 0.6, 0], [-0.8, 0.6, 0, 1], [0, 0, 1, 0]],
        float,
    )
    prepared = prepare_connectome(matrix, geometry=geometry)
    plot_surface_matplotlib(
        prepared,
        geometry,
        title="Simulated PyConnviz connectivity",
        output=output,
        show=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("simulated_surface.png"))
    main(parser.parse_args().output)
