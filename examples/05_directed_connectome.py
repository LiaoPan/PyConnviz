"""Render a small simulated directed surface network with endpoint markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyconnviz import HemisphereMesh, geometry_from_arrays, plot_connectome


def main(output: Path) -> None:
    """Create a deterministic asymmetric matrix and preserve source-to-target edges."""

    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    left = np.array(
        [[-4, -1, -1], [-2, -1, 1], [-4, 1, 1], [-2, 1, -1]], float
    )
    right = left + np.array([6, 0, 0])
    geometry = geometry_from_arrays(
        ("L-a", "L-b", "R-a", "R-b"),
        np.array([left[0], left[3], right[0], right[3]]),
        ("left", "left", "right", "right"),
        {
            "left": HemisphereMesh(left, faces),
            "right": HemisphereMesh(right, faces),
        },
        node_vertices=np.array([0, 3, 0, 3]),
    )
    directed = np.array(
        [[0, 0.7, 0, 0], [0, 0, -0.5, 0], [0.2, 0, 0, 0.9], [0, 0, 0, 0]],
        float,
    )
    plot_connectome(
        directed,
        geometry,
        directed=True,
        max_edges=4,
        show_arrows=True,
        title="Simulated directed connectivity",
        output=output,
        show=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("simulated_directed.png"))
    main(parser.parse_args().output)
