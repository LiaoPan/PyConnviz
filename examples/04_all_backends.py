"""Export one deterministic simulated PreparedConnectome through every backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyconnviz import HemisphereMesh, geometry_from_arrays, plot_connectome, prepare_connectome


def make_data():
    """Return a tiny bilateral geometry and simulated prepared network."""

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
            "left": HemisphereMesh(left, faces, np.linspace(-1, 1, 4)),
            "right": HemisphereMesh(right, faces, np.linspace(-1, 1, 4)),
        },
        mni_coords=np.array(
            [[-30, -10, 20], [-20, 10, 20], [20, -10, 20], [30, 10, 20]]
        ),
        node_vertices=np.array([0, 3, 0, 3]),
    )
    matrix = np.array(
        [
            [0, 0.4, -0.8, 0],
            [0.4, 0, 0.6, 0],
            [-0.8, 0.6, 0, 1],
            [0, 0, 1, 0],
        ],
        float,
    )
    return geometry, prepare_connectome(matrix, geometry=geometry)


def main(output_dir: Path) -> None:
    """Export all simulated backends without preparing the edges twice."""

    output_dir.mkdir(parents=True, exist_ok=True)
    geometry, prepared = make_data()
    plot_connectome(
        prepared,
        geometry,
        backend="surface",
        engine="plotly",
        output=output_dir / "surface.html",
    )
    plot_connectome(
        prepared, geometry, backend="glass", output=output_dir / "glass.svg"
    )
    plot_connectome(
        prepared, geometry, backend="html", output=output_dir / "connectome.html"
    )
    plot_connectome(
        prepared,
        geometry,
        backend="surface",
        engine="matplotlib",
        depth_cue=True,
        output=output_dir / "surface_context.png",
    )
    plot_connectome(
        prepared,
        geometry,
        backend="surface",
        engine="nilearn",
        views=("lateral", "medial", "dorsal"),
        hemispheres=("left", "right"),
        depth_cue=True,
        output=output_dir / "surface_nilearn_context.png",
    )
    plot_connectome(
        prepared, geometry, backend="circle", output=output_dir / "circle.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("simulated_all_backends")
    )
    main(parser.parse_args().output_dir)
