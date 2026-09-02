"""Generate deterministic offline acceptance artifacts for PyConnviz."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
import tracemalloc
from pathlib import Path

import matplotlib
import numpy as np
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.surface import load_surf_data, load_surf_mesh

matplotlib.use("Agg", force=True)
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

from pyconnviz import (
    ConnectomeGeometry,
    HemisphereMesh,
    __version__,
    geometry_from_arrays,
    plot_connectome,
    prepare_connectome,
)
from pyconnviz.plotting.surface_nilearn import freesurfer_surface_paths


def _farthest_vertices(coordinates: np.ndarray, count: int) -> np.ndarray:
    """Select deterministic, spatially distributed vertices."""

    if count < 1 or count > len(coordinates):
        raise ValueError("count must be between 1 and the mesh vertex count")
    selected = np.empty(count, dtype=np.int64)
    current = int(np.argmax(coordinates[:, 2]))
    minimum_squared_distance = np.full(len(coordinates), np.inf, dtype=np.float64)
    for index in range(count):
        selected[index] = current
        difference = coordinates - coordinates[current]
        squared_distance = np.einsum("ij,ij->i", difference, difference)
        np.minimum(minimum_squared_distance, squared_distance, out=minimum_squared_distance)
        current = int(np.argmax(minimum_squared_distance))
    return selected


def make_geometry(*, node_count: int = 20) -> ConnectomeGeometry:
    """Load deterministic bilateral pial geometry bundled with Nilearn."""

    if node_count < 2 or node_count % 2:
        raise ValueError("node_count must be an even integer of at least 2")
    fsaverage = load_fsaverage("fsaverage5")
    sulcal = load_fsaverage_data(
        mesh="fsaverage5", mesh_type="pial", data_type="sulcal"
    )
    meshes: dict[str, HemisphereMesh] = {}
    for hemi in ("left", "right"):
        part = fsaverage.pial.parts[hemi]
        meshes[hemi] = HemisphereMesh(
            part.coordinates,
            part.faces,
            sulcal.data.parts[hemi],
            name=f"nilearn-fsaverage5:{hemi}:pial",
        )
    per_hemi = node_count // 2
    vertices_by_hemi = {
        hemi: _farthest_vertices(mesh.coordinates, per_hemi)
        for hemi, mesh in meshes.items()
    }
    hemispheres = tuple(["left"] * per_hemi + ["right"] * per_hemi)
    node_vertices = np.concatenate(
        (vertices_by_hemi["left"], vertices_by_hemi["right"])
    )
    surface_coords = np.asarray(
        [
            meshes[hemi].coordinates[int(vertex)]
            for hemi, vertex in zip(hemispheres, node_vertices, strict=True)
        ]
    )
    mni_coords = np.asarray(
        [
            fsaverage.pial.parts[hemi].coordinates[int(vertex)]
            for hemi, vertex in zip(hemispheres, node_vertices, strict=True)
        ]
    )
    names = tuple(
        f"{'L' if hemi == 'left' else 'R'}-{index % per_hemi:03d}"
        for index, hemi in enumerate(hemispheres)
    )
    groups = tuple(f"network-{index % 4}" for index in range(node_count))
    return geometry_from_arrays(
        names,
        surface_coords,
        hemispheres,
        meshes,
        mni_coords=mni_coords,
        groups=groups,
        node_vertices=node_vertices,
        subject="nilearn-fsaverage5",
        surface_name="pial",
    )


def make_full_fsaverage_geometry(
    subject_dir: str | Path,
    *,
    node_count: int = 20,
) -> ConnectomeGeometry:
    """Load the user-provided full-resolution fsaverage pial geometry."""

    if node_count < 2 or node_count % 2:
        raise ValueError("node_count must be an even integer of at least 2")
    paths = freesurfer_surface_paths(subject_dir)
    meshes: dict[str, HemisphereMesh] = {}
    for hemi in ("left", "right"):
        part = load_surf_mesh(paths[f"pial_{hemi}"])
        sulc = load_surf_data(paths[f"sulc_{hemi}"])
        meshes[hemi] = HemisphereMesh(
            np.asarray(part.coordinates, dtype=np.float64),
            np.asarray(part.faces, dtype=np.int64),
            np.asarray(sulc, dtype=np.float64),
            name=f"user-fsaverage:{hemi}:pial",
        )
    per_hemi = node_count // 2
    vertices_by_hemi = {
        hemi: _farthest_vertices(mesh.coordinates, per_hemi)
        for hemi, mesh in meshes.items()
    }
    hemispheres = tuple(["left"] * per_hemi + ["right"] * per_hemi)
    node_vertices = np.concatenate(
        (vertices_by_hemi["left"], vertices_by_hemi["right"])
    )
    surface_coords = np.asarray(
        [
            meshes[hemi].coordinates[int(vertex)]
            for hemi, vertex in zip(hemispheres, node_vertices, strict=True)
        ]
    )
    names = tuple(
        f"{'L' if hemi == 'left' else 'R'}-{index % per_hemi:03d}"
        for index, hemi in enumerate(hemispheres)
    )
    groups = tuple(f"network-{index % 4}" for index in range(node_count))
    return geometry_from_arrays(
        names,
        surface_coords,
        hemispheres,
        meshes,
        groups=groups,
        node_vertices=node_vertices,
        subject="user-fsaverage",
        surface_name="pial",
    )


def _network(geometry: ConnectomeGeometry, *, seed: int, max_edges: int):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(len(geometry.node_names), len(geometry.node_names)))
    matrix = (values + values.T) / 2.0
    np.fill_diagonal(matrix, 0)
    return prepare_connectome(matrix, geometry=geometry, max_edges=max_edges)


def _edge_tuples(result) -> list[list[float | int]]:
    return [
        [edge.source, edge.target, edge.weight] for edge in result.prepared.edges
    ]


def _environment_text() -> str:
    packages = {
        "PyConnviz": "pyconnviz",
        "NiBabel": "nibabel",
        "Nilearn": "nilearn",
        "MNE": "mne",
        "MNE-Connectivity": "mne-connectivity",
        "Matplotlib": "matplotlib",
        "Plotly": "plotly",
    }
    lines = [f"Python={platform.python_version()}"]
    for label, package in packages.items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "pyconnviz":
                raise
            version = __version__
        lines.append(f"{label}={version}")
    return "\n".join(lines) + "\n"


def _benchmark() -> dict[str, float | int | bool]:
    geometry = make_geometry(node_count=200)
    rng = np.random.default_rng(20260832)
    values = rng.normal(size=(200, 200))
    matrix = (values + values.T) / 2.0
    np.fill_diagonal(matrix, 0)
    tracemalloc.start()
    start = time.perf_counter()
    prepared = prepare_connectome(matrix, geometry=geometry, max_edges=120)
    prepare_seconds = time.perf_counter() - start
    render_start = time.perf_counter()
    result = plot_connectome(
        prepared,
        geometry,
        views="paper",
        colorbar=False,
        figsize=(9.0, 3.2),
        dpi=72,
        show=False,
    )
    static_render_seconds = time.perf_counter() - render_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    import matplotlib.pyplot as plt

    plt.close(result.artist)
    return {
        "nodes": 200,
        "visible_edges": len(prepared.edges),
        "vertices_per_hemisphere": len(geometry.meshes["left"].coordinates),
        "prepare_seconds": prepare_seconds,
        "static_render_seconds": static_render_seconds,
        "peak_extra_bytes": peak,
        "target_prepare_under_1s": prepare_seconds < 1.0,
        "target_render_under_20s": static_render_seconds < 20.0,
        "target_peak_under_1gb": peak < 1_073_741_824,
    }


def generate_artifacts(outdir: str | Path) -> dict[str, object]:
    """Generate every mandatory artifact from one immutable prepared network."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    # Benchmark before retaining several large figures so timing is not
    # distorted by renderer objects and browser-ready Plotly payloads.
    benchmark = _benchmark()
    geometry = make_geometry()
    prepared = _network(geometry, seed=20260831, max_edges=40)
    per_hemi = len(geometry.node_names) // 2
    overlay_node_values = np.sin(
        (np.arange(len(geometry.node_names)) % per_hemi) * (2.0 * np.pi / per_hemi)
        + 0.35
    )
    results = {}
    results["surface_matplotlib"] = plot_connectome(
        prepared,
        geometry,
        views="paper",
        depth_cue=True,
        title="Simulated signed connectivity — depth-aware context",
        figsize=(14.0, 5.5),
        dpi=250,
        output=(output / "surface_paper.png", output / "surface_paper.svg"),
    )
    full_subject_dir = Path(__file__).resolve().parents[1] / "data" / "fsaverage"
    full_geometry = make_full_fsaverage_geometry(full_subject_dir)
    plot_connectome(
        prepared,
        full_geometry,
        backend="surface",
        engine="matplotlib",
        style="paper",
        views="paper",
        depth_cue=True,
        title="User-provided full-resolution fsaverage — depth-aware pial context",
        figsize=(14.0, 5.5),
        dpi=250,
        output=output / "surface_fsaverage_full.png",
    )
    results["surface_nilearn"] = plot_connectome(
        prepared,
        full_geometry,
        backend="surface",
        engine="nilearn",
        surf_mesh=full_subject_dir,
        views=["lateral", "medial", "dorsal"],
        hemispheres=["left", "right"],
        bg_on_data=True,
        symmetric_cmap=None,
        symmetric_cbar="auto",
        inflate=False,
        depth_cue=True,
        title="Simulated connectivity — translucent native pial context",
        figsize=(10.0, 12.0),
        dpi=140,
        output=output / "surface_nilearn_native_three_views.png",
    )
    plot_connectome(
        prepared,
        geometry,
        views="whole",
        depth_cue=True,
        title="Simulated whole-brain depth-aware context",
        figsize=(12.0, 5.5),
        dpi=160,
        output=output / "surface_wholebrain.png",
    )
    plot_connectome(
        prepared,
        geometry,
        style="soft",
        node_overlay="gaussian",
        node_values=overlay_node_values,
        node_color_values=prepared.node_strength,
        node_size_values=prepared.node_strength,
        overlay_sigma_mm=12.0,
        overlay_radius_mm=30.0,
        views="paper",
        depth_cue=True,
        title="Simulated Gaussian visual interpolation",
        figsize=(12.0, 5.5),
        dpi=160,
        output=output / "surface_soft_overlay.png",
    )
    static_export: dict[str, object]
    try:
        results["surface_plotly"] = plot_connectome(
            prepared,
            geometry,
            backend="surface",
            engine="plotly",
            node_overlay="none",
            include_plotlyjs=True,
            static_views=("left", "right", "dorsal", "ventral"),
            image_width=1400,
            image_height=1000,
            output=(
                output / "surface_interactive.html",
                output / "surface_interactive.png",
            ),
        )
        static_export = {"renderer": "plotly-kaleido", "fallback": False}
    except RuntimeError as error:
        # HTML does not require a browser. Keep the Plotly backend under test,
        # then create the required multi-view review PNG from the exact same
        # PreparedConnectome when Kaleido's external Chrome cannot start.
        results["surface_plotly"] = plot_connectome(
            prepared,
            geometry,
            backend="surface",
            engine="plotly",
            node_overlay="none",
            include_plotlyjs=True,
            output=output / "surface_interactive.html",
        )
        fallback_result = plot_connectome(
            prepared,
            geometry,
            backend="surface",
            engine="matplotlib",
            style="paper",
            views="four",
            node_overlay="none",
            depth_cue=True,
            title="Interactive surface static review — browser-independent fallback",
            figsize=(14.0, 4.5),
            dpi=160,
            output=output / "surface_interactive.png",
        )
        static_export = {
            "renderer": "matplotlib-fallback",
            "fallback": True,
            "plotly_error_type": type(error).__name__,
            "same_prepared_instance": fallback_result.prepared is prepared,
        }
    results["glass"] = plot_connectome(
        prepared,
        geometry,
        backend="glass",
        title="Simulated MNI glass connectome",
        output=output / "glass.svg",
    )
    results["html"] = plot_connectome(
        prepared,
        geometry,
        backend="html",
        title="Simulated MNI connectome — Nilearn view_connectome",
        output=output / "nilearn_connectome.html",
    )
    results["circle"] = plot_connectome(
        prepared,
        geometry,
        backend="circle",
        title="Simulated connectivity circle",
        output=output / "circle.png",
    )
    directed = prepare_connectome(
        np.array([[0, 0.8, 0], [-0.3, 0, 0.4], [0, 0, 0]], float), directed=True
    )
    manifest = {
        "backends": {name: _edge_tuples(result) for name, result in results.items()},
        "surface_interactive_static_export": static_export,
        "directed_case": {
            "directed": directed.directed,
            "edges": [
                [edge.source, edge.target, edge.weight] for edge in directed.edges
            ],
        },
    }
    (output / "edge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output / "environment.txt").write_text(_environment_text(), encoding="utf-8")
    (output / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    return {
        "edge_count": len(prepared.edges),
        "benchmark": benchmark,
        "surface_interactive_static_export": static_export,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    report = generate_artifacts(args.outdir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
