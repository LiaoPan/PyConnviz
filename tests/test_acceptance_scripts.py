from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.check_acceptance_artifacts import AcceptanceError, check_artifacts
from scripts.generate_acceptance_artifacts import (
    generate_artifacts,
    make_full_fsaverage_geometry,
    make_geometry,
)
from scripts.upstream_validation import run_surface_case

REQUIRED = (
    "surface_paper.png",
    "surface_paper.svg",
    "surface_wholebrain.png",
    "surface_soft_overlay.png",
    "surface_nilearn_native_three_views.png",
    "surface_interactive.html",
    "surface_interactive.png",
    "surface_fsaverage_full.png",
    "glass.svg",
    "nilearn_connectome.html",
    "circle.png",
    "edge_manifest.json",
    "environment.txt",
    "benchmark.json",
)


def write_valid_fixture(root: Path) -> None:
    root.mkdir()
    rng = np.random.default_rng(42)
    pixels = rng.integers(20, 230, size=(80, 120, 3), dtype=np.uint8)
    for name in (
        "surface_paper.png",
        "surface_wholebrain.png",
        "surface_soft_overlay.png",
        "surface_nilearn_native_three_views.png",
        "surface_interactive.png",
        "surface_fsaverage_full.png",
        "circle.png",
    ):
        Image.fromarray(pixels).save(root / name)
    (root / "surface_paper.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10"/></svg>',
        encoding="utf-8",
    )
    (root / "glass.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><circle r="2"/></svg>',
        encoding="utf-8",
    )
    plotly_html = (
        '<html><body><script>Plotly.newPlot("x", '
        '[{"name":"Nodes"},{"name":"Edges"}], '
        '{"meta":{"render_mode":"ball-and-stick"}})'
        "</script></body></html>"
    )
    generic_html = "<html><body><script>Plotly.newPlot('x', [])</script></body></html>"
    (root / "surface_interactive.html").write_text(plotly_html, encoding="utf-8")
    (root / "nilearn_connectome.html").write_text(generic_html, encoding="utf-8")
    edges = [[0, 1, 0.5], [1, 2, -0.2]]
    (root / "edge_manifest.json").write_text(
        json.dumps(
            {
                "backends": {
                    "surface_matplotlib": edges,
                    "surface_nilearn": edges,
                    "surface_plotly": edges,
                    "glass": edges,
                    "html": edges,
                    "circle": edges,
                },
                "directed_case": {
                    "directed": True,
                    "edges": [[0, 1, 0.8], [1, 0, -0.3]],
                },
                "static_surface_geometry": {
                    "surface_matplotlib": {
                        "render_mode": "ball-and-stick",
                        "renderer": "matplotlib-poly3d",
                        "panels": 3,
                        "node_collections": 3,
                        "edge_collections": 3,
                        "node_mesh_triangles": 100,
                        "edge_mesh_triangles": 100,
                        "node_diameter_range": [6.0, 16.0],
                    },
                    "surface_nilearn": {
                        "render_mode": "ball-and-stick",
                        "renderer": "nilearn-plot-img-on-surf+matplotlib-poly3d",
                        "panels": 6,
                        "node_collections": 6,
                        "edge_collections": 6,
                        "node_mesh_triangles": 100,
                        "edge_mesh_triangles": 100,
                        "node_diameter_range": [6.0, 16.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "environment.txt").write_text(
        "Python=3.13\nPyConnviz=0.1.0\nNiBabel=5.4\nNilearn=0.14\nMNE=1.12\n"
        "MNE-Connectivity=0.9\nMatplotlib=3.11\nPlotly=7.0\n",
        encoding="utf-8",
    )
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "nodes": 200,
                "visible_edges": 120,
                "vertices_per_hemisphere": 10000,
                "prepare_seconds": 0.1,
                "static_render_seconds": 1.0,
                "peak_extra_bytes": 1000000,
            }
        ),
        encoding="utf-8",
    )


def test_checker_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(AcceptanceError, match="Missing required"):
        check_artifacts(tmp_path, strict=False)


def test_checker_accepts_structurally_valid_fixture(tmp_path: Path) -> None:
    root = tmp_path / "valid"
    write_valid_fixture(root)
    report = check_artifacts(root, strict=False)
    assert report["files_checked"] == len(REQUIRED)
    assert report["backend_edge_count"] == 2


def test_checker_rejects_cross_backend_edge_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "mismatch"
    write_valid_fixture(root)
    path = root / "edge_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["backends"]["circle"] = [[0, 1, 9.0]]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AcceptanceError, match="edge tuples differ"):
        check_artifacts(root, strict=False)


def test_checker_rejects_legacy_static_surface_geometry(tmp_path: Path) -> None:
    root = tmp_path / "legacy-static"
    write_valid_fixture(root)
    path = root / "edge_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["static_surface_geometry"]["surface_matplotlib"]["render_mode"] = (
        "screen-space"
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AcceptanceError, match="ball-and-stick"):
        check_artifacts(root, strict=False)


def test_checker_rejects_flat_png_and_invalid_html(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    write_valid_fixture(root)
    Image.new("RGB", (80, 120), "white").save(root / "surface_paper.png")
    with pytest.raises(AcceptanceError, match="variance"):
        check_artifacts(root, strict=False)
    Image.fromarray(np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)).save(
        root / "surface_paper.png"
    )
    (root / "surface_interactive.html").write_text("not html", encoding="utf-8")
    with pytest.raises(AcceptanceError, match="HTML"):
        check_artifacts(root, strict=False)


def test_checker_accepts_nilearn_iframe_srcdoc_html(tmp_path: Path) -> None:
    root = tmp_path / "nilearn-iframe"
    write_valid_fixture(root)
    (root / "nilearn_connectome.html").write_text(
        '<iframe srcdoc="&lt;!DOCTYPE html&gt;&lt;html&gt;&lt;body&gt;'
        'Nilearn connectome&lt;/body&gt;&lt;/html&gt;"></iframe>',
        encoding="utf-8",
    )
    report = check_artifacts(root, strict=False)
    assert report["files_checked"] == len(REQUIRED)


def test_checker_requires_ball_and_stick_contract_for_plotly_surface_html(
    tmp_path: Path,
) -> None:
    root = tmp_path / "screen-space-plotly"
    write_valid_fixture(root)
    (root / "surface_interactive.html").write_text(
        "<html><body><script>Plotly.newPlot('x', [])</script></body></html>",
        encoding="utf-8",
    )

    with pytest.raises(AcceptanceError, match="ball-and-stick"):
        check_artifacts(root, strict=False)


def test_acceptance_geometry_is_bundled_fsaverage5_and_deterministic() -> None:
    first = make_geometry(node_count=20)
    second = make_geometry(node_count=20)

    assert first.subject == "nilearn-fsaverage5"
    assert first.surface_name == "pial"
    for hemi in ("left", "right"):
        mesh = first.meshes[hemi]
        assert mesh.name == f"nilearn-fsaverage5:{hemi}:pial"
        assert mesh.coordinates.shape == (10_242, 3)
        assert mesh.faces.shape == (20_480, 3)
        assert mesh.sulc.shape == (10_242,)
        assert np.all(np.isfinite(mesh.sulc))
        assert np.ptp(mesh.sulc) > 0
    np.testing.assert_array_equal(first.surface_coords, second.surface_coords)
    np.testing.assert_array_equal(first.node_vertices, second.node_vertices)
    assert first.mni_coords.shape == (20, 3)


def test_full_fsaverage_geometry_uses_user_pial_mesh_and_is_deterministic() -> None:
    subject_dir = Path(__file__).resolve().parents[1] / "data" / "fsaverage"
    first = make_full_fsaverage_geometry(subject_dir, node_count=20)
    second = make_full_fsaverage_geometry(subject_dir, node_count=20)

    assert first.subject == "user-fsaverage"
    assert first.surface_name == "pial"
    assert first.node_names == make_geometry(node_count=20).node_names
    for hemi in ("left", "right"):
        mesh = first.meshes[hemi]
        assert mesh.name == f"user-fsaverage:{hemi}:pial"
        assert mesh.coordinates.shape == (163_842, 3)
        assert mesh.faces.shape == (327_680, 3)
        assert mesh.sulc.shape == (163_842,)
        assert np.all(np.isfinite(mesh.sulc))
        assert np.ptp(mesh.sulc) > 0
    np.testing.assert_array_equal(first.surface_coords, second.surface_coords)
    np.testing.assert_array_equal(first.node_vertices, second.node_vertices)


def test_generator_contract_keeps_native_connectivity_surface_neutral() -> None:
    source = inspect.getsource(generate_artifacts)

    assert "load_sample_motor_activation_image" not in source
    assert "stat_map=" not in source


def test_generator_contract_emits_a_static_plotly_surface_image() -> None:
    source = inspect.getsource(generate_artifacts)

    assert 'output / "surface_interactive.html"' in source
    assert 'output / "surface_interactive.png"' in source
    assert 'static_views=("left", "right", "dorsal", "ventral"),' in source


def test_generator_contract_refreshes_full_fsaverage_with_paper_renderer() -> None:
    source = inspect.getsource(generate_artifacts)
    start = source.index("    plot_connectome(\n        prepared,\n        full_geometry,")
    end = source.index('    results["surface_nilearn"]')
    full_call = source[start:end]

    assert 'output=output / "surface_fsaverage_full.png"' in full_call
    assert 'engine="matplotlib"' in full_call
    assert 'style="paper"' in full_call
    assert 'views="paper"' in full_call


def test_generator_contract_enables_depth_cue_for_static_surface_artifacts() -> None:
    source = inspect.getsource(generate_artifacts)
    output_markers = (
        'output / "surface_paper.png"',
        'output / "surface_fsaverage_full.png"',
        'output / "surface_nilearn_native_three_views.png"',
        'output / "surface_wholebrain.png"',
        'output / "surface_soft_overlay.png"',
    )

    for marker in output_markers:
        marker_index = source.index(marker)
        call_start = source.rfind("plot_connectome(", 0, marker_index)
        assert "depth_cue=True" in source[call_start:marker_index], marker


def test_upstream_surface_static_calls_use_fixed_style_visibility_budget() -> None:
    source = inspect.getsource(run_surface_case)
    matplotlib_call = source[
        source.index("matplotlib_result = plot_connectome(") :
        source.index("native_result = plot_connectome(")
    ]
    native_call = source[
        source.index("native_result = plot_connectome(") :
        source.index("interactive_geometry =")
    ]

    for call in (matplotlib_call, native_call):
        assert 'style="paper"' in call
        assert "depth_cue=True" in call
        assert "cortex_alpha=" not in call
        assert "edge_alpha=" not in call
        assert "edge_width_range=" not in call


def test_generated_acceptance_set_passes_strict_checker(tmp_path: Path) -> None:
    root = tmp_path / "acceptance"
    generate_artifacts(root)
    report = check_artifacts(root, strict=True)
    assert report["files_checked"] == len(REQUIRED)
