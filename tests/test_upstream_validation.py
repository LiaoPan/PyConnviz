from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from base64 import b64encode
from html import escape
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pyconnviz import prepare_connectome
from scripts import validate_upstream_examples as validation_cli
from scripts.upstream_validation import (
    UPSTREAM_SOURCES,
    CheckResult,
    SourceSpec,
    build_report,
    compare_pngs,
    edge_oracle,
    environment_versions,
    extract_html_marker_coords,
    geometry_from_mni,
    make_labeled_montage,
    make_mne_connectivity,
    nearest_same_hemisphere_vertices,
    nilearn_html_edge_oracle,
    nilearn_static_edge_oracle,
    run_mne_case,
    run_mne_visual_cases,
    sha256_file,
    snapshot_sources,
    strength_oracle,
    validate_artifact,
    write_reports,
)


def test_edge_oracle_uses_unique_undirected_nonzero_pairs_and_absolute_percentile() -> None:
    matrix = np.array(
        [
            [0.0, -0.2, 0.9],
            [-0.2, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ]
    )

    edges, threshold = edge_oracle(matrix, percentile=50.0)

    assert threshold == pytest.approx(0.5)
    assert edges == ((0, 2, 0.9),)


def test_edge_oracle_rejects_invalid_matrix_and_accepts_all_zero_candidates() -> None:
    with pytest.raises(ValueError, match="square"):
        edge_oracle(np.zeros((2, 3)), percentile=80.0)
    edges, threshold = edge_oracle(np.zeros((2, 2)), percentile=80.0)
    assert edges == ()
    assert threshold == 0.0
    with pytest.raises(ValueError, match="between 0 and 100"):
        edge_oracle(np.eye(2), percentile=101.0)


def test_strength_oracle_sums_absolute_visible_undirected_weights() -> None:
    values = strength_oracle(((0, 2, -0.9), (1, 2, 0.5)), 3)

    np.testing.assert_allclose(values, [0.9, 0.5, 1.4])


def test_sha256_and_reports_are_deterministic(tmp_path: Path) -> None:
    payload = b"official upstream source\n"
    source = tmp_path / "source.py"
    source.write_bytes(payload)
    checks = [
        CheckResult(
            id="nilearn.edges",
            passed=True,
            expected="oracle edge tuples",
            observed="exact match",
            max_abs_error=0.0,
        ),
        CheckResult(
            id="mne.symmetric_dense_auto",
            passed=False,
            expected="undirected mirrored lower triangle",
            observed="directed lower-triangular matrix",
            max_abs_error=0.11,
            details={"workaround": 'symmetrize="lower"'},
            required=False,
        ),
    ]

    assert sha256_file(source) == hashlib.sha256(payload).hexdigest()
    report = build_report(checks, {"packages": {"nilearn": "0.14.0"}})
    json_path, markdown_path = write_reports(report, tmp_path / "reports")

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["overall_passed"] is True
    assert persisted["summary"] == {
        "all": {"failed": 1, "passed": 1, "total": 2},
        "diagnostic": {"failed": 1, "passed": 0, "total": 1},
        "required": {"failed": 0, "passed": 1, "total": 1},
    }
    assert persisted["checks"][1]["details"]["workaround"] == 'symmetrize="lower"'
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Environment" in markdown
    assert "## Required failures" in markdown
    assert "## Diagnostic observations" in markdown
    assert "mne.symmetric_dense_auto" in markdown
    assert "| DIFF | diagnostic | `mne.symmetric_dense_auto` | 0.11 |" in markdown
    assert "| FAIL | diagnostic |" not in markdown
    assert "Machine-readable expected/observed values are in `report.json`." in markdown
    assert json_path.read_text(encoding="utf-8").endswith("\n")


def test_required_failure_controls_overall_but_diagnostic_failure_does_not() -> None:
    diagnostic_only = build_report(
        [CheckResult("upstream.default", False, 148, 152, required=False)],
        {},
    )
    required_failure = build_report(
        [CheckResult("scientific.edges", False, 148, 149)],
        {},
    )

    assert diagnostic_only["overall_passed"] is True
    assert required_failure["overall_passed"] is False


def test_png_comparison_and_labeled_montage_are_machine_checkable(tmp_path: Path) -> None:
    pixels = np.zeros((20, 30, 3), dtype=np.uint8)
    pixels[5:15, 8:22] = [30, 120, 220]
    reference = tmp_path / "reference.png"
    identical = tmp_path / "identical.png"
    changed = tmp_path / "changed.png"
    Image.fromarray(pixels).save(reference)
    Image.fromarray(pixels).save(identical)
    changed_pixels = pixels.copy()
    changed_pixels[10, 10] = [255, 0, 0]
    Image.fromarray(changed_pixels).save(changed)

    exact = compare_pngs(reference, identical)
    different = compare_pngs(reference, changed)
    montage = make_labeled_montage(
        (reference, changed),
        ("Reference", "Candidate"),
        tmp_path / "montage.png",
    )

    assert exact["pixel_identical"] is True
    assert exact["max_abs_error"] == 0
    assert different["pixel_identical"] is False
    assert different["different_pixel_count"] == 1
    assert validate_artifact(montage, kind="png")["non_flat"] is True


def test_nearest_surface_projection_never_crosses_hemispheres() -> None:
    coords = np.array([[-9.0, 0.0, 0.0], [11.0, 0.0, 0.0]])
    hemispheres = ("left", "right")
    vertex_mni = {
        "left": np.array([[-10.0, 0.0, 0.0], [-20.0, 0.0, 0.0]]),
        "right": np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
    }

    vertices, distances = nearest_same_hemisphere_vertices(
        coords, hemispheres, vertex_mni
    )

    np.testing.assert_array_equal(vertices, [0, 0])
    np.testing.assert_allclose(distances, [1.0, 1.0])


def _mirrored_lower(connectivity) -> np.ndarray:
    dense = np.asarray(connectivity.get_data(output="dense"), dtype=np.float64)[:, :, 0]
    lower = np.tril(dense, -1)
    return lower + lower.T + np.diag(np.diag(dense))


def test_real_mne_coherence_explicit_lower_matches_independent_oracle() -> None:
    connectivity = make_mne_connectivity()
    expected = _mirrored_lower(connectivity)

    prepared = prepare_connectome(connectivity, symmetrize="lower")

    assert type(connectivity).__name__ == "SpectralConnectivity"
    assert prepared.directed is False
    assert prepared.node_names == ("A", "B", "C", "D")
    np.testing.assert_allclose(prepared.matrix, expected, rtol=0, atol=1e-12)


def test_real_mne_symmetric_metric_is_undirected_by_default() -> None:
    connectivity = make_mne_connectivity()
    expected = _mirrored_lower(connectivity)

    prepared = prepare_connectome(connectivity)

    assert prepared.directed is False
    np.testing.assert_allclose(prepared.matrix, expected, rtol=0, atol=1e-12)


def test_mne_audit_records_default_and_explicit_triangle_recovery_passes() -> None:
    checks, evidence = run_mne_case()
    by_id = {check.id: check for check in checks}

    assert evidence["object_type"] == "SpectralConnectivity"
    assert evidence["method"] == "coh"
    assert by_id["mne.public_contract"].passed is True
    assert by_id["mne.symmetric_dense_auto"].passed is True
    assert by_id["mne.auto_edges"].passed is True
    assert by_id["mne.auto_strength"].passed is True
    assert by_id["mne.explicit_lower_matrix"].passed is True
    assert by_id["mne.explicit_lower_edges"].passed is True
    assert by_id["mne.explicit_lower_strength"].passed is True


def test_mne_native_and_pyconnviz_circle_pair_uses_same_visible_edges(
    tmp_path: Path,
) -> None:
    checks, evidence = run_mne_visual_cases(tmp_path, methods=("coh",))
    by_id = {check.id: check for check in checks}
    case = evidence["coh"]

    assert by_id["mne.coh.circle_edge_identity"].passed is True
    assert by_id["mne.coh.circle_pixel_parity"].passed is True
    assert case["method"] == "coh"
    assert case["edge_count"] == 6
    for artifact in case["visual_artifacts"].values():
        assert Path(artifact["path"]).is_file()


def test_snapshot_sources_preserves_exact_bytes_and_provenance(tmp_path: Path) -> None:
    payload = b"# official example\nprint('real upstream')\n"
    source = tmp_path / "official.py"
    source.write_bytes(payload)
    spec = SourceSpec("snapshot.py", source.as_uri(), "nilearn")

    records = snapshot_sources(tmp_path / "evidence", sources=(spec,))

    snapshot = tmp_path / "evidence" / "upstream_sources" / "snapshot.py"
    assert snapshot.read_bytes() == payload
    assert records == [
        {
            "filename": "snapshot.py",
            "url": source.as_uri(),
            "package": "nilearn",
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "retrieved_at_utc": records[0]["retrieved_at_utc"],
        }
    ]
    assert records[0]["retrieved_at_utc"].endswith("Z")


def test_snapshot_sources_skip_download_requires_complete_cache(tmp_path: Path) -> None:
    payload = b"cached source\n"
    source = tmp_path / "official.py"
    source.write_bytes(payload)
    spec = SourceSpec("snapshot.py", source.as_uri(), "mne-connectivity")
    output = tmp_path / "evidence"
    snapshot_sources(output, sources=(spec,))

    cached = snapshot_sources(output, sources=(spec,), skip_download=True)

    assert cached[0]["sha256"] == hashlib.sha256(payload).hexdigest()
    (output / "upstream_sources" / "snapshot.py").unlink()
    with pytest.raises(FileNotFoundError, match=r"snapshot\.py"):
        snapshot_sources(output, sources=(spec,), skip_download=True)


def test_environment_versions_records_validation_stack() -> None:
    versions = environment_versions()

    assert set(versions) == {
        "python",
        "pyconnviz",
        "numpy",
        "nilearn",
        "mne",
        "mne-connectivity",
        "matplotlib",
    }
    assert versions["nilearn"] == "0.14.0"
    assert versions["mne-connectivity"] == "0.9.0"


def test_geometry_from_mni_preserves_labels_coordinates_and_hemispheres() -> None:
    labels = ["Left A", "Left B", "Right A", "Midline"]
    coords = np.array(
        [
            [-42.0, 10.0, 12.0],
            [-8.0, -20.0, 45.0],
            [38.0, 12.0, 8.0],
            [0.0, 30.0, -4.0],
        ]
    )

    geometry = geometry_from_mni(labels, coords)

    assert geometry.node_names == tuple(labels)
    assert geometry.hemispheres == ("left", "left", "right", "right")
    np.testing.assert_array_equal(geometry.mni_coords, coords)
    np.testing.assert_array_equal(geometry.surface_coords, coords)
    assert geometry.subject == "MSDL-MNI"
    assert geometry.surface_name == "not-surface-registered"
    assert set(geometry.meshes) == {"left", "right"}
    for mesh in geometry.meshes.values():
        assert mesh.coordinates.shape == (4, 3)
        assert mesh.faces.shape == (4, 3)


def test_validate_artifact_accepts_real_structures_and_rejects_flat_or_invalid(
    tmp_path: Path,
) -> None:
    pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    png = tmp_path / "figure.png"
    Image.fromarray(pixels).save(png)
    html = tmp_path / "connectome.html"
    html.write_text(
        '<iframe srcdoc="&lt;!DOCTYPE html&gt;&lt;html&gt;connectome&lt;/html&gt;"></iframe>',
        encoding="utf-8",
    )

    assert validate_artifact(png, kind="png")["non_flat"] is True
    assert validate_artifact(html, kind="html")["has_html"] is True

    Image.new("RGB", (16, 12), "white").save(png)
    with pytest.raises(ValueError, match="flat"):
        validate_artifact(png, kind="png")
    html.write_text("not a connectome document", encoding="utf-8")
    with pytest.raises(ValueError, match="HTML"):
        validate_artifact(html, kind="html")


def test_extract_html_marker_coords_decodes_nilearn_float32_payload() -> None:
    expected = np.array([[-42.5, 1.25, 9.0], [38.0, -2.0, 11.5]], dtype=np.float32)
    encoded = {
        axis: b64encode(expected[:, index].astype("<f4").tobytes()).decode("ascii")
        for index, axis in enumerate(("x", "y", "z"))
    }
    document = (
        '{"_marker_x": "'
        + encoded["x"]
        + '", "_marker_y": "'
        + encoded["y"]
        + '", "_marker_z": "'
        + encoded["z"]
        + '"}'
    )

    actual = extract_html_marker_coords(document, node_count=2)

    np.testing.assert_array_equal(actual, expected)


def test_extract_html_marker_coords_handles_saved_iframe_srcdoc() -> None:
    expected = np.array([[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    encoded = [
        b64encode(expected[:, index].astype("<f4").tobytes()).decode("ascii")
        for index in range(3)
    ]
    inner = (
        f'{{"_marker_x": "{encoded[0]}", "_marker_y": "{encoded[1]}", '
        f'"_marker_z": "{encoded[2]}"}}'
    )
    wrapped = f'<iframe srcdoc="{escape(inner, quote=True)}"></iframe>'

    actual = extract_html_marker_coords(wrapped, node_count=2)

    np.testing.assert_array_equal(actual, expected)


def test_nilearn_reference_oracles_capture_static_and_html_percentile_semantics() -> None:
    matrix = np.array(
        [
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.5],
            [0.9, 0.5, 0.0],
        ]
    )

    static_edges, static_threshold = nilearn_static_edge_oracle(matrix, percentile=50.0)
    html_edges, html_threshold = nilearn_html_edge_oracle(matrix, percentile=50.0)

    assert static_threshold == pytest.approx(0.50001)
    assert static_edges == ((0, 2, 0.9),)
    assert html_threshold == pytest.approx(0.10001)
    assert html_edges == ((0, 2, 0.9), (1, 2, 0.5))


def _write_cached_official_sources(outdir: Path) -> None:
    source_dir = outdir / "upstream_sources"
    source_dir.mkdir(parents=True)
    for spec in UPSTREAM_SOURCES:
        (source_dir / spec.filename).write_text(
            f"# cached official source: {spec.url}\n",
            encoding="utf-8",
        )


def test_validation_cli_non_strict_and_strict_exit_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts"
    _write_cached_official_sources(output)
    np.save(output / "nilearn_correlation_matrix.npy", np.zeros((2, 2)))
    np.save(output / "nilearn_mni_coords.npy", np.zeros((2, 3)))
    (output / "nilearn_labels.json").write_text('["A", "B"]\n', encoding="utf-8")
    passing = CheckResult("mne.real", True, "pass", "pass")
    failing = CheckResult("nilearn.native", False, "148 edges", "149 edges")
    monkeypatch.setattr(
        validation_cli.audit,
        "run_mne_case",
        lambda: ([passing], {"object_type": "SpectralConnectivity"}),
    )
    monkeypatch.setattr(
        validation_cli.audit,
        "run_mne_visual_cases",
        lambda outdir: ([], {"coh": {"edge_count": 1}}),
    )
    monkeypatch.setattr(
        validation_cli.audit,
        "run_nilearn_case",
        lambda data_dir, outdir, skip_download: (
            [failing],
            {"matrix_shape": [39, 39]},
        ),
    )
    monkeypatch.setattr(
        validation_cli.audit,
        "run_nilearn_power_case",
        lambda data_dir, outdir, skip_download: ([], {"edge_count": 1}),
    )
    monkeypatch.setattr(
        validation_cli.audit,
        "run_surface_case",
        lambda matrix, labels, coords, fsaverage_dir, outdir: ([], {"edge_count": 1}),
    )
    common = [
        "--data-dir",
        str(tmp_path / "data"),
        "--outdir",
        str(output),
        "--skip-download",
    ]

    assert validation_cli.main(common) == 0
    assert validation_cli.main([*common, "--strict"]) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert report["overall_passed"] is False
    assert report["summary"]["required"] == {"failed": 1, "passed": 1, "total": 2}
    assert len(provenance["upstream_sources"]) == 5
    assert provenance["cases"]["mne_numeric"]["object_type"] == "SpectralConnectivity"


def test_validation_cli_does_not_hide_unexpected_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts"
    _write_cached_official_sources(output)

    def fail() -> None:
        raise RuntimeError("computation failed")

    monkeypatch.setattr(validation_cli.audit, "run_mne_case", fail)

    with pytest.raises(RuntimeError, match="computation failed"):
        validation_cli.main(
            [
                "--data-dir",
                str(tmp_path / "data"),
                "--outdir",
                str(output),
                "--skip-download",
            ]
        )


def test_validation_cli_supports_direct_script_execution() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_upstream_examples.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--skip-download" in completed.stdout
    assert "--strict" in completed.stdout
