"""Independent numerical helpers for validating real upstream examples."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from base64 import b64decode
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

import numpy as np

EdgeTuple = tuple[int, int, float]
StrengthMode = Literal["total", "in", "out"]


@dataclass(frozen=True)
class CheckResult:
    """One explicit scientific or provenance assertion."""

    id: str
    passed: bool
    expected: object
    observed: object
    max_abs_error: float | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    required: bool = True

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("check id must be non-empty")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if self.max_abs_error is not None and (
            not np.isfinite(self.max_abs_error) or self.max_abs_error < 0
        ):
            raise ValueError("max_abs_error must be finite, non-negative, or None")


@dataclass(frozen=True)
class SourceSpec:
    """One official example source to preserve as audit evidence."""

    filename: str
    url: str
    package: str


UPSTREAM_SOURCES = (
    SourceSpec(
        "nilearn_plot_probabilistic_atlas_extraction.py",
        "https://nilearn.github.io/stable/_downloads/"
        "c4ffee04091cacf1483ed5d324ce9c8e/plot_probabilistic_atlas_extraction.py",
        "nilearn",
    ),
    SourceSpec(
        "nilearn_plot_sphere_based_connectome.py",
        "https://nilearn.github.io/stable/_downloads/"
        "cb064f9ac4369c8f91e970b8f1ae7965/plot_sphere_based_connectome.py",
        "nilearn",
    ),
    SourceSpec(
        "mne_connectivity_connectivity_classes.py",
        "https://mne.tools/mne-connectivity/stable/_downloads/"
        "980054063d6073a3d7942ac239432571/connectivity_classes.py",
        "mne-connectivity",
    ),
    SourceSpec(
        "mne_connectivity_compare_coherency_methods.py",
        "https://mne.tools/mne-connectivity/stable/_downloads/"
        "584a0ffc2ce7de5433ac678cbf086b83/compare_coherency_methods.py",
        "mne-connectivity",
    ),
    SourceSpec(
        "mne_connectivity_sensor_connectivity.py",
        "https://mne.tools/mne-connectivity/stable/_downloads/"
        "c97704849885b49e703ad1a2fa96a3dd/sensor_connectivity.py",
        "mne-connectivity",
    ),
)


def _utc_timestamp(value: float | None = None) -> str:
    moment = (
        datetime.now(timezone.utc)
        if value is None
        else datetime.fromtimestamp(value, timezone.utc)
    )
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def snapshot_sources(
    outdir: str | Path,
    *,
    sources: Sequence[SourceSpec] = UPSTREAM_SOURCES,
    skip_download: bool = False,
) -> list[dict[str, object]]:
    """Retrieve exact official example sources and return their provenance."""

    source_dir = Path(outdir) / "upstream_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for spec in sources:
        destination = source_dir / spec.filename
        if skip_download:
            if not destination.is_file() or destination.stat().st_size == 0:
                raise FileNotFoundError(
                    f"cached upstream source is missing or empty: {destination}"
                )
            retrieved_at = _utc_timestamp(destination.stat().st_mtime)
        else:
            request = Request(
                spec.url,
                headers={"User-Agent": "PyConnviz-upstream-validation/0.1"},
            )
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError(f"official source download is empty: {spec.url}")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            retrieved_at = _utc_timestamp()
        records.append(
            {
                "filename": spec.filename,
                "url": spec.url,
                "package": spec.package,
                "byte_count": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "retrieved_at_utc": retrieved_at,
            }
        )
    return records


def environment_versions() -> dict[str, str]:
    """Record the complete package stack that controls the audit."""

    package_names = (
        "pyconnviz",
        "numpy",
        "nilearn",
        "mne",
        "mne-connectivity",
        "matplotlib",
    )
    versions = {"python": platform.python_version()}
    for package in package_names:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package != "pyconnviz":
                raise
            from pyconnviz import __version__

            versions[package] = __version__
    return versions


def geometry_from_mni(labels: Sequence[object], coords: Any):
    """Build glass/HTML-only geometry while preserving exact MNI coordinates."""

    from pyconnviz import HemisphereMesh, geometry_from_arrays

    names = tuple(str(label) for label in labels)
    coordinates = np.asarray(coords, dtype=np.float64)
    if not names or coordinates.shape != (len(names), 3):
        raise ValueError(
            f"coords must have shape ({len(names)}, 3) for the supplied non-empty labels"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coords must contain only finite MNI values")
    base = np.array(
        [
            [0.0, 0.0, 8.0],
            [-5.0, -4.0, -4.0],
            [5.0, -4.0, -4.0],
            [0.0, 6.0, -4.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    meshes = {
        "left": HemisphereMesh(base + np.array([-55.0, 0.0, 0.0]), faces, name="placeholder:left"),
        "right": HemisphereMesh(
            base + np.array([55.0, 0.0, 0.0]), faces, name="placeholder:right"
        ),
    }
    hemispheres = tuple("left" if x < 0 else "right" for x in coordinates[:, 0])
    return geometry_from_arrays(
        names,
        coordinates,
        hemispheres,
        meshes,
        mni_coords=coordinates,
        subject="MSDL-MNI",
        surface_name="not-surface-registered",
    )


def nearest_same_hemisphere_vertices(
    mni_coords: Any,
    hemispheres: Sequence[str],
    vertex_mni: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Find nearest surface vertices while enforcing the supplied hemisphere."""

    from scipy.spatial import cKDTree

    coordinates = np.asarray(mni_coords, dtype=np.float64)
    resolved_hemispheres = tuple(hemispheres)
    if coordinates.shape != (len(resolved_hemispheres), 3):
        raise ValueError(
            "mni_coords must have one three-dimensional coordinate per hemisphere entry"
        )
    if any(hemi not in {"left", "right"} for hemi in resolved_hemispheres):
        raise ValueError("hemispheres must contain only 'left' or 'right'")
    trees = {}
    for hemi in ("left", "right"):
        values = np.asarray(vertex_mni[hemi], dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            raise ValueError(f"vertex_mni[{hemi!r}] must have shape (n_vertices, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"vertex_mni[{hemi!r}] must contain finite coordinates")
        trees[hemi] = cKDTree(values)
    vertices = np.empty(len(coordinates), dtype=np.int64)
    distances = np.empty(len(coordinates), dtype=np.float64)
    for hemi in ("left", "right"):
        selected = np.flatnonzero(
            np.asarray([value == hemi for value in resolved_hemispheres], dtype=bool)
        )
        if not len(selected):
            continue
        selected_distances, selected_vertices = trees[hemi].query(coordinates[selected])
        vertices[selected] = np.asarray(selected_vertices, dtype=np.int64)
        distances[selected] = np.asarray(selected_distances, dtype=np.float64)
    return vertices, distances


def validate_artifact(path: str | Path, *, kind: Literal["png", "html"]) -> dict[str, object]:
    """Reject empty or structurally meaningless validation artifacts."""

    artifact = Path(path)
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError(f"artifact is missing or empty: {artifact}")
    if kind == "png":
        from PIL import Image

        with Image.open(artifact) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
            width, height = image.size
        variance = float(np.var(pixels))
        if variance <= 0:
            raise ValueError(f"PNG artifact is flat: {artifact}")
        return {
            "path": str(artifact),
            "byte_count": artifact.stat().st_size,
            "width": width,
            "height": height,
            "variance": variance,
            "non_flat": True,
        }
    if kind == "html":
        document = artifact.read_text(encoding="utf-8")
        normalized = document.lower()
        has_html = "<html" in normalized or "&lt;html" in normalized
        has_payload = any(token in normalized for token in ("<iframe", "<script", "connectome"))
        if not has_html or not has_payload:
            raise ValueError(f"artifact is not a valid connectome HTML document: {artifact}")
        return {
            "path": str(artifact),
            "byte_count": artifact.stat().st_size,
            "has_html": True,
            "has_payload": True,
        }
    raise ValueError("kind must be 'png' or 'html'")


def compare_pngs(reference: str | Path, candidate: str | Path) -> dict[str, object]:
    """Compare rendered PNG pixels without treating metadata bytes as image content."""

    from PIL import Image

    reference_path = Path(reference)
    candidate_path = Path(candidate)
    with Image.open(reference_path) as image:
        reference_pixels = np.asarray(image.convert("RGBA"), dtype=np.int16)
        reference_size = image.size
    with Image.open(candidate_path) as image:
        candidate_pixels = np.asarray(image.convert("RGBA"), dtype=np.int16)
        candidate_size = image.size
    same_shape = reference_pixels.shape == candidate_pixels.shape
    if not same_shape:
        return {
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "reference_size": list(reference_size),
            "candidate_size": list(candidate_size),
            "pixel_identical": False,
            "different_pixel_count": None,
            "max_abs_error": None,
            "mean_abs_error": None,
        }
    absolute = np.abs(reference_pixels - candidate_pixels)
    changed = np.any(absolute != 0, axis=2)
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_size": list(reference_size),
        "candidate_size": list(candidate_size),
        "pixel_identical": bool(not np.any(changed)),
        "different_pixel_count": int(np.count_nonzero(changed)),
        "max_abs_error": int(np.max(absolute)) if absolute.size else 0,
        "mean_abs_error": float(np.mean(absolute)) if absolute.size else 0.0,
    }


def make_labeled_montage(
    images: Sequence[str | Path],
    labels: Sequence[str],
    output: str | Path,
    *,
    columns: int | None = None,
) -> Path:
    """Place complete source figures in a deterministic labeled comparison grid."""

    from PIL import Image, ImageDraw, ImageFont

    paths = tuple(Path(path) for path in images)
    captions = tuple(str(label) for label in labels)
    if not paths or len(paths) != len(captions):
        raise ValueError("images and labels must be non-empty sequences of equal length")
    if columns is None:
        columns = min(2, len(paths))
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    loaded: list[Any] = []
    try:
        for path in paths:
            image = Image.open(path).convert("RGB")
            image.load()
            loaded.append(image)
        cell_width = max(image.width for image in loaded)
        font_size = max(18, min(48, cell_width // 35))
        label_height = font_size + 28
        rows = (len(loaded) + columns - 1) // columns
        row_heights = [
            max(
                image.height
                for image in loaded[row * columns : min((row + 1) * columns, len(loaded))]
            )
            for row in range(rows)
        ]
        row_offsets = [0]
        for height in row_heights[:-1]:
            row_offsets.append(row_offsets[-1] + height + label_height)
        canvas = Image.new(
            "RGB",
            (cell_width * columns, sum(row_heights) + label_height * rows),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=font_size)
        for index, (image, caption) in enumerate(zip(loaded, captions, strict=True)):
            row, column = divmod(index, columns)
            x0 = column * cell_width
            y0 = row_offsets[row]
            text_box = draw.textbbox((0, 0), caption, font=font)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (x0 + max((cell_width - text_width) // 2, 4), y0 + 12),
                caption,
                fill="black",
                font=font,
            )
            canvas.paste(
                image,
                (
                    x0 + (cell_width - image.width) // 2,
                    y0 + label_height + (row_heights[row] - image.height) // 2,
                ),
            )
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
    finally:
        for image in loaded:
            image.close()
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"montage was not created: {destination}")
    return destination


def extract_html_marker_coords(document: str, *, node_count: int) -> np.ndarray:
    """Decode the public artifact's Nilearn marker coordinates for QA."""

    document = unescape(document)
    axes = []
    for axis in ("x", "y", "z"):
        match = re.search(rf'"_marker_{axis}"\s*:\s*"([A-Za-z0-9+/=]+)"', document)
        if match is None:
            raise ValueError(f"Nilearn HTML is missing _marker_{axis}")
        values = np.frombuffer(b64decode(match.group(1)), dtype="<f4")
        if values.shape != (node_count,):
            raise ValueError(
                f"Nilearn HTML _marker_{axis} has {len(values)} values; expected {node_count}"
            )
        axes.append(values)
    return np.column_stack(axes)


def _require_cached_nilearn_inputs(data_dir: Path) -> None:
    requirements = {
        "MSDL atlas": ("*msdl_rois.nii*",),
        "development fMRI": ("*bold.nii*",),
        "development confounds": ("*Confounds*.tsv", "*confounds*.tsv"),
    }
    missing = []
    for label, patterns in requirements.items():
        if not any(path.is_file() for pattern in patterns for path in data_dir.rglob(pattern)):
            missing.append(label)
    if missing:
        raise FileNotFoundError(
            "--skip-download requires cached Nilearn inputs; missing: " + ", ".join(missing)
        )


def _path_provenance(path: str | Path) -> dict[str, object]:
    value = Path(path)
    return {
        "path": str(value),
        "byte_count": value.stat().st_size,
        "sha256": sha256_file(value),
    }


def run_nilearn_case(
    data_dir: str | Path,
    outdir: str | Path,
    *,
    skip_download: bool = False,
) -> tuple[list[CheckResult], dict[str, object]]:
    """Run and audit the official one-subject MSDL connectome example."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from nilearn.connectome import ConnectivityMeasure
    from nilearn.datasets import fetch_atlas_msdl, fetch_development_fmri
    from nilearn.maskers import NiftiMapsMasker
    from nilearn.plotting import plot_connectome as nilearn_plot_connectome
    from nilearn.plotting import view_connectome

    from pyconnviz import plot_connectome, prepare_connectome
    from pyconnviz.plotting._visual import edge_visuals, node_visuals

    cache = Path(data_dir)
    output = Path(outdir)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if skip_download:
        _require_cached_nilearn_inputs(cache)

    atlas = fetch_atlas_msdl(data_dir=cache)
    data = fetch_development_fmri(n_subjects=1, data_dir=cache)
    labels = tuple(str(label) for label in atlas["labels"])
    masker = NiftiMapsMasker(
        maps_img=atlas["maps"],
        standardize_confounds=True,
        memory=output / "nilearn_cache",
        memory_level=1,
        verbose=1,
    )
    time_series = np.asarray(
        masker.fit_transform(data.func[0], confounds=data.confounds),
        dtype=np.float64,
    )
    correlation = ConnectivityMeasure(kind="correlation", verbose=1)
    matrix = np.asarray(correlation.fit_transform([time_series])[0], dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    coords = np.asarray(atlas.region_coords, dtype=np.float64)
    geometry = geometry_from_mni(labels, coords)

    matrix_path = output / "nilearn_correlation_matrix.npy"
    coords_path = output / "nilearn_mni_coords.npy"
    labels_path = output / "nilearn_labels.json"
    np.save(matrix_path, matrix, allow_pickle=False)
    np.save(coords_path, coords, allow_pickle=False)
    labels_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reference_png = output / "nilearn_reference_connectome.png"
    reference_html = output / "nilearn_reference_connectome.html"
    matched_reference_png = output / "nilearn_matched_connectome.png"
    matched_reference_html = output / "nilearn_matched_connectome.html"
    pyconnviz_png = output / "pyconnviz_nilearn_example_glass.png"
    pyconnviz_html = output / "pyconnviz_nilearn_example.html"
    comparison_png = output / "nilearn_msdl_glass_comparison.png"
    nilearn_plot_connectome(
        matrix,
        coords,
        edge_threshold="80%",
        title="Nilearn official MSDL example — top 20%",
        output_file=reference_png,
    )
    native_view = view_connectome(matrix, coords, edge_threshold="80%")
    native_view.save_as_html(reference_html)

    prepared = prepare_connectome(
        matrix,
        geometry=geometry,
        directed=False,
        edge_threshold="80%",
    )
    static_node_colors, static_node_sizes = node_visuals(
        prepared,
        style="paper",
        size_range=(25.0, 90.0),
    )
    edge_cmap, edge_vmin, edge_vmax, signed = edge_visuals(prepared)
    nilearn_plot_connectome(
        np.asarray(prepared.visible_matrix),
        coords,
        node_color=static_node_colors,
        node_size=static_node_sizes,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_threshold=None,
        output_file=matched_reference_png,
        display_mode="lyrz",
        title=None,
        black_bg=False,
        colorbar=bool(prepared.edges),
    )
    html_node_colors, html_node_sizes = node_visuals(
        prepared,
        style="paper",
        size_range=(3.0, 7.0),
    )
    matched_view = view_connectome(
        np.asarray(prepared.visible_matrix),
        coords,
        edge_threshold=None,
        edge_cmap=edge_cmap,
        symmetric_cmap=signed,
        node_color=html_node_colors,
        node_size=float(np.mean(html_node_sizes)) if len(html_node_sizes) else 3.0,
        colorbar=bool(prepared.edges),
        title=None,
        node_labels=list(prepared.node_names),
    )
    matched_view.save_as_html(matched_reference_html)
    glass_result = plot_connectome(
        prepared,
        geometry,
        backend="glass",
        style="paper",
        display_mode="lyrz",
        title=None,
        output=pyconnviz_png,
    )
    html_result = plot_connectome(
        prepared,
        geometry,
        backend="html",
        style="paper",
        title=None,
        output=pyconnviz_html,
    )
    pixel_comparison = compare_pngs(matched_reference_png, pyconnviz_png)
    make_labeled_montage(
        (matched_reference_png, pyconnviz_png),
        ("Nilearn direct (matched input)", "PyConnviz glass (same input)"),
        comparison_png,
    )

    actual_edges = _prepared_edge_tuples(prepared)
    scientific_edges, scientific_threshold = edge_oracle(matrix, percentile=80.0)
    static_edges, static_threshold = nilearn_static_edge_oracle(matrix, percentile=80.0)
    native_html_edges, native_html_threshold = nilearn_html_edge_oracle(
        matrix, percentile=80.0
    )
    expected_strength = strength_oracle(scientific_edges, len(labels))
    decoded_coords = extract_html_marker_coords(
        pyconnviz_html.read_text(encoding="utf-8"),
        node_count=len(labels),
    )
    artifact_rows = {
        "nilearn_reference_png": validate_artifact(reference_png, kind="png"),
        "nilearn_reference_html": validate_artifact(reference_html, kind="html"),
        "nilearn_matched_png": validate_artifact(matched_reference_png, kind="png"),
        "nilearn_matched_html": validate_artifact(matched_reference_html, kind="html"),
        "pyconnviz_glass_png": validate_artifact(pyconnviz_png, kind="png"),
        "pyconnviz_html": validate_artifact(pyconnviz_html, kind="html"),
        "comparison_png": validate_artifact(comparison_png, kind="png"),
    }
    shape_passed = bool(
        time_series.shape == (168, 39)
        and matrix.shape == (39, 39)
        and coords.shape == (39, 3)
        and len(labels) == 39
        and np.all(np.isfinite(time_series))
        and np.all(np.isfinite(matrix))
        and np.all(np.isfinite(coords))
    )
    coordinates_error = float(np.max(np.abs(decoded_coords.astype(float) - coords)))
    backend_identity = bool(
        glass_result.prepared is prepared
        and html_result.prepared is prepared
        and _prepared_edge_tuples(glass_result.prepared) == actual_edges
        and _prepared_edge_tuples(html_result.prepared) == actual_edges
    )
    checks = [
        CheckResult(
            id="nilearn.real_data_pipeline",
            passed=shape_passed,
            expected={
                "time_series_shape": [168, 39],
                "matrix_shape": [39, 39],
                "coords_shape": [39, 3],
                "finite": True,
            },
            observed={
                "time_series_shape": list(time_series.shape),
                "matrix_shape": list(matrix.shape),
                "coords_shape": list(coords.shape),
                "finite": bool(
                    np.all(np.isfinite(time_series))
                    and np.all(np.isfinite(matrix))
                    and np.all(np.isfinite(coords))
                ),
            },
        ),
        _array_check(
            "nilearn.matrix_preserved",
            prepared.matrix,
            matrix,
            expected_description="exact official 39x39 correlation matrix with zero diagonal",
            observed_description="PyConnviz prepared matrix",
        ),
        CheckResult(
            id="nilearn.node_order",
            passed=prepared.node_names == labels,
            expected=list(labels),
            observed=list(prepared.node_names),
        ),
        CheckResult(
            id="nilearn.mni_coordinates",
            passed=bool(
                np.array_equal(geometry.mni_coords, coords)
                and np.allclose(decoded_coords, coords, rtol=0.0, atol=1e-5)
            ),
            expected="atlas region_coords preserved and encoded into PyConnviz HTML",
            observed="decoded Nilearn HTML marker coordinates",
            max_abs_error=coordinates_error,
            details={"html_storage_dtype": "little-endian float32", "atol": 1e-5},
        ),
        CheckResult(
            id="nilearn.scientific_top20_edges",
            passed=actual_edges == scientific_edges,
            expected=[list(edge) for edge in scientific_edges],
            observed=[list(edge) for edge in actual_edges],
            details={
                "candidate_count": len(labels) * (len(labels) - 1) // 2,
                "selected_count": len(actual_edges),
                "threshold": scientific_threshold,
                "definition": (
                    "all finite unique dense pairs define percentile; zeros are not drawn"
                ),
            },
        ),
        CheckResult(
            id="nilearn.native_static_edges",
            passed=actual_edges == static_edges,
            expected=[list(edge) for edge in static_edges],
            observed=[list(edge) for edge in actual_edges],
            details={
                "nilearn_selected_count": len(static_edges),
                "pyconnviz_selected_count": len(actual_edges),
                "nilearn_threshold": static_threshold,
                "semantics": "lower triangle; strict above score-at-percentile",
            },
        ),
        CheckResult(
            id="nilearn.native_html_edges",
            passed=actual_edges == native_html_edges,
            expected=[list(edge) for edge in native_html_edges],
            observed=[list(edge) for edge in actual_edges],
            details={
                "nilearn_selected_count": len(native_html_edges),
                "pyconnviz_selected_count": len(actual_edges),
                "nilearn_threshold": native_html_threshold,
                "semantics": "full dense matrix including diagonal and duplicate triangles",
            },
            required=False,
        ),
        _array_check(
            "nilearn.node_strength",
            prepared.node_strength,
            expected_strength,
            expected_description="row sum of absolute visible undirected matrix",
            observed_description="PyConnviz node strength",
        ),
        CheckResult(
            id="nilearn.backend_edge_identity",
            passed=backend_identity,
            expected="glass and HTML receive the same PreparedConnectome instance and edge tuple",
            observed={
                "same_instance_glass": glass_result.prepared is prepared,
                "same_instance_html": html_result.prepared is prepared,
                "edge_count": len(actual_edges),
            },
        ),
        CheckResult(
            id="nilearn.matched_static_pixels",
            passed=bool(pixel_comparison["pixel_identical"]),
            expected="pixel-identical direct Nilearn and PyConnviz glass renders",
            observed=pixel_comparison,
            max_abs_error=(
                None
                if pixel_comparison["max_abs_error"] is None
                else float(pixel_comparison["max_abs_error"])
            ),
            details={
                "comparison_scope": (
                    "same visible matrix, MNI coordinates, node visuals, edge visuals, "
                    "display mode, and no secondary threshold"
                )
            },
        ),
        CheckResult(
            id="nilearn.artifacts",
            passed=True,
            expected="matched/default references, PyConnviz outputs, and montage are valid",
            observed=artifact_rows,
        ),
    ]
    evidence: dict[str, object] = {
        "atlas": "MSDL",
        "subject_count": 1,
        "functional_image": _path_provenance(data.func[0]),
        "confounds": _path_provenance(data.confounds[0]),
        "atlas_maps": _path_provenance(atlas["maps"]),
        "time_series_shape": list(time_series.shape),
        "matrix_shape": list(matrix.shape),
        "coordinates_shape": list(coords.shape),
        "labels": labels,
        "scientific_threshold": scientific_threshold,
        "nilearn_static_threshold": static_threshold,
        "nilearn_html_threshold": native_html_threshold,
        "pyconnviz_edge_count": len(actual_edges),
        "nilearn_static_edge_count": len(static_edges),
        "nilearn_html_edge_count": len(native_html_edges),
        "numeric_artifacts": {
            "correlation_matrix": _path_provenance(matrix_path),
            "mni_coords": _path_provenance(coords_path),
            "labels": _path_provenance(labels_path),
        },
        "visual_artifacts": artifact_rows,
        "surface_accuracy_claimed": False,
    }
    return checks, evidence


def run_nilearn_power_case(
    data_dir: str | Path,
    outdir: str | Path,
    *,
    skip_download: bool = False,
) -> tuple[list[CheckResult], dict[str, object]]:
    """Run the official Power-264 sphere extraction and matched glass comparison."""

    import warnings

    import matplotlib

    matplotlib.use("Agg", force=True)
    from nilearn.datasets import fetch_coords_power_2011, fetch_development_fmri
    from nilearn.maskers import NiftiSpheresMasker
    from nilearn.plotting import plot_connectome as nilearn_plot_connectome
    from sklearn.covariance import GraphicalLassoCV
    from sklearn.exceptions import ConvergenceWarning

    from pyconnviz import plot_connectome, prepare_connectome
    from pyconnviz.plotting._visual import edge_visuals, node_visuals

    cache = Path(data_dir)
    output = Path(outdir) / "comparisons" / "nilearn_power264"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if skip_download:
        _require_cached_nilearn_inputs(cache)

    dataset = fetch_development_fmri(n_subjects=1, data_dir=cache)
    power = fetch_coords_power_2011()
    coords = np.vstack(
        (power.rois["x"], power.rois["y"], power.rois["z"])
    ).T.astype(np.float64)
    labels = tuple(f"Power-{int(roi):03d}" for roi in power.rois["roi"])
    masker = NiftiSpheresMasker(
        seeds=coords,
        smoothing_fwhm=6,
        radius=5.0,
        detrend=True,
        standardize_confounds=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=dataset.t_r,
        memory=output / "nilearn_cache",
        memory_level=1,
        verbose=0,
    )
    time_series = np.asarray(
        masker.fit_transform(dataset.func[0], confounds=dataset.confounds[0]),
        dtype=np.float64,
    )
    covariance_estimator = GraphicalLassoCV(cv=3, verbose=0, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        covariance_estimator.fit(time_series)
    convergence_warnings = [
        str(item.message)
        for item in caught_warnings
        if issubclass(item.category, ConvergenceWarning)
    ]
    matrix = np.asarray(covariance_estimator.covariance_, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    geometry = geometry_from_mni(labels, coords)
    prepared = prepare_connectome(
        matrix,
        geometry=geometry,
        directed=False,
        edge_threshold="99.8%",
    )

    matrix_path = output / "power_covariance_matrix.npy"
    coords_path = output / "power_mni_coords.npy"
    timeseries_path = output / "power_time_series.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    np.save(coords_path, coords, allow_pickle=False)
    np.save(timeseries_path, time_series, allow_pickle=False)

    native_default_png = output / "nilearn_native_default_glass.png"
    native_matched_png = output / "nilearn_native_matched_glass.png"
    pyconnviz_png = output / "pyconnviz_glass.png"
    pyconnviz_html = output / "pyconnviz_connectome.html"
    comparison_png = output / "glass_comparison.png"
    manifest_path = output / "manifest.json"
    nilearn_plot_connectome(
        matrix,
        coords,
        edge_threshold="99.8%",
        node_size=20,
        title="Nilearn official Power-264 defaults",
        output_file=native_default_png,
    )
    node_colors, node_sizes = node_visuals(
        prepared,
        style="paper",
        size_range=(25.0, 90.0),
    )
    edge_cmap, edge_vmin, edge_vmax, _ = edge_visuals(prepared)
    nilearn_plot_connectome(
        np.asarray(prepared.visible_matrix),
        coords,
        node_color=node_colors,
        node_size=node_sizes,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        edge_threshold=None,
        output_file=native_matched_png,
        display_mode="lyrz",
        title=None,
        black_bg=False,
        colorbar=bool(prepared.edges),
    )
    glass_result = plot_connectome(
        prepared,
        geometry,
        backend="glass",
        style="paper",
        display_mode="lyrz",
        title=None,
        output=pyconnviz_png,
    )
    html_result = plot_connectome(
        prepared,
        geometry,
        backend="html",
        style="paper",
        title="PyConnviz — Power-264",
        output=pyconnviz_html,
    )
    pixel_comparison = compare_pngs(native_matched_png, pyconnviz_png)
    make_labeled_montage(
        (native_matched_png, pyconnviz_png),
        ("Nilearn direct - Power-264", "PyConnviz - Power-264"),
        comparison_png,
    )

    expected_edges, threshold = edge_oracle(matrix, percentile=99.8)
    static_edges, static_threshold = nilearn_static_edge_oracle(
        matrix, percentile=99.8
    )
    actual_edges = _prepared_edge_tuples(prepared)
    expected_strength = strength_oracle(expected_edges, len(labels))
    artifact_rows = {
        "nilearn_native_default": validate_artifact(native_default_png, kind="png"),
        "nilearn_native_matched": validate_artifact(native_matched_png, kind="png"),
        "pyconnviz_glass": validate_artifact(pyconnviz_png, kind="png"),
        "pyconnviz_html": validate_artifact(pyconnviz_html, kind="html"),
        "comparison": validate_artifact(comparison_png, kind="png"),
    }
    manifest = {
        "schema_version": 1,
        "case": "nilearn_power264",
        "time_series_shape": list(time_series.shape),
        "matrix_shape": list(matrix.shape),
        "coordinate_shape": list(coords.shape),
        "threshold_percentile": 99.8,
        "resolved_threshold": threshold,
        "nilearn_static_threshold": static_threshold,
        "expected_edges": [list(edge) for edge in expected_edges],
        "pyconnviz_edges": [list(edge) for edge in actual_edges],
        "pixel_comparison": pixel_comparison,
        "visual_artifacts": artifact_rows,
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    finite = bool(
        np.all(np.isfinite(time_series))
        and np.all(np.isfinite(matrix))
        and np.all(np.isfinite(coords))
    )
    checks = [
        CheckResult(
            id="nilearn.power.real_data_pipeline",
            passed=bool(
                time_series.shape == (168, 264)
                and matrix.shape == (264, 264)
                and coords.shape == (264, 3)
                and finite
            ),
            expected={
                "time_series_shape": [168, 264],
                "matrix_shape": [264, 264],
                "coords_shape": [264, 3],
                "finite": True,
            },
            observed={
                "time_series_shape": list(time_series.shape),
                "matrix_shape": list(matrix.shape),
                "coords_shape": list(coords.shape),
                "finite": finite,
            },
        ),
        _array_check(
            "nilearn.power.matrix_preserved",
            prepared.matrix,
            matrix,
            expected_description="Power-264 GraphicalLasso covariance with zero diagonal",
            observed_description="PyConnviz prepared matrix",
        ),
        CheckResult(
            id="nilearn.power.percentile_edges",
            passed=actual_edges == expected_edges,
            expected=[list(edge) for edge in expected_edges],
            observed=[list(edge) for edge in actual_edges],
            details={
                "definition": (
                    "all finite unique dense pairs define percentile; zeros are not drawn"
                ),
                "threshold": threshold,
                "selected_count": len(actual_edges),
            },
        ),
        CheckResult(
            id="nilearn.power.native_static_edges",
            passed=actual_edges == static_edges,
            expected=[list(edge) for edge in static_edges],
            observed=[list(edge) for edge in actual_edges],
            details={"nilearn_threshold": static_threshold},
        ),
        CheckResult(
            id="nilearn.power.estimator_convergence",
            passed=not convergence_warnings,
            expected="GraphicalLassoCV completes without convergence warnings",
            observed=(
                convergence_warnings
                if convergence_warnings
                else "converged without warning"
            ),
            details={
                "scope": (
                    "upstream estimator diagnostic; edge/render parity remains exact for the "
                    "returned public covariance_ matrix"
                )
            },
            required=False,
        ),
        _array_check(
            "nilearn.power.node_strength",
            prepared.node_strength,
            expected_strength,
            expected_description="independent absolute visible-edge strength",
            observed_description="PyConnviz node strength",
        ),
        CheckResult(
            id="nilearn.power.backend_edge_identity",
            passed=bool(
                glass_result.prepared is prepared
                and html_result.prepared is prepared
                and _prepared_edge_tuples(glass_result.prepared) == actual_edges
                and _prepared_edge_tuples(html_result.prepared) == actual_edges
            ),
            expected="glass and HTML share the same prepared Power-264 edge set",
            observed={"edge_count": len(actual_edges)},
        ),
        CheckResult(
            id="nilearn.power.matched_static_pixels",
            passed=bool(pixel_comparison["pixel_identical"]),
            expected="pixel-identical direct Nilearn and PyConnviz glass renders",
            observed=pixel_comparison,
            max_abs_error=(
                None
                if pixel_comparison["max_abs_error"] is None
                else float(pixel_comparison["max_abs_error"])
            ),
        ),
        CheckResult(
            id="nilearn.power.artifacts",
            passed=True,
            expected="valid Power-264 native/PyConnviz figures, HTML, and montage",
            observed=artifact_rows,
        ),
    ]
    evidence: dict[str, object] = {
        "atlas": "Power-264 coordinates",
        "subject_count": 1,
        "estimator": "GraphicalLassoCV(cv=3)",
        "estimator_convergence_warnings": convergence_warnings,
        "time_series_shape": list(time_series.shape),
        "matrix_shape": list(matrix.shape),
        "edge_count": len(actual_edges),
        "resolved_threshold": threshold,
        "nilearn_static_threshold": static_threshold,
        "numeric_artifacts": {
            "time_series": _path_provenance(timeseries_path),
            "covariance_matrix": _path_provenance(matrix_path),
            "mni_coords": _path_provenance(coords_path),
        },
        "visual_artifacts": artifact_rows,
        "manifest": _path_provenance(manifest_path),
    }
    return checks, evidence


def geometry_from_fsaverage_projection(
    labels: Sequence[object],
    coords: Any,
    fsaverage_dir: str | Path,
    *,
    max_distance_mm: float = 10.0,
):
    """Project MNI centres to same-hemisphere full fsaverage pial vertices."""

    import mne
    from nilearn.surface import load_surf_data, load_surf_mesh

    from pyconnviz import HemisphereMesh, geometry_from_arrays

    names = tuple(str(label) for label in labels)
    coordinates = np.asarray(coords, dtype=np.float64)
    if coordinates.shape != (len(names), 3):
        raise ValueError(f"coords must have shape ({len(names)}, 3)")
    subject_dir = Path(fsaverage_dir).resolve()
    subject = subject_dir.name
    subjects_dir = subject_dir.parent
    meshes = {}
    vertex_mni = {}
    for hemi, prefix, mne_hemi in (
        ("left", "lh", 0),
        ("right", "rh", 1),
    ):
        pial_path = subject_dir / "surf" / f"{prefix}.pial"
        sulc_path = subject_dir / "surf" / f"{prefix}.sulc"
        pial = load_surf_mesh(pial_path)
        sulc = load_surf_data(sulc_path)
        meshes[hemi] = HemisphereMesh(
            np.asarray(pial.coordinates, dtype=np.float64),
            np.asarray(pial.faces, dtype=np.int64),
            np.asarray(sulc, dtype=np.float64),
            name=f"{subject}:{prefix}.pial",
        )
        vertex_mni[hemi] = np.asarray(
            mne.vertex_to_mni(
                np.arange(len(pial.coordinates), dtype=np.int64),
                mne_hemi,
                subject,
                subjects_dir=subjects_dir,
            ),
            dtype=np.float64,
        )
    hemispheres = tuple("left" if coord[0] < 0 else "right" for coord in coordinates)
    vertices, distances = nearest_same_hemisphere_vertices(
        coordinates, hemispheres, vertex_mni
    )
    included = np.flatnonzero(distances <= float(max_distance_mm)).astype(np.int64)
    if not len(included):
        raise ValueError("no nodes fall within max_distance_mm of fsaverage pial")
    included_hemispheres = tuple(hemispheres[index] for index in included)
    included_vertices = vertices[included]
    surface_coords = np.asarray(
        [
            meshes[hemi].coordinates[int(vertex)]
            for hemi, vertex in zip(
                included_hemispheres, included_vertices, strict=True
            )
        ],
        dtype=np.float64,
    )
    geometry = geometry_from_arrays(
        tuple(names[index] for index in included),
        surface_coords,
        included_hemispheres,
        meshes,
        mni_coords=coordinates[included],
        node_vertices=included_vertices,
        subject=subject,
        surface_name="pial",
    )
    records = [
        {
            "input_index": index,
            "label": names[index],
            "hemisphere": hemispheres[index],
            "nearest_vertex": int(vertices[index]),
            "distance_mm": float(distances[index]),
            "included": bool(distances[index] <= float(max_distance_mm)),
        }
        for index in range(len(names))
    ]
    return geometry, included, records


def _fsaverage5_display_geometry(
    node_names: Sequence[str],
    mni_coords: Any,
    hemispheres: Sequence[str],
):
    """Build a downsampled Nilearn fsaverage5 display mesh for interactive HTML."""

    from nilearn.datasets import load_fsaverage, load_fsaverage_data

    from pyconnviz import HemisphereMesh, geometry_from_arrays

    fsaverage = load_fsaverage("fsaverage5")
    sulcal = load_fsaverage_data(
        mesh="fsaverage5", mesh_type="pial", data_type="sulcal"
    )
    meshes = {}
    vertex_coordinates = {}
    for hemi in ("left", "right"):
        part = fsaverage.pial.parts[hemi]
        meshes[hemi] = HemisphereMesh(
            np.asarray(part.coordinates, dtype=np.float64),
            np.asarray(part.faces, dtype=np.int64),
            np.asarray(sulcal.data.parts[hemi], dtype=np.float64),
            name=f"nilearn-fsaverage5:{hemi}:pial",
        )
        vertex_coordinates[hemi] = meshes[hemi].coordinates
    coords = np.asarray(mni_coords, dtype=np.float64)
    vertices, _ = nearest_same_hemisphere_vertices(
        coords, hemispheres, vertex_coordinates
    )
    surface_coords = np.asarray(
        [
            meshes[hemi].coordinates[int(vertex)]
            for hemi, vertex in zip(hemispheres, vertices, strict=True)
        ],
        dtype=np.float64,
    )
    return geometry_from_arrays(
        tuple(node_names),
        surface_coords,
        tuple(hemispheres),
        meshes,
        mni_coords=coords,
        node_vertices=vertices,
        subject="nilearn-fsaverage5",
        surface_name="pial",
    )


def run_surface_case(
    matrix: Any,
    labels: Sequence[object],
    coords: Any,
    fsaverage_dir: str | Path,
    outdir: str | Path,
    *,
    max_projection_distance_mm: float = 10.0,
    max_edges: int = 40,
) -> tuple[list[CheckResult], dict[str, object]]:
    """Render one real MSDL network on the user-provided fsaverage surface."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    from pyconnviz import plot_connectome, prepare_connectome

    values = np.asarray(matrix, dtype=np.float64)
    coordinates = np.asarray(coords, dtype=np.float64)
    names = tuple(str(label) for label in labels)
    if values.shape != (len(names), len(names)):
        raise ValueError("matrix shape must match labels")
    geometry, included, projection_records = geometry_from_fsaverage_projection(
        names,
        coordinates,
        fsaverage_dir,
        max_distance_mm=max_projection_distance_mm,
    )
    subset = values[np.ix_(included, included)]
    prepared = prepare_connectome(
        subset,
        geometry=geometry,
        directed=False,
        max_edges=max_edges,
    )
    expected_edges = all_edge_oracle(subset)[:max_edges]
    actual_edges = _prepared_edge_tuples(prepared)
    expected_strength = strength_oracle(expected_edges, len(included))

    output = Path(outdir) / "comparisons" / "surface_msdl_fsaverage"
    output.mkdir(parents=True, exist_ok=True)
    matplotlib_png = output / "pyconnviz_surface_paper.png"
    native_png = output / "pyconnviz_surface_nilearn_native.png"
    interactive_html = output / "pyconnviz_surface_interactive.html"
    comparison_png = output / "surface_comparison.png"
    projection_path = output / "projection_manifest.json"
    manifest_path = output / "manifest.json"

    matplotlib_result = plot_connectome(
        prepared,
        geometry,
        backend="surface",
        engine="matplotlib",
        style="paper",
        views="paper",
        node_overlay="none",
        depth_cue=True,
        title="MSDL connectivity — translucent depth-aware pial context",
        figsize=(13.0, 5.0),
        dpi=140,
        rasterize_surface=True,
        output=matplotlib_png,
        show=False,
    )
    native_result = plot_connectome(
        prepared,
        geometry,
        backend="surface",
        engine="nilearn",
        surf_mesh=Path(fsaverage_dir),
        style="paper",
        views=("lateral", "medial", "dorsal"),
        hemispheres=("left", "right"),
        bg_on_data=True,
        symmetric_cmap=None,
        symmetric_cbar="auto",
        inflate=False,
        depth_cue=True,
        title="MSDL connectivity — translucent native Nilearn context",
        figsize=(10.0, 12.0),
        dpi=120,
        rasterize_surface=True,
        output=native_png,
        show=False,
    )
    interactive_geometry = _fsaverage5_display_geometry(
        geometry.node_names,
        geometry.mni_coords,
        geometry.hemispheres,
    )
    interactive_result = plot_connectome(
        prepared,
        interactive_geometry,
        backend="surface",
        engine="plotly",
        style="paper",
        node_overlay="none",
        cortex_alpha=0.26,
        include_plotlyjs=True,
        output=interactive_html,
    )
    plt.close(matplotlib_result.artist)
    plt.close(native_result.artist)
    make_labeled_montage(
        (matplotlib_png, native_png),
        (
            "PyConnviz depth-aware context - 3 views",
            "PyConnviz + Nilearn translucent context - 6 panels",
        ),
        comparison_png,
        columns=1,
    )
    projection_path.write_text(
        json.dumps(projection_records, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_rows = {
        "surface_paper": validate_artifact(matplotlib_png, kind="png"),
        "surface_nilearn_native": validate_artifact(native_png, kind="png"),
        "surface_interactive": validate_artifact(interactive_html, kind="html"),
        "comparison": validate_artifact(comparison_png, kind="png"),
    }
    included_records = [record for record in projection_records if record["included"]]
    excluded_records = [record for record in projection_records if not record["included"]]
    max_included_distance = max(float(record["distance_mm"]) for record in included_records)
    backend_identity = bool(
        matplotlib_result.prepared is prepared
        and native_result.prepared is prepared
        and interactive_result.prepared is prepared
        and _prepared_edge_tuples(matplotlib_result.prepared) == actual_edges
        and _prepared_edge_tuples(native_result.prepared) == actual_edges
        and _prepared_edge_tuples(interactive_result.prepared) == actual_edges
    )
    manifest = {
        "schema_version": 1,
        "case": "surface_msdl_fsaverage",
        "input_node_count": len(names),
        "included_node_count": len(included),
        "excluded_nodes": excluded_records,
        "max_projection_distance_mm": max_projection_distance_mm,
        "max_included_distance_mm": max_included_distance,
        "max_edges": max_edges,
        "expected_edges": [list(edge) for edge in expected_edges],
        "pyconnviz_edges": [list(edge) for edge in actual_edges],
        "visual_artifacts": artifact_rows,
    }
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    checks = [
        CheckResult(
            id="surface.fsaverage_projection",
            passed=bool(
                len(included) >= int(np.floor(0.9 * len(names)))
                and max_included_distance <= max_projection_distance_mm
                and len(excluded_records) == len(names) - len(included)
            ),
            expected={
                "at_least_90_percent_nodes_within_mm": max_projection_distance_mm,
                "same_hemisphere_only": True,
                "excluded_nodes_recorded": True,
            },
            observed={
                "input_node_count": len(names),
                "included_node_count": len(included),
                "excluded_nodes": excluded_records,
                "max_included_distance_mm": max_included_distance,
            },
            max_abs_error=max_included_distance,
            details={
                "scope": (
                    "nearest-pial fsaverage visualization; not individual cortical registration"
                )
            },
        ),
        _array_check(
            "surface.matrix_subset_preserved",
            prepared.matrix,
            subset,
            expected_description="MSDL matrix subset for included cortical nodes",
            observed_description="PyConnviz surface prepared matrix",
        ),
        CheckResult(
            id="surface.top40_edges",
            passed=actual_edges == expected_edges,
            expected=[list(edge) for edge in expected_edges],
            observed=[list(edge) for edge in actual_edges],
        ),
        _array_check(
            "surface.node_strength",
            prepared.node_strength,
            expected_strength,
            expected_description="independent surface visible-edge strength",
            observed_description="PyConnviz surface node strength",
        ),
        CheckResult(
            id="surface.backend_edge_identity",
            passed=backend_identity,
            expected="Matplotlib, Nilearn native, and Plotly HTML share one edge set",
            observed={
                "same_prepared_instance": backend_identity,
                "edge_count": len(actual_edges),
            },
        ),
        CheckResult(
            id="surface.artifacts",
            passed=True,
            expected="valid multi-view surface PNGs, interactive HTML, and montage",
            observed=artifact_rows,
        ),
    ]
    evidence: dict[str, object] = {
        "input_atlas": "MSDL",
        "surface_subject_dir": str(Path(fsaverage_dir).resolve()),
        "projection": "same-hemisphere nearest fsaverage pial vertex in MNI space",
        "projection_scope": "visualization, not individual registration truth",
        "input_node_count": len(names),
        "included_node_count": len(included),
        "excluded_nodes": excluded_records,
        "edge_count": len(actual_edges),
        "projection_manifest": _path_provenance(projection_path),
        "manifest": _path_provenance(manifest_path),
        "visual_artifacts": artifact_rows,
    }
    return checks, evidence


def edge_oracle(
    matrix: Any,
    *,
    percentile: float,
    directed: bool = False,
) -> tuple[tuple[EdgeTuple, ...], float]:
    """Select percentile-thresholded edges without calling PyConnviz code."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"matrix must be square; got shape {values.shape}")
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if directed:
        pairs = (
            (source, target)
            for source in range(len(values))
            for target in range(len(values))
            if source != target
        )
    else:
        pairs = (
            (source, target)
            for source in range(len(values))
            for target in range(source + 1, len(values))
        )
    candidates = [
        (source, target, float(values[source, target]))
        for source, target in pairs
        if np.isfinite(values[source, target])
    ]
    if not candidates:
        raise ValueError("matrix has no finite candidate edges")
    threshold = float(np.percentile([abs(weight) for _, _, weight in candidates], percentile))
    selected = tuple(
        sorted(
            (edge for edge in candidates if edge[2] != 0 and abs(edge[2]) > threshold),
            key=lambda edge: (-abs(edge[2]), edge[0], edge[1]),
        )
    )
    return selected, threshold


def all_edge_oracle(
    matrix: Any,
    *,
    directed: bool = False,
) -> tuple[EdgeTuple, ...]:
    """Return every finite non-zero candidate edge in PyConnviz sort order."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"matrix must be square; got shape {values.shape}")
    if directed:
        pairs = (
            (source, target)
            for source in range(len(values))
            for target in range(len(values))
            if source != target
        )
    else:
        pairs = (
            (source, target)
            for source in range(len(values))
            for target in range(source + 1, len(values))
        )
    return tuple(
        sorted(
            (
                (source, target, float(values[source, target]))
                for source, target in pairs
                if np.isfinite(values[source, target]) and values[source, target] != 0
            ),
            key=lambda edge: (-abs(edge[2]), edge[0], edge[1]),
        )
    )


def _require_symmetric_matrix(matrix: Any) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"matrix must be square; got shape {values.shape}")
    if not np.allclose(values, values.T, rtol=1e-7, atol=1e-9, equal_nan=False):
        raise ValueError("Nilearn reference oracle requires a symmetric matrix")
    return values


def nilearn_static_edge_oracle(
    matrix: Any,
    *,
    percentile: float,
) -> tuple[tuple[EdgeTuple, ...], float]:
    """Reproduce Nilearn 0.14 static symmetric graph threshold semantics."""

    values = _require_symmetric_matrix(matrix)
    lower = values[np.tril_indices_from(values, k=-1)]
    threshold = float(np.percentile(np.abs(lower), percentile)) + 1e-5
    edges = tuple(
        sorted(
            (
                (source, target, float(values[source, target]))
                for source in range(len(values))
                for target in range(source + 1, len(values))
                if np.isfinite(values[source, target])
                and values[source, target] != 0
                and abs(values[source, target]) >= threshold
            ),
            key=lambda edge: (-abs(edge[2]), edge[0], edge[1]),
        )
    )
    return edges, threshold


def nilearn_html_edge_oracle(
    matrix: Any,
    *,
    percentile: float,
) -> tuple[tuple[EdgeTuple, ...], float]:
    """Reproduce Nilearn 0.14 view_connectome dense-percentile semantics."""

    values = _require_symmetric_matrix(matrix)
    magnitudes = np.abs(values).ravel()
    index = int(magnitudes.size * 0.01 * float(percentile))
    threshold = float(np.partition(magnitudes, index)[index]) + 1e-5
    edges = tuple(
        sorted(
            (
                (source, target, float(values[source, target]))
                for source in range(len(values))
                for target in range(source + 1, len(values))
                if np.isfinite(values[source, target])
                and values[source, target] != 0
                and abs(values[source, target]) > threshold
            ),
            key=lambda edge: (-abs(edge[2]), edge[0], edge[1]),
        )
    )
    return edges, threshold


def strength_oracle(
    edges: Sequence[EdgeTuple],
    node_count: int,
    *,
    directed: bool = False,
    mode: StrengthMode = "total",
) -> np.ndarray:
    """Compute absolute visible-edge strength independently of PyConnviz."""

    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise ValueError("node_count must be a positive integer")
    if mode not in {"total", "in", "out"}:
        raise ValueError("mode must be 'total', 'in', or 'out'")
    if not directed and mode != "total":
        raise ValueError("undirected strength supports only mode='total'")
    incoming = np.zeros(node_count, dtype=np.float64)
    outgoing = np.zeros(node_count, dtype=np.float64)
    for source, target, weight in edges:
        if source == target or not 0 <= source < node_count or not 0 <= target < node_count:
            raise ValueError(f"invalid edge indices: {(source, target)}")
        magnitude = abs(float(weight))
        if not np.isfinite(magnitude):
            raise ValueError("edge weights must be finite")
        outgoing[source] += magnitude
        incoming[target] += magnitude
        if not directed:
            outgoing[target] += magnitude
    if not directed:
        return outgoing
    if mode == "in":
        return incoming
    if mode == "out":
        return outgoing
    return incoming + outgoing


def make_mne_connectivity(*, method: str = "coh", seed: int = 7):
    """Compute a deterministic real MNE-Connectivity coherence object."""

    from mne_connectivity import spectral_connectivity_epochs

    rng = np.random.default_rng(seed)
    epochs = rng.standard_normal((8, 4, 256))
    return spectral_connectivity_epochs(
        epochs,
        names=["A", "B", "C", "D"],
        method=method,
        mode="multitaper",
        sfreq=128.0,
        fmin=8.0,
        fmax=13.0,
        faverage=True,
        n_jobs=1,
        verbose=False,
    )


def _prepared_edge_tuples(prepared: Any) -> tuple[EdgeTuple, ...]:
    return tuple(
        (int(edge.source), int(edge.target), float(edge.weight)) for edge in prepared.edges
    )


def _array_check(
    check_id: str,
    actual: Any,
    expected: Any,
    *,
    expected_description: str,
    observed_description: str,
    atol: float = 1e-12,
) -> CheckResult:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    same_shape = actual_array.shape == expected_array.shape
    error = (
        float(np.max(np.abs(actual_array - expected_array)))
        if same_shape and actual_array.size
        else (0.0 if same_shape else None)
    )
    passed = bool(
        same_shape
        and np.allclose(actual_array, expected_array, rtol=0.0, atol=atol, equal_nan=False)
    )
    return CheckResult(
        id=check_id,
        passed=passed,
        expected=expected_description,
        observed=observed_description,
        max_abs_error=error,
        details={
            "expected_shape": list(expected_array.shape),
            "observed_shape": list(actual_array.shape),
            "atol": atol,
        },
    )


def run_mne_case() -> tuple[list[CheckResult], dict[str, object]]:
    """Audit PyConnviz against an object computed by MNE-Connectivity itself."""

    from pyconnviz import prepare_connectome

    connectivity = make_mne_connectivity()
    dense = np.asarray(connectivity.get_data(output="dense"), dtype=np.float64)
    matrix = dense[:, :, 0]
    lower = np.tril(matrix, -1)
    expected_matrix = lower + lower.T + np.diag(np.diag(matrix))
    expected_edges = all_edge_oracle(expected_matrix)
    expected_strength = strength_oracle(expected_edges, int(connectivity.n_nodes))

    prepared_auto = prepare_connectome(connectivity)
    prepared_none = prepare_connectome(connectivity, symmetrize="none")
    prepared_lower = prepare_connectome(connectivity, symmetrize="lower")
    auto_edges = _prepared_edge_tuples(prepared_auto)
    lower_edges = _prepared_edge_tuples(prepared_lower)
    attrs = getattr(connectivity, "attrs", {})
    symmetric_attribute = getattr(connectivity, "symmetric", None)
    symmetric_metadata = attrs.get("symmetric") if isinstance(attrs, Mapping) else None

    public_contract = bool(
        type(connectivity).__name__ == "SpectralConnectivity"
        and int(connectivity.n_nodes) == 4
        and tuple(connectivity.names) == ("A", "B", "C", "D")
        and tuple(connectivity.dims) == ("node_in -> node_out", "freqs")
        and dense.shape == (4, 4, 1)
        and np.asarray(connectivity.freqs).shape == (1,)
        and np.isclose(float(connectivity.freqs[0]), 10.5)
    )
    checks = [
        CheckResult(
            id="mne.public_contract",
            passed=public_contract,
            expected="real SpectralConnectivity with 4 named nodes and one 8-13 Hz average",
            observed={
                "type": type(connectivity).__name__,
                "n_nodes": int(connectivity.n_nodes),
                "names": list(connectivity.names),
                "dims": list(connectivity.dims),
                "dense_shape": list(dense.shape),
                "freqs": list(connectivity.freqs),
            },
        ),
        _array_check(
            "mne.dense_values_read",
            prepared_none.matrix,
            matrix,
            expected_description="PyConnviz matrix equals public dense singleton-frequency slice",
            observed_description='PyConnviz symmetrize="none" matrix before triangle recovery',
        ),
        CheckResult(
            id="mne.symmetric_dense_auto",
            passed=bool(
                not prepared_auto.directed
                and np.allclose(prepared_auto.matrix, expected_matrix, rtol=0.0, atol=1e-12)
            ),
            expected="undirected mirrored lower triangle",
            observed=(
                "undirected mirrored lower triangle"
                if not prepared_auto.directed
                else "directed lower-triangular matrix"
            ),
            max_abs_error=float(np.max(np.abs(prepared_auto.matrix - expected_matrix))),
            details={
                "connectivity.symmetric": symmetric_attribute,
                "connectivity.attrs.symmetric": symmetric_metadata,
                "prepared_directed": prepared_auto.directed,
                "symmetry_basis": prepared_auto.metadata.get("symmetry_basis"),
            },
        ),
        CheckResult(
            id="mne.auto_edges",
            passed=auto_edges == expected_edges,
            expected=[list(edge) for edge in expected_edges],
            observed=[list(edge) for edge in auto_edges],
            details={"orientation_is_scientifically_significant": True},
        ),
        _array_check(
            "mne.auto_strength",
            prepared_auto.node_strength,
            expected_strength,
            expected_description="absolute undirected strength from mirrored coherence",
            observed_description="PyConnviz default total strength",
        ),
        _array_check(
            "mne.explicit_lower_matrix",
            prepared_lower.matrix,
            expected_matrix,
            expected_description="mirrored public lower triangle",
            observed_description='PyConnviz symmetrize="lower" matrix',
        ),
        CheckResult(
            id="mne.explicit_lower_edges",
            passed=lower_edges == expected_edges,
            expected=[list(edge) for edge in expected_edges],
            observed=[list(edge) for edge in lower_edges],
        ),
        _array_check(
            "mne.explicit_lower_strength",
            prepared_lower.node_strength,
            expected_strength,
            expected_description="independent absolute undirected strength",
            observed_description='PyConnviz symmetrize="lower" strength',
        ),
    ]
    evidence: dict[str, object] = {
        "object_type": type(connectivity).__name__,
        "method": str(connectivity.method),
        "mode": "multitaper",
        "sfreq_hz": 128.0,
        "frequency_band_hz": [8.0, 13.0],
        "frequency_average_hz": list(connectivity.freqs),
        "epochs_shape": [8, 4, 256],
        "raveled_shape": list(connectivity.shape),
        "dense_shape": list(dense.shape),
        "dense_matrix": matrix,
        "expected_symmetric_matrix": expected_matrix,
        "upstream_symmetric_attribute": symmetric_attribute,
        "upstream_symmetric_metadata": symmetric_metadata,
    }
    return checks, evidence


def _circle_geometry(node_names: Sequence[str]):
    """Build minimal ordered geometry for the coordinate-free circle backend."""

    from pyconnviz import HemisphereMesh, geometry_from_arrays

    count = len(node_names)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    surface_coords = np.column_stack(
        (40.0 * np.cos(angles), 40.0 * np.sin(angles), np.zeros(count))
    )
    hemispheres = tuple("left" if index < count / 2 else "right" for index in range(count))
    base = np.array(
        [[0.0, 0.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
        dtype=np.int64,
    )
    meshes = {
        "left": HemisphereMesh(base - np.array([50.0, 0.0, 0.0]), faces),
        "right": HemisphereMesh(base + np.array([50.0, 0.0, 0.0]), faces),
    }
    return geometry_from_arrays(tuple(node_names), surface_coords, hemispheres, meshes)


def run_mne_visual_cases(
    outdir: str | Path,
    *,
    methods: Sequence[str] = ("coh", "plv"),
) -> tuple[list[CheckResult], dict[str, object]]:
    """Render matched native/PyConnviz MNE circle pairs for real upstream objects."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from mne_connectivity.viz import plot_connectivity_circle

    from pyconnviz import plot_connectome, prepare_connectome
    from pyconnviz.plotting._visual import edge_visuals, node_visuals

    output = Path(outdir)
    checks: list[CheckResult] = []
    evidence: dict[str, object] = {}
    for index, method_value in enumerate(methods):
        method = str(method_value).lower()
        connectivity = make_mne_connectivity(method=method, seed=7 + index)
        raw_dense = np.asarray(connectivity.get_data(output="dense"), dtype=np.float64)
        raw_matrix = raw_dense[:, :, 0]
        lower = np.tril(raw_matrix, -1)
        expected_matrix = lower + lower.T + np.diag(np.diag(raw_matrix))
        expected_edges = all_edge_oracle(expected_matrix)
        prepared = prepare_connectome(connectivity)
        actual_edges = _prepared_edge_tuples(prepared)
        geometry = _circle_geometry(prepared.node_names)

        case_dir = output / "comparisons" / f"mne_{method}"
        case_dir.mkdir(parents=True, exist_ok=True)
        native_png = case_dir / "mne_native_circle.png"
        pyconnviz_png = case_dir / "pyconnviz_circle.png"
        comparison_png = case_dir / "circle_comparison.png"
        manifest_path = case_dir / "manifest.json"

        colors, _ = node_visuals(
            prepared,
            style="paper",
            size_range=(1.0, 1.0),
        )
        cmap, vmin, vmax, _ = edge_visuals(prepared)
        figure, _ = plot_connectivity_circle(
            raw_matrix,
            list(prepared.node_names),
            n_lines=None,
            node_colors=colors,
            facecolor="white",
            textcolor="black",
            node_edgecolor="black",
            colormap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=bool(prepared.edges),
            title=None,
            interactive=False,
            show=False,
        )
        figure.savefig(native_png, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        pyconnviz_result = plot_connectome(
            prepared,
            geometry,
            backend="circle",
            style="paper",
            title=None,
            colorbar=True,
            output=pyconnviz_png,
            show=False,
        )
        plt.close(pyconnviz_result.artist[0])
        pixel_comparison = compare_pngs(native_png, pyconnviz_png)
        make_labeled_montage(
            (native_png, pyconnviz_png),
            (f"MNE direct - {method}", f"PyConnviz - {method}"),
            comparison_png,
        )
        visual_artifacts = {
            "mne_native_circle": validate_artifact(native_png, kind="png"),
            "pyconnviz_circle": validate_artifact(pyconnviz_png, kind="png"),
            "comparison": validate_artifact(comparison_png, kind="png"),
        }
        manifest = {
            "schema_version": 1,
            "case": f"mne_{method}",
            "method": method,
            "raw_dense_shape": list(raw_dense.shape),
            "prepared_directed": prepared.directed,
            "symmetry_basis": prepared.metadata.get("symmetry_basis"),
            "expected_edges": [list(edge) for edge in expected_edges],
            "pyconnviz_edges": [list(edge) for edge in actual_edges],
            "pixel_comparison": pixel_comparison,
            "visual_artifacts": visual_artifacts,
        }
        manifest_path.write_text(
            json.dumps(_jsonable(manifest), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        checks.extend(
            [
                _array_check(
                    f"mne.{method}.auto_symmetric_matrix",
                    prepared.matrix,
                    expected_matrix,
                    expected_description="mirrored public all-to-all lower triangle",
                    observed_description="PyConnviz automatic matrix",
                ),
                CheckResult(
                    id=f"mne.{method}.circle_edge_identity",
                    passed=actual_edges == expected_edges,
                    expected=[list(edge) for edge in expected_edges],
                    observed=[list(edge) for edge in actual_edges],
                    details={"n_lines": None, "secondary_edge_selection": False},
                ),
                CheckResult(
                    id=f"mne.{method}.circle_pixel_parity",
                    passed=bool(pixel_comparison["pixel_identical"]),
                    expected="pixel-identical native MNE and PyConnviz circle renders",
                    observed=pixel_comparison,
                    max_abs_error=(
                        None
                        if pixel_comparison["max_abs_error"] is None
                        else float(pixel_comparison["max_abs_error"])
                    ),
                ),
            ]
        )
        evidence[method] = {
            "method": method,
            "object_type": type(connectivity).__name__,
            "edge_count": len(actual_edges),
            "symmetry_basis": prepared.metadata.get("symmetry_basis"),
            "visual_artifacts": visual_artifacts,
            "manifest": _path_provenance(manifest_path),
        }
    return checks, evidence


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file using bounded reads."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_report(
    checks: Sequence[CheckResult],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Build the authoritative machine-readable report."""

    check_rows = [_jsonable(asdict(check)) for check in checks]
    required_rows = [row for row in check_rows if bool(row["required"])]
    diagnostic_rows = [row for row in check_rows if not bool(row["required"])]

    def counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
        passed = sum(bool(row["passed"]) for row in rows)
        return {"passed": passed, "failed": len(rows) - passed, "total": len(rows)}

    required_summary = counts(required_rows)
    return {
        "schema_version": 2,
        "overall_passed": required_summary["failed"] == 0,
        "summary": {
            "required": required_summary,
            "diagnostic": counts(diagnostic_rows),
            "all": counts(check_rows),
        },
        "provenance": _jsonable(provenance),
        "checks": check_rows,
    }


def _markdown_report(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    if not isinstance(summary, Mapping):
        raise TypeError("report summary must be a mapping")
    state = "PASS" if report["overall_passed"] else "FAIL"
    required_summary = summary.get("required")
    diagnostic_summary = summary.get("diagnostic")
    if not isinstance(required_summary, Mapping) or not isinstance(
        diagnostic_summary, Mapping
    ):
        raise TypeError("report summary must contain required and diagnostic mappings")
    lines = [
        "# PyConnviz upstream accuracy validation",
        "",
        f"**Overall:** {state}",
        "",
        (
            f"Required checks: {required_summary['passed']} passed, "
            f"{required_summary['failed']} failed, {required_summary['total']} total."
        ),
        (
            f"Diagnostics: {diagnostic_summary['passed']} matched, "
            f"{diagnostic_summary['failed']} observations differed, "
            f"{diagnostic_summary['total']} total."
        ),
        "",
        "Machine-readable expected/observed values are in `report.json`.",
        "",
        "## Environment",
        "",
    ]
    provenance = report.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise TypeError("report provenance must be a mapping")
    packages = provenance.get("packages", {})
    if isinstance(packages, Mapping) and packages:
        for package, version in sorted(packages.items()):
            lines.append(f"- {package}: `{version}`")
    else:
        lines.append("- Package versions were not supplied.")
    sources = provenance.get("upstream_sources", [])
    if isinstance(sources, list) and sources:
        lines.extend(["", "## Upstream source snapshots", ""])
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            lines.append(
                f"- [{source['filename']}]({source['url']}) — "
                f"SHA-256 `{source['sha256']}`, {source['byte_count']} bytes"
            )
    scope = provenance.get("scope", {})
    if isinstance(scope, Mapping) and scope:
        lines.extend(["", "## Scope", ""])
        lines.append(
            "- Surface placement accuracy claimed: "
            f"`{bool(scope.get('surface_accuracy_claimed', False))}`"
        )
        lines.append(
            "- Surface visualization audited: "
            f"`{bool(scope.get('surface_visualization_audited', False))}`"
        )
        lines.append(
            "- Individual surface-registration truth claimed: "
            f"`{bool(scope.get('surface_registration_truth_claimed', False))}`"
        )
        lines.append(
            f"- Product code modified by this audit: `{bool(scope.get('product_code_modified'))}`"
        )
    checks = report["checks"]
    if not isinstance(checks, list):
        raise TypeError("report checks must be a list")
    failures = [
        check
        for check in checks
        if isinstance(check, Mapping) and check["required"] and not check["passed"]
    ]
    diagnostics = [
        check for check in checks if isinstance(check, Mapping) and not check["required"]
    ]
    lines.extend(["", "## Required failures", ""])
    if not failures:
        lines.append("None.")
    for check in failures:
        lines.extend(
            [
                f"### `{check['id']}`",
                "",
                f"- Expected: `{_compact_markdown_value(check['expected'])}`",
                f"- Observed: `{_compact_markdown_value(check['observed'])}`",
            ]
        )
        details = check.get("details")
        if details:
            lines.append(f"- Details: `{_compact_markdown_value(details)}`")
        lines.append("")
    lines.extend(["", "## Diagnostic observations", ""])
    if not diagnostics:
        lines.append("None.")
    for check in diagnostics:
        state_label = "MATCH" if check["passed"] else "DIFF"
        lines.extend(
            [
                f"### [{state_label}] `{check['id']}`",
                "",
                f"- Expected: `{_compact_markdown_value(check['expected'])}`",
                f"- Observed: `{_compact_markdown_value(check['observed'])}`",
            ]
        )
        details = check.get("details")
        if details:
            lines.append(f"- Details: `{_compact_markdown_value(details)}`")
        lines.append("")
    lines.extend(
        [
            "## All checks",
            "",
            "| Status | Role | Check | Maximum absolute error |",
            "|---|---|---|---:|",
        ]
    )
    for check in checks:
        if not isinstance(check, Mapping):
            raise TypeError("report check entries must be mappings")
        if check["passed"]:
            marker = "PASS"
        elif check["required"]:
            marker = "FAIL"
        else:
            marker = "DIFF"
        role = "required" if check["required"] else "diagnostic"
        error = "—" if check.get("max_abs_error") is None else str(check["max_abs_error"])
        lines.append(f"| {marker} | {role} | `{check['id']}` | {error} |")
    lines.append("")
    return "\n".join(lines)


def _compact_markdown_value(value: object, *, limit: int = 240) -> str:
    if isinstance(value, list) and len(value) > 8:
        return f"list with {len(value)} entries; see report.json"
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    rendered = rendered.replace("`", "'").replace("\n", " ")
    if len(rendered) > limit:
        return rendered[: limit - 1] + "…"
    return rendered


def write_reports(
    report: Mapping[str, object],
    outdir: str | Path,
) -> tuple[Path, Path]:
    """Write JSON authority and matching human-readable Markdown."""

    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    json_path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path
