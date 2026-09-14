"""Validate PyConnviz acceptance artifacts without opening a display."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image

REQUIRED_FILES = (
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
PNG_FILES = (
    "surface_paper.png",
    "surface_wholebrain.png",
    "surface_soft_overlay.png",
    "surface_nilearn_native_three_views.png",
    "surface_interactive.png",
    "surface_fsaverage_full.png",
    "circle.png",
)
SVG_FILES = ("surface_paper.svg", "glass.svg")
HTML_FILES = ("surface_interactive.html", "nilearn_connectome.html")
MIN_SMOOTH_SPHERE_TRIANGLES = 1_656


class AcceptanceError(RuntimeError):
    """Raised when one or more generated artifacts violate the contract."""


def _check_png(path: Path, *, minimum: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    if width < minimum[0] or height < minimum[1]:
        raise AcceptanceError(
            f"{path.name} dimensions {(width, height)} are below required {minimum}"
        )
    variance = float(np.var(rgb))
    mean = float(np.mean(rgb))
    if variance < 1.0:
        raise AcceptanceError(f"{path.name} pixel variance is too low: {variance}")
    if mean <= 2.0 or mean >= 253.0:
        raise AcceptanceError(f"{path.name} is nearly all black or white (mean={mean})")
    return {"width": width, "height": height, "variance": variance, "mean": mean}


def _check_svg(path: Path) -> None:
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as error:
        raise AcceptanceError(f"{path.name} is not valid SVG/XML") from error
    if not root.tag.lower().endswith("svg"):
        raise AcceptanceError(f"{path.name} root element is not SVG")


def _check_html(path: Path, *, require_ball_stick: bool = False) -> None:
    content = path.read_text(encoding="utf-8", errors="replace").lower()
    full_document = "<html" in content and "<body" in content
    nilearn_iframe = (
        "<iframe" in content
        and "srcdoc=" in content
        and "&lt;html" in content
        and "&lt;body" in content
    )
    if not full_document and not nilearn_iframe:
        raise AcceptanceError(f"{path.name} has no valid HTML/body structure")
    if "plotly" not in content and "nilearn" not in content and "connectome" not in content:
        raise AcceptanceError(f"{path.name} does not contain a Plotly/Nilearn connectome")
    if require_ball_stick:
        compact = "".join(content.split())
        required = ('"name":"nodes"', '"name":"edges"', '"render_mode":"ball-and-stick"')
        missing = [token for token in required if token not in compact]
        if missing:
            raise AcceptanceError(
                f"{path.name} is missing the true 3D ball-and-stick contract: {missing}"
            )


def _canonical_edges(edges: Any) -> list[tuple[int, int, float]]:
    try:
        return [(int(source), int(target), float(weight)) for source, target, weight in edges]
    except (TypeError, ValueError) as error:
        raise AcceptanceError("edge manifest entries must be [source, target, weight]") from error


def _check_manifest(path: Path) -> int:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AcceptanceError("edge_manifest.json is not valid JSON") from error
    backends = manifest.get("backends", {})
    required_backends = {
        "surface_matplotlib",
        "surface_nilearn",
        "surface_plotly",
        "glass",
        "html",
        "circle",
    }
    if set(backends) != required_backends:
        raise AcceptanceError(
            f"edge manifest backend keys must be {sorted(required_backends)}"
        )
    canonical = {name: _canonical_edges(edges) for name, edges in backends.items()}
    reference = canonical["surface_matplotlib"]
    for name, edges in canonical.items():
        if edges != reference:
            raise AcceptanceError(
                f"Prepared edge tuples differ between surface_matplotlib and {name}"
            )
    static_geometry = manifest.get("static_surface_geometry", {})
    expected_static = {
        "surface_matplotlib": ("matplotlib-poly3d", 3),
        "surface_nilearn": ("nilearn-plot-img-on-surf+matplotlib-poly3d", 6),
    }
    if set(static_geometry) != set(expected_static):
        raise AcceptanceError("static surface geometry records are incomplete")
    for name, (renderer, panels) in expected_static.items():
        record = static_geometry[name]
        if record.get("render_mode") != "ball-and-stick":
            raise AcceptanceError(f"{name} must record ball-and-stick rendering")
        if record.get("renderer") != renderer:
            raise AcceptanceError(f"{name} renderer must be {renderer}")
        if record.get("panels") != panels:
            raise AcceptanceError(f"{name} must record {panels} panels")
        if record.get("node_collections") != panels:
            raise AcceptanceError(f"{name} must contain one node mesh per panel")
        if record.get("edge_collections") != panels:
            raise AcceptanceError(f"{name} must contain one edge mesh per panel")
        for field in ("node_mesh_triangles", "edge_mesh_triangles"):
            if not isinstance(record.get(field), int) or record[field] <= 0:
                raise AcceptanceError(f"{name} must contain positive {field}")
        sphere_triangles = record.get("node_sphere_triangles")
        if (
            not isinstance(sphere_triangles, int)
            or sphere_triangles < MIN_SMOOTH_SPHERE_TRIANGLES
        ):
            raise AcceptanceError(
                f"{name} smooth sphere mesh must contain at least "
                f"{MIN_SMOOTH_SPHERE_TRIANGLES} triangles per node"
            )
        diameters = record.get("node_diameter_range")
        if not isinstance(diameters, list) or len(diameters) != 2 or not np.allclose(
            diameters,
            [6.0, 16.0],
        ):
            raise AcceptanceError(f"{name} must record the 6-16 mm node diameters")
    directed = manifest.get("directed_case", {})
    directed_edges = _canonical_edges(directed.get("edges", []))
    if directed.get("directed") is not True:
        raise AcceptanceError("directed_case must record directed=true")
    directions = {(source, target) for source, target, _ in directed_edges}
    if not any((target, source) in directions for source, target in directions):
        raise AcceptanceError("directed_case must preserve distinct reciprocal directions")
    return len(reference)


def _check_environment(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    required = (
        "Python=",
        "PyConnviz=",
        "NiBabel=",
        "Nilearn=",
        "MNE=",
        "MNE-Connectivity=",
        "Matplotlib=",
        "Plotly=",
    )
    missing = [key for key in required if key not in content]
    if missing:
        raise AcceptanceError(f"environment.txt is missing: {missing}")


def _check_benchmark(path: Path, *, strict: bool) -> dict[str, Any]:
    try:
        benchmark = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AcceptanceError("benchmark.json is not valid JSON") from error
    numeric = (
        "nodes",
        "visible_edges",
        "vertices_per_hemisphere",
        "prepare_seconds",
        "static_render_seconds",
        "peak_extra_bytes",
    )
    if any(not isinstance(benchmark.get(key), (int, float)) for key in numeric):
        raise AcceptanceError(f"benchmark.json must contain numeric fields: {numeric}")
    if strict:
        if benchmark["nodes"] != 200 or benchmark["visible_edges"] != 120:
            raise AcceptanceError("benchmark must use 200 nodes and 120 visible edges")
        if benchmark["vertices_per_hemisphere"] < 9000:
            raise AcceptanceError("benchmark mesh must have about 10k vertices per hemisphere")
        if benchmark["prepare_seconds"] >= 5.0:
            raise AcceptanceError("prepare benchmark exceeded broad 5 second CI threshold")
        if benchmark["static_render_seconds"] >= 60.0:
            raise AcceptanceError("static render exceeded broad 60 second CI threshold")
        if benchmark["peak_extra_bytes"] >= 1_073_741_824:
            raise AcceptanceError("benchmark peak extra memory exceeded 1 GiB")
    return benchmark


def check_artifacts(directory: str | Path, *, strict: bool = True) -> dict[str, Any]:
    """Check required files, structure, backend consistency, and benchmarks."""

    root = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise AcceptanceError(f"Missing required acceptance files: {missing}")
    empty = [name for name in REQUIRED_FILES if (root / name).stat().st_size == 0]
    if empty:
        raise AcceptanceError(f"Empty acceptance files: {empty}")
    png_report = {}
    for name in PNG_FILES:
        minimum = (2200, 1200) if strict and name == "surface_paper.png" else (
            (800, 500) if strict else (20, 20)
        )
        png_report[name] = _check_png(root / name, minimum=minimum)
    for name in SVG_FILES:
        _check_svg(root / name)
    for name in HTML_FILES:
        _check_html(
            root / name,
            require_ball_stick=name == "surface_interactive.html",
        )
    edge_count = _check_manifest(root / "edge_manifest.json")
    _check_environment(root / "environment.txt")
    benchmark = _check_benchmark(root / "benchmark.json", strict=strict)
    return {
        "files_checked": len(REQUIRED_FILES),
        "backend_edge_count": edge_count,
        "png": png_report,
        "benchmark": benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    report = check_artifacts(args.directory, strict=not args.no_strict)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
