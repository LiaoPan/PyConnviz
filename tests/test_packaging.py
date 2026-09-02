from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
GALLERY_FILES = (
    "surface-matplotlib.png",
    "surface-nilearn.png",
    "surface-plotly.png",
    "glass-brain.png",
    "circle.png",
)


def _project_config() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_setuptools_build_uses_pyproject_without_setup_py() -> None:
    config = _project_config()
    build_system = config["build-system"]
    setuptools = config["tool"]["setuptools"]

    assert build_system["build-backend"] == "setuptools.build_meta"
    assert any(
        requirement.startswith("setuptools>=")
        for requirement in build_system["requires"]
    )
    assert "wheel" in build_system["requires"]
    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["packages"]["find"]["where"] == ["src"]

    assert not (ROOT / "setup.py").exists()


def test_packaging_metadata_and_manifest_are_release_ready() -> None:
    config = _project_config()
    project = config["project"]
    init_source = (ROOT / "src" / "pyconnviz" / "__init__.py").read_text(
        encoding="utf-8"
    )
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    version_match = re.search(r'^__version__ = "([^"]+)"$', init_source, re.MULTILINE)
    assert version_match is not None
    assert project["version"] == version_match.group(1)
    assert project["readme"] == "README.md"
    assert project["requires-python"] == ">=3.10"
    assert {
        "build>=1.2",
        "setuptools>=77",
        "twine>=5",
        "wheel>=0.45",
    }.issubset(set(project["optional-dependencies"]["dev"]))
    expected_manifest_lines = (
        "include LICENSE",
        "include README.md",
        "include README.zh-CN.md",
        "include pyproject.toml",
        "include scripts/release.sh",
        "recursive-include examples *.py",
        "recursive-include docs/images *.png",
    )
    assert tuple(manifest.splitlines()) == expected_manifest_lines


@pytest.mark.parametrize(
    ("filename", "heading"),
    (
        ("README.md", "### Install from PyPI"),
        ("README.zh-CN.md", "### 从 PyPI 安装"),
    ),
)
def test_readmes_document_installation_from_pypi(
    filename: str,
    heading: str,
) -> None:
    readme = (ROOT / filename).read_text(encoding="utf-8")

    assert heading in readme
    assert "https://pypi.org/project/pyconnviz/" in readme
    assert "python -m pip install --upgrade pyconnviz" in readme
    assert 'python -m pip install --upgrade "pyconnviz[interactive]"' in readme
    assert (
        'python -m pip install --upgrade "pyconnviz[interactive,export]"'
        in readme
    )


def test_readme_has_quickstart_gallery_all_backends_and_release_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for heading in (
        "## Gallery",
        "## Installation",
        "## Recommended views",
        "## Quickstart",
        "## All rendering modes",
        "## Build and publish",
    ):
        assert heading in readme
    for signature in (
        'backend="surface"',
        'engine="matplotlib"',
        'engine="nilearn"',
        'engine="plotly"',
        'backend="glass"',
        'backend="html"',
        'backend="circle"',
        "geometry_from_arrays",
        "geometry_from_mne_labels",
        "prepare_connectome",
        "edge_threshold",
        "edge_mask",
        "max_edges",
        "node_overlay",
        "directed=True",
        "python -m build",
        "twine check dist/*",
    ):
        assert signature in readme
    assert "python setup.py" not in readme

    for filename in GALLERY_FILES:
        image = ROOT / "docs" / "images" / filename
        payload = image.read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 1024
        assert f"docs/images/{filename}" in readme
    assert readme.count('width="100%"') == 4
    assert 'width="60%"' in readme
    supplementary_start = readme.index(
        "<th>Supplementary: Matplotlib three-view surface</th>"
    )
    supplementary_end = readme.index("</table>", supplementary_start)
    assert 'width="50%"' not in readme[supplementary_start:supplementary_end]


def test_readme_documents_static_visibility_defaults_and_panel_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "0.08 / 0.08 / 0.18" in readme
    assert "single-hemisphere lateral panels" in readme
    assert "cross-hemisphere edges" in readme


def test_readme_recommends_plotly_then_glass_before_fixed_surface_views() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.index("docs/images/surface-plotly.png") < readme.index(
        "docs/images/glass-brain.png"
    )
    assert readme.index("docs/images/glass-brain.png") < readme.index(
        "docs/images/surface-matplotlib.png"
    )

    start = readme.index("## Recommended views")
    end = readme.index("## Quickstart", start)
    recommended = readme[start:end]
    assert recommended.index('engine="plotly"') < recommended.index(
        'backend="glass"'
    )
    assert recommended.index('backend="glass"') < recommended.index(
        'engine="matplotlib"'
    )
    assert "primary interactive anatomical view" in recommended
    assert "primary static connectivity overview" in recommended
    assert re.search(
        r"supplementary fixed-view\s+anatomical context",
        recommended,
    )


def _fenced_blocks(markdown: str) -> list[str]:
    return re.findall(r"```[^\n]*\n.*?```", markdown, flags=re.DOTALL)


def test_readmes_offer_language_switch_and_chinese_is_complete() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")

    english_header = "\n".join(english.splitlines()[:8])
    chinese_header = "\n".join(chinese.splitlines()[:8])
    assert "[简体中文](README.zh-CN.md)" in english_header
    assert "[English](README.md)" in chinese_header
    assert _fenced_blocks(chinese) == _fenced_blocks(english)
    assert chinese.count("\n## ") == english.count("\n## ")

    for heading in (
        "## 效果展示",
        "## 安装",
        "## 推荐视图",
        "## 快速开始",
        "## 所有绘图模式",
        "## 构建与发布",
    ):
        assert heading in chinese

    for signature in (
        'backend="surface"',
        'engine="matplotlib"',
        'engine="nilearn"',
        'engine="plotly"',
        'backend="glass"',
        'backend="html"',
        'backend="circle"',
        "0.08 / 0.08 / 0.18",
        "python -m build",
        "twine check dist/*",
    ):
        assert signature in chinese

    for filename in GALLERY_FILES:
        assert f"docs/images/{filename}" in chinese
    assert chinese.index("docs/images/surface-plotly.png") < chinese.index(
        "docs/images/glass-brain.png"
    )
    assert chinese.index("docs/images/glass-brain.png") < chinese.index(
        "docs/images/surface-matplotlib.png"
    )
