from __future__ import annotations

import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release.sh"


def _run_script(*args: str, stdin: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        stdin=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def test_release_script_has_valid_shell_and_documented_interface() -> None:
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR

    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = _run_script("--help")
    assert help_result.returncode == 0, help_result.stderr
    for text in ("Usage:", "build", "testpypi", "pypi", "--yes", "PYCONNVIZ_PYTHON"):
        assert text in help_result.stdout


def test_release_script_rejects_invalid_and_unsafe_invocations_before_build() -> None:
    invalid = _run_script("unknown")
    assert invalid.returncode == 2
    assert "Unknown argument" in invalid.stderr

    production = _run_script("pypi", stdin=subprocess.DEVNULL)
    assert production.returncode == 2
    assert "--yes" in production.stderr
    assert "Building" not in production.stdout


def test_release_script_uses_staged_artifacts_and_is_documented() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    upload_lines = [line for line in source.splitlines() if "twine upload" in line]

    assert "python setup.py" not in source
    assert '"${artifacts[@]}"' in source
    assert len(upload_lines) == 2
    assert all("dist/*" not in line for line in upload_lines)

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include scripts/release.sh" in manifest.splitlines()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for command in (
        "./scripts/release.sh build",
        "./scripts/release.sh testpypi",
        "./scripts/release.sh pypi",
    ):
        assert command in readme
    assert "python setup.py" not in readme
