"""Run the reproducible real-upstream PyConnviz accuracy audit."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

if __package__:
    from . import upstream_validation as audit
else:
    import upstream_validation as audit


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate PyConnviz against official Nilearn and MNE-Connectivity examples."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "upstream_validation",
        help="Nilearn dataset cache directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=project_root / "artifacts" / "upstream_validation",
        help="Directory for source snapshots, figures, and reports.",
    )
    parser.add_argument(
        "--fsaverage-dir",
        type=Path,
        default=project_root / "data" / "fsaverage",
        help="User-provided full-resolution FreeSurfer fsaverage directory.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require existing official-source snapshots and cached Nilearn inputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return status 1 when any required accuracy check fails.",
    )
    return parser


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    """Run all audit cases and return a shell-compatible status."""

    args = _parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    outdir = args.outdir.resolve()
    fsaverage_dir = args.fsaverage_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / "matplotlib_cache"))

    source_records = audit.snapshot_sources(
        outdir,
        skip_download=args.skip_download,
    )
    mne_checks, mne_evidence = audit.run_mne_case()
    mne_visual_checks, mne_visual_evidence = audit.run_mne_visual_cases(outdir)
    nilearn_checks, nilearn_evidence = audit.run_nilearn_case(
        data_dir,
        outdir,
        skip_download=args.skip_download,
    )
    power_checks, power_evidence = audit.run_nilearn_power_case(
        data_dir,
        outdir,
        skip_download=args.skip_download,
    )
    matrix = np.load(outdir / "nilearn_correlation_matrix.npy")
    coords = np.load(outdir / "nilearn_mni_coords.npy")
    labels = json.loads((outdir / "nilearn_labels.json").read_text(encoding="utf-8"))
    surface_checks, surface_evidence = audit.run_surface_case(
        matrix,
        labels,
        coords,
        fsaverage_dir,
        outdir,
    )
    provenance = {
        "generated_at_utc": _timestamp(),
        "packages": audit.environment_versions(),
        "upstream_sources": source_records,
        "cases": {
            "mne_numeric": mne_evidence,
            "mne_visual": mne_visual_evidence,
            "nilearn_msdl": nilearn_evidence,
            "nilearn_power264": power_evidence,
            "surface_msdl_fsaverage": surface_evidence,
        },
        "options": {
            "data_dir": str(data_dir),
            "outdir": str(outdir),
            "fsaverage_dir": str(fsaverage_dir),
            "skip_download": bool(args.skip_download),
            "strict": bool(args.strict),
        },
        "scope": {
            "surface_accuracy_claimed": False,
            "surface_visualization_audited": True,
            "surface_registration_truth_claimed": False,
            "product_code_modified": True,
        },
    }
    report = audit.build_report(
        [
            *mne_checks,
            *mne_visual_checks,
            *nilearn_checks,
            *power_checks,
            *surface_checks,
        ],
        provenance,
    )
    serializable_provenance = report["provenance"]
    (outdir / "provenance.json").write_text(
        json.dumps(
            serializable_provenance,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    audit.write_reports(report, outdir)

    summary = report["summary"]["required"]
    state = "PASS" if report["overall_passed"] else "FAIL"
    print(
        f"PyConnviz upstream validation: {state}; "
        f"{summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['total']} total; report={outdir / 'report.md'}"
    )
    return int(bool(args.strict) and not bool(report["overall_passed"]))


if __name__ == "__main__":
    raise SystemExit(main())
