"""Render simulated connectivity on a user-provided local FreeSurfer subject."""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

from pyconnviz import geometry_from_mne_labels, plot_connectome


def main(subjects_dir: Path, subject: str, parc: str, output: Path) -> None:
    """Load local labels without downloading data and render simulated edges."""

    labels = mne.read_labels_from_annot(
        subject=subject, parc=parc, subjects_dir=subjects_dir
    )
    labels = [
        label
        for label in labels
        if "unknown" not in label.name.lower()
        and "corpuscallosum" not in label.name.lower()
    ]
    geometry = geometry_from_mne_labels(
        labels,
        subject=subject,
        subjects_dir=subjects_dir,
        surface="inflated",
    )
    rng = np.random.default_rng(20260831)
    matrix = rng.normal(size=(len(labels), len(labels)))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    plot_connectome(
        matrix,
        geometry,
        max_edges=min(120, len(labels) * (len(labels) - 1) // 2),
        title="Simulated connectivity on local FreeSurfer labels",
        output=output,
        show=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects_dir", type=Path)
    parser.add_argument("--subject", default="fsaverage")
    parser.add_argument("--parc", default="aparc")
    parser.add_argument(
        "--output", type=Path, default=Path("simulated_mne_labels.png")
    )
    args = parser.parse_args()
    main(args.subjects_dir, args.subject, args.parc, args.output)

