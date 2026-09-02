"""Select a frequency range from a local MNE-Connectivity result."""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
from mne_connectivity import read_connectivity

from pyconnviz import geometry_from_mne_labels, plot_connectome


def main(
    connectivity_file: Path,
    subjects_dir: Path,
    subject: str,
    parc: str,
    output: Path,
) -> None:
    """Read only local connectivity and FreeSurfer data, then render alpha band."""

    connectivity = read_connectivity(connectivity_file)
    labels = mne.read_labels_from_annot(
        subject=subject, parc=parc, subjects_dir=subjects_dir
    )
    labels = [label for label in labels if "unknown" not in label.name.lower()]
    geometry = geometry_from_mne_labels(
        labels, subject=subject, subjects_dir=subjects_dir, surface="inflated"
    )
    plot_connectome(
        connectivity,
        geometry,
        freq=(8.0, 13.0),
        reduction="mean",
        max_edges=120,
        title="Alpha-band connectivity",
        output=output,
        show=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("connectivity_file", type=Path)
    parser.add_argument("subjects_dir", type=Path)
    parser.add_argument("--subject", default="fsaverage")
    parser.add_argument("--parc", default="aparc")
    parser.add_argument("--output", type=Path, default=Path("alpha_surface.svg"))
    args = parser.parse_args()
    main(
        args.connectivity_file,
        args.subjects_dir,
        args.subject,
        args.parc,
        args.output,
    )

