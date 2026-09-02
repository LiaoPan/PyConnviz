"""Small compatibility helpers isolated from scientific and rendering logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .models import ConnectivityShapeError, GeometryError


def get_dense_connectivity_data(connectivity: Any) -> NDArray[np.generic]:
    """Return dense data using the public MNE-Connectivity API."""

    try:
        data = connectivity.get_data(output="dense")
    except TypeError as error:
        raise ConnectivityShapeError(
            "This connectivity object does not support public get_data(output='dense'). "
            "Convert a bivariate result to a dense ROI-by-ROI matrix before plotting."
        ) from error
    return np.array(data, copy=True)


def get_public_coord(connectivity: Any, names: tuple[str, ...]) -> NDArray[np.generic] | None:
    """Read one coordinate through public attributes or the public coords mapping."""

    coords = getattr(connectivity, "coords", None)
    for name in names:
        if coords is not None and name in coords:
            value = coords[name]
            if hasattr(value, "values"):
                value = value.values
            return np.asarray(value)
        if hasattr(connectivity, name):
            value = getattr(connectivity, name)
            if value is not None:
                return np.asarray(value)
    return None


def get_public_plotly_figure(surface_figure: Any) -> Any:
    """Return Nilearn's documented PlotlySurfaceFigure.figure attribute."""

    if not hasattr(surface_figure, "figure"):
        raise TypeError(
            "Nilearn Plotly surface result must expose the public 'figure' attribute"
        )
    figure = surface_figure.figure
    if figure is None:
        raise TypeError("Nilearn Plotly surface result has no public figure instance")
    return figure


def read_mne_curvature(mne: Any, path: str | Path) -> NDArray[np.float64]:
    """Read FreeSurfer curvature through MNE's public surface module."""

    surface = getattr(mne, "surface", None)
    reader = getattr(surface, "read_curvature", None)
    if not callable(reader):
        raise GeometryError(
            "This MNE version does not expose public mne.surface.read_curvature; "
            "install a supported MNE release (mne>=1.6)."
        )
    return np.array(reader(path, binary=False), dtype=np.float64, copy=True)
