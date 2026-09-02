"""PyConnviz: Nilearn-first connectivity visualization."""

from .api import plot_connectome
from .connectivity import prepare_connectome
from .geometry import geometry_from_arrays, geometry_from_mne_labels
from .models import (
    ConnectivityShapeError,
    ConnectomeGeometry,
    CoordinateSpaceError,
    Edge,
    GeometryError,
    HemisphereMesh,
    OptionalDependencyError,
    PlotResult,
    PreparedConnectome,
    PyConnvizError,
    ViewSpec,
)
from .styles import get_style

__version__ = "0.1.0"

__all__ = [
    "ConnectivityShapeError",
    "ConnectomeGeometry",
    "CoordinateSpaceError",
    "Edge",
    "GeometryError",
    "HemisphereMesh",
    "OptionalDependencyError",
    "PlotResult",
    "PreparedConnectome",
    "PyConnvizError",
    "ViewSpec",
    "__version__",
    "geometry_from_arrays",
    "geometry_from_mne_labels",
    "get_style",
    "plot_connectome",
    "prepare_connectome",
]
