"""Construction and manipulation of surface and MNI connectome geometry."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._compat import read_mne_curvature
from .models import ConnectomeGeometry, GeometryError, HemisphereMesh


def geometry_from_arrays(
    node_names: Sequence[str],
    surface_coords: Any,
    hemispheres: Sequence[str],
    meshes: Mapping[str, HemisphereMesh],
    *,
    mni_coords: Any = None,
    groups: Sequence[str] | None = None,
    node_vertices: Any = None,
    roi_vertices: Sequence[Any] | None = None,
    subject: str | None = None,
    surface_name: str | None = None,
) -> ConnectomeGeometry:
    """Build validated geometry without inferring or converting coordinates."""

    return ConnectomeGeometry(
        node_names=tuple(node_names),
        surface_coords=surface_coords,
        hemispheres=tuple(hemispheres),
        meshes=dict(meshes),
        mni_coords=mni_coords,
        groups=None if groups is None else tuple(groups),
        node_vertices=node_vertices,
        roi_vertices=None if roi_vertices is None else tuple(roi_vertices),
        subject=subject,
        surface_name=surface_name,
        coordinate_units="mm",
    )


def compute_vertex_normals(mesh: HemisphereMesh) -> NDArray[np.float64]:
    """Return finite unit area-weighted normals for every mesh vertex."""

    coordinates = mesh.coordinates
    normals = np.zeros_like(coordinates, dtype=np.float64)
    if len(mesh.faces):
        triangles = coordinates[mesh.faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        for corner in range(3):
            np.add.at(normals, mesh.faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > np.finfo(float).eps
    normals[valid] /= lengths[valid, np.newaxis]

    if np.any(~valid):
        center = np.mean(coordinates, axis=0) if len(coordinates) else np.zeros(3)
        fallback = coordinates[~valid] - center
        fallback_lengths = np.linalg.norm(fallback, axis=1)
        usable = fallback_lengths > np.finfo(float).eps
        fallback[usable] /= fallback_lengths[usable, np.newaxis]
        fallback[~usable] = np.array([0.0, 0.0, 1.0])
        normals[~valid] = fallback
    return normals


def node_surface_normals(geometry: ConnectomeGeometry) -> NDArray[np.float64]:
    """Look up the display-surface normal associated with every node."""

    if geometry.node_vertices is None:
        raise GeometryError(
            "node_vertices are required to compute display normals; provide representative "
            "surface vertex indices when constructing geometry"
        )
    per_hemi = {hemi: compute_vertex_normals(mesh) for hemi, mesh in geometry.meshes.items()}
    return np.asarray(
        [
            per_hemi[hemi][int(vertex)]
            for hemi, vertex in zip(
                geometry.hemispheres, geometry.node_vertices, strict=True
            )
        ],
        dtype=np.float64,
    )


def offset_node_coordinates(
    geometry: ConnectomeGeometry, *, offset_mm: float = 1.5
) -> NDArray[np.float64]:
    """Move display coordinates along unit surface normals without mutating geometry."""

    if isinstance(offset_mm, bool) or not isinstance(offset_mm, (int, float, np.number)):
        raise GeometryError("offset_mm must be a finite non-negative number")
    offset = float(offset_mm)
    if not np.isfinite(offset) or offset < 0:
        raise GeometryError("offset_mm must be a finite non-negative number")
    return np.asarray(geometry.surface_coords + node_surface_normals(geometry) * offset)


def to_nilearn_polymesh(geometry: ConnectomeGeometry):
    """Convert internal meshes to Nilearn public in-memory surface objects."""

    try:
        surface = importlib.import_module("nilearn.surface")
    except ImportError as error:  # required dependency, but keep error actionable
        raise GeometryError("Nilearn is required to convert surface meshes") from error
    parts = {
        hemi: surface.InMemoryMesh(
            np.array(mesh.coordinates, copy=True), np.array(mesh.faces, copy=True)
        )
        for hemi, mesh in geometry.meshes.items()
    }
    return surface.PolyMesh(left=parts.get("left"), right=parts.get("right"))


def _validate_label_hemispheres(labels: Sequence[Any]) -> tuple[str, ...]:
    hemispheres = []
    for index, label in enumerate(labels):
        hemi = getattr(label, "hemi", None)
        if hemi not in {"lh", "rh"}:
            raise GeometryError(
                f"labels[{index}] must belong to exactly one hemisphere "
                f"('lh' or 'rh'); got {hemi!r}"
            )
        hemispheres.append("left" if hemi == "lh" else "right")
    return tuple(hemispheres)


def _load_subject_meshes(
    mne: Any,
    *,
    subject: str,
    subjects_dir: Path,
    surface_name: str,
    needed_hemispheres: set[str],
) -> dict[str, HemisphereMesh]:
    surf_dir = subjects_dir / subject / "surf"
    meshes: dict[str, HemisphereMesh] = {}
    for hemisphere in sorted(needed_hemispheres):
        prefix = "lh" if hemisphere == "left" else "rh"
        surface_path = surf_dir / f"{prefix}.{surface_name}"
        if not surface_path.is_file():
            raise GeometryError(
                f"Required FreeSurfer display surface does not exist: {surface_path}. "
                "Prepare the subject explicitly; PyConnviz does not download templates."
            )
        coordinates, faces = mne.read_surface(surface_path)
        sulc_path = surf_dir / f"{prefix}.sulc"
        sulc = None
        if sulc_path.is_file():
            sulc = read_mne_curvature(mne, sulc_path)
        else:
            warnings.warn(
                f"FreeSurfer sulc file is missing: {sulc_path}; cortical shading is unavailable",
                RuntimeWarning,
                stacklevel=3,
            )
        meshes[hemisphere] = HemisphereMesh(
            coordinates=coordinates,
            faces=faces,
            sulc=sulc,
            name=f"{subject}:{prefix}.{surface_name}",
        )
    return meshes


def geometry_from_mne_labels(
    labels: Sequence[Any],
    *,
    subject: str,
    subjects_dir: str | Path,
    src: Any = None,
    surface: str = "inflated",
    sphere_surface: str = "sphere",
) -> ConnectomeGeometry:
    """Build display-surface and MNI geometry from ordered MNE labels.

    The input order is the connectivity node order. No labels are filtered or
    sorted internally, and no subject data are downloaded.
    """

    ordered_labels = tuple(labels)
    if not ordered_labels:
        raise GeometryError("labels must contain at least one unilateral MNE Label")
    hemispheres = _validate_label_hemispheres(ordered_labels)
    subjects_path = Path(subjects_dir)
    try:
        mne = importlib.import_module("mne")
    except ImportError as error:
        raise GeometryError("MNE-Python is required for geometry_from_mne_labels") from error
    meshes = _load_subject_meshes(
        mne,
        subject=subject,
        subjects_dir=subjects_path,
        surface_name=surface,
        needed_hemispheres=set(hemispheres),
    )

    node_vertices: list[int] = []
    node_names: list[str] = []
    roi_vertices: list[NDArray[np.int64]] = []
    for index, label in enumerate(ordered_labels):
        if not hasattr(label, "center_of_mass"):
            raise GeometryError(f"labels[{index}] does not expose Label.center_of_mass")
        vertex = label.center_of_mass(
            subject=subject,
            restrict_vertices=False if src is None else src,
            subjects_dir=subjects_path,
            surf=sphere_surface,
        )
        node_vertices.append(int(vertex))
        node_names.append(str(getattr(label, "name", f"node-{index}")))
        vertices = getattr(label, "vertices", None)
        if vertices is None:
            raise GeometryError(f"labels[{index}] does not expose vertices")
        roi_vertices.append(np.asarray(vertices, dtype=np.int64))

    surface_coords = np.asarray(
        [
            meshes[hemi].coordinates[vertex]
            for hemi, vertex in zip(hemispheres, node_vertices, strict=True)
        ],
        dtype=np.float64,
    )
    mne_hemispheres = [0 if hemi == "left" else 1 for hemi in hemispheres]
    mni_coords = np.asarray(
        mne.vertex_to_mni(node_vertices, mne_hemispheres, subject, subjects_path),
        dtype=np.float64,
    )
    if mni_coords.ndim == 1:
        mni_coords = mni_coords[np.newaxis, :]
    return geometry_from_arrays(
        node_names,
        surface_coords,
        hemispheres,
        meshes,
        mni_coords=mni_coords,
        node_vertices=np.asarray(node_vertices, dtype=np.int64),
        roi_vertices=roi_vertices,
        subject=subject,
        surface_name=surface,
    )
