"""Batched Matplotlib artists for physical ball-and-stick geometry."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib.colors import LightSource, to_rgba_array
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import ArrayLike, NDArray

from ._plotly_primitives import (
    TriangleMesh,
    cone_mesh,
    merge_meshes,
    sphere_mesh,
    tube_mesh,
)

_SPHERE_LATITUDE_STEPS = 32
_SPHERE_LONGITUDE_STEPS = 48
_TUBE_SIDES = 8
_CONE_SIDES = 8
_LIGHT_SOURCE = LightSource(azdeg=315, altdeg=45)


class BallStickCollection(Poly3DCollection):
    """One auditable, shaded collection of disconnected physical meshes."""

    diameters: NDArray[np.float64]
    mesh_vertex_count: int
    mesh_triangle_count: int
    depth_cue: bool
    source_colors: NDArray[np.float64]
    part_centers: NDArray[np.float64]

    def __init__(
        self,
        parts: Sequence[TriangleMesh],
        colors: Sequence[ArrayLike],
        diameters: ArrayLike,
        *,
        gid: str,
        alpha: float,
        depth_cue: bool,
        zorder: float,
    ) -> None:
        resolved_colors = to_rgba_array(colors)
        resolved_diameters = np.asarray(diameters, dtype=np.float64)
        if len(parts) != len(resolved_colors) or len(parts) != len(resolved_diameters):
            raise ValueError("parts, colors, and diameters must have the same length")
        merged = merge_meshes(parts)
        facecolors = np.concatenate(
            [
                np.repeat(color[None, :], len(part.faces), axis=0)
                for part, color in zip(parts, resolved_colors, strict=True)
            ],
            axis=0,
        )
        super().__init__(
            merged.vertices[merged.faces],
            facecolors=facecolors,
            alpha=float(alpha),
            shade=bool(depth_cue),
            lightsource=_LIGHT_SOURCE,
            zsort="average",
        )
        self.set_edgecolor("none")
        self.set_gid(gid)
        self.set_zorder(zorder)
        self.diameters = np.array(resolved_diameters, copy=True)
        self.mesh_vertex_count = len(merged.vertices)
        self.mesh_triangle_count = len(merged.faces)
        self.depth_cue = bool(depth_cue)
        self.source_colors = np.array(resolved_colors, copy=True)
        self.part_centers = np.asarray(
            [np.mean(part.vertices, axis=0) for part in parts],
            dtype=np.float64,
        )


def sphere_collection(
    centers: ArrayLike,
    diameters: ArrayLike,
    colors: Sequence[ArrayLike],
    *,
    depth_cue: bool,
) -> BallStickCollection | None:
    """Return one shaded collection containing all requested node spheres."""

    resolved_centers = np.asarray(centers, dtype=np.float64)
    resolved_diameters = np.asarray(diameters, dtype=np.float64)
    parts = [
        sphere_mesh(
            center,
            diameter / 2.0,
            latitude_steps=_SPHERE_LATITUDE_STEPS,
            longitude_steps=_SPHERE_LONGITUDE_STEPS,
        )
        for center, diameter in zip(
            resolved_centers,
            resolved_diameters,
            strict=True,
        )
    ]
    if not parts:
        return None
    return BallStickCollection(
        parts,
        colors,
        resolved_diameters,
        gid="pyconnviz-nodes",
        alpha=1.0,
        depth_cue=depth_cue,
        zorder=12,
    )


def tube_collection(
    curves: Sequence[ArrayLike],
    diameters: ArrayLike,
    colors: Sequence[ArrayLike],
    *,
    alpha: float,
    depth_cue: bool,
) -> BallStickCollection | None:
    """Return one shaded collection containing all requested edge tubes."""

    resolved_diameters = np.asarray(diameters, dtype=np.float64)
    parts = [
        tube_mesh(curve, diameter / 2.0, sides=_TUBE_SIDES)
        for curve, diameter in zip(curves, resolved_diameters, strict=True)
    ]
    if not parts:
        return None
    return BallStickCollection(
        parts,
        colors,
        resolved_diameters,
        gid="pyconnviz-edges",
        alpha=alpha,
        depth_cue=depth_cue,
        zorder=10,
    )


def cone_collection(
    base_centers: ArrayLike,
    tips: ArrayLike,
    diameters: ArrayLike,
    colors: Sequence[ArrayLike],
    *,
    alpha: float,
    depth_cue: bool,
) -> BallStickCollection | None:
    """Return one shaded collection containing all requested direction cones."""

    resolved_bases = np.asarray(base_centers, dtype=np.float64)
    resolved_tips = np.asarray(tips, dtype=np.float64)
    resolved_diameters = np.asarray(diameters, dtype=np.float64)
    parts = [
        cone_mesh(base, tip, diameter / 2.0, sides=_CONE_SIDES)
        for base, tip, diameter in zip(
            resolved_bases,
            resolved_tips,
            resolved_diameters,
            strict=True,
        )
    ]
    if not parts:
        return None
    return BallStickCollection(
        parts,
        colors,
        resolved_diameters,
        gid="pyconnviz-directions",
        alpha=alpha,
        depth_cue=depth_cue,
        zorder=11,
    )
