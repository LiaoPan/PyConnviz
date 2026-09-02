"""Thin Nilearn glass-brain and HTML wrappers over prepared connectivity."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from ..models import ConnectomeGeometry, CoordinateSpaceError, PlotResult, PreparedConnectome
from ._visual import edge_visuals, node_visuals


def _get_nilearn_plotting():
    return importlib.import_module("nilearn.plotting")


def _require_mni(geometry: ConnectomeGeometry) -> np.ndarray:
    if geometry.mni_coords is None:
        raise CoordinateSpaceError(
            "Nilearn glass/html backends require geometry.mni_coords in millimetres; "
            "surface_coords cannot be substituted"
        )
    return np.asarray(geometry.mni_coords)


def _matching_nodes(prepared: PreparedConnectome, geometry: ConnectomeGeometry) -> None:
    if tuple(prepared.node_names) != tuple(geometry.node_names):
        raise ValueError("geometry node_names must match prepared node_names in order")


def plot_glass_connectome(
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    *,
    style: str = "paper",
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    edge_cmap: str | None = None,
    display_mode: str = "lyrz",
    title: str | None = None,
    colorbar: bool = True,
    output: str | Path | None = None,
    show: bool = False,
) -> PlotResult:
    """Render Nilearn's static MNI glass-brain without secondary filtering."""

    _matching_nodes(prepared, geometry)
    coords = _require_mni(geometry)
    output_path = None if output is None else Path(output)
    if output_path is not None and output_path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("Glass connectome output must use PNG, SVG, or PDF")
    colors, sizes = node_visuals(
        prepared,
        style=style,
        node_values=node_values,
        node_color_values=node_color_values,
        node_size_values=node_size_values,
        size_range=(25.0, 90.0),
    )
    cmap, vmin, vmax, _ = edge_visuals(prepared, edge_cmap=edge_cmap)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    display = _get_nilearn_plotting().plot_connectome(
        np.asarray(prepared.visible_matrix),
        coords,
        node_color=colors,
        node_size=sizes,
        edge_cmap=cmap,
        edge_vmin=vmin,
        edge_vmax=vmax,
        edge_threshold=None,
        output_file=None if output_path is None else output_path,
        display_mode=display_mode,
        title=title,
        black_bg=style == "dark",
        colorbar=colorbar and bool(prepared.edges),
    )
    if output_path is not None and (
        not output_path.is_file() or output_path.stat().st_size == 0
    ):
        raise OSError(f"Nilearn did not create a non-empty glass output: {output_path}")
    if show:
        importlib.import_module("matplotlib.pyplot").show()
    return PlotResult(
        backend="glass",
        engine=None,
        artist=display,
        prepared=prepared,
        output_files=() if output_path is None else (output_path,),
    )


def plot_html_connectome(
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    *,
    style: str = "paper",
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    edge_cmap: str | None = None,
    title: str | None = None,
    colorbar: bool = True,
    output: str | Path | None = None,
) -> PlotResult:
    """Render Nilearn's MNI HTML connectome without secondary filtering."""

    _matching_nodes(prepared, geometry)
    coords = _require_mni(geometry)
    output_path = None if output is None else Path(output)
    if output_path is not None and output_path.suffix.lower() != ".html":
        raise ValueError("Nilearn interactive connectome output must use HTML")
    colors, sizes = node_visuals(
        prepared,
        style=style,
        node_values=node_values,
        node_color_values=node_color_values,
        node_size_values=node_size_values,
        size_range=(3.0, 7.0),
    )
    cmap, _, _, signed = edge_visuals(prepared, edge_cmap=edge_cmap)
    view = _get_nilearn_plotting().view_connectome(
        np.asarray(prepared.visible_matrix),
        coords,
        edge_threshold=None,
        edge_cmap=cmap,
        symmetric_cmap=signed,
        node_color=colors,
        node_size=float(np.mean(sizes)) if len(sizes) else 3.0,
        colorbar=colorbar and bool(prepared.edges),
        title=title,
        node_labels=list(prepared.node_names),
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        view.save_as_html(output_path)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OSError(f"Nilearn did not create a non-empty HTML output: {output_path}")
    return PlotResult(
        backend="html",
        engine=None,
        artist=view,
        prepared=prepared,
        output_files=() if output_path is None else (output_path,),
    )
