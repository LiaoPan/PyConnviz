"""MNE-Connectivity circle rendering over the exact visible matrix."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..models import ConnectomeGeometry, PlotResult, PreparedConnectome
from ._visual import edge_visuals, node_visuals


def _get_circle_function():
    return importlib.import_module("mne_connectivity.viz").plot_connectivity_circle


def _permutation(order: Sequence[int] | None, node_count: int) -> np.ndarray:
    if order is None:
        return np.arange(node_count, dtype=np.int64)
    array = np.asarray(order)
    if (
        array.shape != (node_count,)
        or not np.issubdtype(array.dtype, np.integer)
        or set(array.tolist()) != set(range(node_count))
    ):
        raise ValueError(f"node_order must be a permutation of 0..{node_count - 1}")
    return array.astype(np.int64, copy=True)


def plot_circle_connectome(
    prepared: PreparedConnectome,
    geometry: ConnectomeGeometry,
    *,
    style: str = "paper",
    node_values: Any = None,
    node_color_values: Any = None,
    node_size_values: Any = None,
    edge_cmap: str | None = None,
    node_order: Sequence[int] | None = None,
    title: str | None = None,
    colorbar: bool = True,
    output: str | Path | None = None,
    show: bool = False,
) -> PlotResult:
    """Render an MNE circle without n_lines or any other secondary edge selection."""

    if tuple(prepared.node_names) != tuple(geometry.node_names):
        raise ValueError("geometry node_names must match prepared node_names in order")
    output_path = None if output is None else Path(output)
    if output_path is not None and output_path.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError("Circle output must use PNG, SVG, or PDF")
    order = _permutation(node_order, len(prepared.node_names))
    display_matrix = np.asarray(prepared.visible_matrix)[np.ix_(order, order)]
    display_names = [prepared.node_names[index] for index in order]
    colors, _ = node_visuals(
        prepared,
        style=style,
        node_values=node_values,
        node_color_values=node_color_values,
        node_size_values=node_size_values,
        size_range=(1.0, 1.0),
    )
    display_colors = [colors[index] for index in order]
    cmap, vmin, vmax, _ = edge_visuals(prepared, edge_cmap=edge_cmap)
    facecolor = "#111318" if style == "dark" else "white"
    textcolor = "white" if style == "dark" else "black"
    figure, axis = _get_circle_function()(
        display_matrix,
        display_names,
        n_lines=None,
        node_colors=display_colors,
        facecolor=facecolor,
        textcolor=textcolor,
        node_edgecolor=textcolor,
        colormap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar and bool(prepared.edges),
        title=title,
        interactive=False,
        show=show,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=facecolor)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise OSError(f"MNE-Connectivity did not create a non-empty circle: {output_path}")
    return PlotResult(
        backend="circle",
        engine=None,
        artist=(figure, axis),
        prepared=prepared,
        output_files=() if output_path is None else (output_path,),
    )
