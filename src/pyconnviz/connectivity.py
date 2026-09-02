"""Connectivity adaptation and the single authoritative edge-selection pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ._compat import get_dense_connectivity_data, get_public_coord
from .models import ConnectivityShapeError, ConnectomeGeometry, Edge, PreparedConnectome

Reduction = Literal["mean", "median"]
ComplexMode = Literal["raise", "magnitude", "real", "imag"]
SymmetrizeMode = Literal["auto", "none", "mean", "maxabs", "lower", "upper"]
KeepSign = Literal["both", "positive", "negative"]
StrengthMode = Literal["total", "in", "out"]

_DIM_ALIASES = {
    "freq": ("freq", "freqs", "frequency", "frequencies"),
    "time": ("time", "times"),
    "epoch": ("epoch", "epochs"),
    "component": ("component", "components", "pattern", "patterns"),
}

# MNE-Connectivity stores all-to-all bivariate estimates in one dense triangle.
# Only scalar metrics whose public definition is invariant to exchanging the two
# nodes are safe to mirror automatically. Directed, signed-antisymmetric, complex,
# and multivariate methods deliberately stay out of this set.
_MNE_SYMMETRIC_SCALAR_METHODS = frozenset(
    {
        "coh",
        "plv",
        "ciplv",
        "ppc",
        "pli",
        "pli2_unbiased",
        "wpli",
        "wpli2_debiased",
    }
)
_MISSING = object()


@dataclass(frozen=True)
class _AdaptedMatrix:
    values: NDArray[np.generic]
    node_names: tuple[str, ...]
    metadata: dict[str, Any]
    declared_symmetric: bool


def _normalize_axis(axis: int, ndim: int, *, name: str) -> int:
    if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)):
        raise ConnectivityShapeError(f"{name} entries must be integer axes")
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ConnectivityShapeError(f"{name} axis {axis} is out of range for {ndim} dimensions")
    return normalized


def _validate_reduction(reduction: str) -> Reduction:
    if reduction not in {"mean", "median"}:
        raise ValueError("reduction must be 'mean' or 'median'")
    return reduction  # type: ignore[return-value]


def _reduce(array: NDArray[np.generic], axes: tuple[int, ...], reduction: Reduction):
    reducer = np.mean if reduction == "mean" else np.median
    return reducer(array, axis=axes)


def _adapt_numpy(
    connectivity: Any,
    *,
    node_axes: tuple[int, int] | None,
    reduce_axes: Sequence[int] | None,
    reduction: Reduction,
) -> _AdaptedMatrix:
    values = np.array(connectivity, copy=True)
    if values.ndim < 2:
        raise ConnectivityShapeError(
            f"connectivity must have at least two dimensions; got shape {values.shape}"
        )
    if values.ndim == 2:
        if node_axes is not None or reduce_axes is not None:
            raise ConnectivityShapeError(
                "node_axes and reduce_axes are only valid for high-dimensional NumPy input"
            )
        matrix = values
    else:
        if node_axes is None:
            raise ConnectivityShapeError(
                "High-dimensional NumPy input requires explicit node_axes=(input_axis, output_axis)"
            )
        if len(node_axes) != 2:
            raise ConnectivityShapeError("node_axes must contain exactly two axes")
        normalized_nodes = tuple(
            _normalize_axis(axis, values.ndim, name="node_axes") for axis in node_axes
        )
        if normalized_nodes[0] == normalized_nodes[1]:
            raise ConnectivityShapeError("node_axes must identify two distinct axes")
        if reduce_axes is None:
            raise ConnectivityShapeError(
                "High-dimensional NumPy input requires reduce_axes covering all non-node axes"
            )
        normalized_reduce = tuple(
            _normalize_axis(axis, values.ndim, name="reduce_axes") for axis in reduce_axes
        )
        expected_reduce = set(range(values.ndim)) - set(normalized_nodes)
        if set(normalized_reduce) != expected_reduce or len(set(normalized_reduce)) != len(
            normalized_reduce
        ):
            raise ConnectivityShapeError(
                "reduce_axes must contain all non-node axes exactly once"
            )
        moved = np.moveaxis(values, normalized_nodes, (0, 1))
        matrix = _reduce(moved, tuple(range(2, moved.ndim)), reduction)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ConnectivityShapeError(
            f"connectivity must reduce to a square matrix; got shape {matrix.shape}"
        )
    names = tuple(str(index) for index in range(matrix.shape[0]))
    return _AdaptedMatrix(
        np.asarray(matrix), names, {"source_type": "numpy", "reduction": reduction}, False
    )


def _semantic_dim(name: str) -> str | None:
    normalized = name.lower().replace("_", "").replace("-", "")
    for semantic, aliases in _DIM_ALIASES.items():
        if any(normalized == alias.replace("_", "") for alias in aliases):
            return semantic
    return None


def _dense_extra_dims(connectivity: Any, extra_count: int) -> list[str]:
    raw_dims = [str(value) for value in getattr(connectivity, "dims", ())]
    semantic_dims = [semantic for value in raw_dims if (semantic := _semantic_dim(value))]
    if len(semantic_dims) == extra_count:
        return semantic_dims
    discovered: list[str] = []
    for semantic, aliases in _DIM_ALIASES.items():
        if get_public_coord(connectivity, aliases) is not None:
            discovered.append(semantic)
    if len(discovered) == extra_count:
        return discovered
    if extra_count == 0:
        return []
    raise ConnectivityShapeError(
        "Could not map dense MNE-Connectivity dimensions to frequency, time, or epoch. "
        f"Public dims={tuple(raw_dims)!r}, dense extra dimensions={extra_count}."
    )


def _selector_indices(
    coord: NDArray[np.generic], selector: Any, *, semantic: str
) -> tuple[NDArray[np.int64], bool]:
    values = np.asarray(coord).reshape(-1)
    if isinstance(selector, tuple):
        if len(selector) != 2:
            raise ValueError(f"{semantic} range selector must be a two-value tuple")
        low, high = selector
        if low > high:
            raise ValueError(f"{semantic} selector lower bound must not exceed upper bound")
        indices = np.flatnonzero((values >= low) & (values <= high)).astype(np.int64)
        if not len(indices):
            raise ValueError(
                f"{semantic} selector {selector!r} does not select any available coordinate"
            )
        return indices, True
    if semantic == "epoch" and isinstance(selector, (int, np.integer)):
        index = int(selector)
        if index < 0:
            index += len(values)
        if index < 0 or index >= len(values):
            raise ValueError(f"epoch index {selector} is out of range for {len(values)} epochs")
        return np.array([index], dtype=np.int64), False
    if not np.issubdtype(values.dtype, np.number):
        matches = np.flatnonzero(values == selector)
        if not len(matches):
            raise ValueError(f"{semantic} selector {selector!r} is not available")
        return matches[:1].astype(np.int64), False
    index = int(np.argmin(np.abs(values.astype(float) - float(selector))))
    return np.array([index], dtype=np.int64), False


def _reduce_mne_dimensions(
    values: NDArray[np.generic],
    connectivity: Any,
    extra_dims: list[str],
    *,
    freq: Any,
    time: Any,
    epoch: Any,
    reduction: Reduction,
    metadata: dict[str, Any],
) -> NDArray[np.generic]:
    selectors = {"freq": freq, "time": time, "epoch": epoch}
    current_dims = list(extra_dims)
    result = values
    for semantic in list(extra_dims):
        axis = 2 + current_dims.index(semantic)
        if semantic == "component":
            if result.shape[axis] != 1:
                raise ConnectivityShapeError(
                    "A multivariate connectivity component dimension cannot be silently reduced. "
                    "Define a scientifically justified component summary before calling PyConnviz."
                )
            result = np.take(result, 0, axis=axis)
            current_dims.remove(semantic)
            continue
        selector = selectors[semantic]
        coord = get_public_coord(connectivity, _DIM_ALIASES[semantic])
        if coord is None:
            coord = np.arange(result.shape[axis])
        if len(np.asarray(coord).reshape(-1)) != result.shape[axis]:
            raise ConnectivityShapeError(
                f"Public {semantic} coordinate length does not match dense data axis {axis}"
            )
        if selector is None:
            if result.shape[axis] != 1:
                raise ConnectivityShapeError(
                    f"Dense connectivity has {result.shape[axis]} {semantic} values; "
                    f"provide a {semantic} selector"
                )
            result = np.take(result, 0, axis=axis)
        else:
            indices, aggregate = _selector_indices(coord, selector, semantic=semantic)
            selected = np.take(result, indices, axis=axis)
            result = _reduce(selected, (axis,), reduction) if aggregate else np.take(
                selected, 0, axis=axis
            )
            metadata[semantic] = selector
        current_dims.remove(semantic)
    return result


def _declared_symmetry(connectivity: Any) -> tuple[bool, str | None]:
    """Return a conservative public-contract symmetry declaration and its basis."""

    direct = getattr(connectivity, "symmetric", _MISSING)
    if direct is not _MISSING and direct is not None:
        return bool(direct), "public:symmetric" if bool(direct) else None
    attrs = getattr(connectivity, "attrs", None)
    if attrs is not None and "symmetric" in attrs and attrs["symmetric"] is not None:
        symmetric = bool(attrs["symmetric"])
        return symmetric, "public:attrs.symmetric" if symmetric else None

    method = getattr(connectivity, "method", _MISSING)
    indices = getattr(connectivity, "indices", _MISSING)
    if (
        isinstance(method, str)
        and method.lower() in _MNE_SYMMETRIC_SCALAR_METHODS
        and indices is None
    ):
        normalized_method = method.lower()
        return True, f"mne:{normalized_method}:all-to-all"
    return False, None


def _adapt_mne_connectivity(
    connectivity: Any,
    *,
    freq: Any,
    time: Any,
    epoch: Any,
    reduction: Reduction,
) -> _AdaptedMatrix:
    values = get_dense_connectivity_data(connectivity)
    n_nodes = getattr(connectivity, "n_nodes", None)
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes <= 0:
        raise ConnectivityShapeError("MNE-Connectivity input must expose a positive public n_nodes")
    if values.ndim < 2 or values.shape[:2] != (int(n_nodes), int(n_nodes)):
        raise ConnectivityShapeError(
            "MNE-Connectivity dense data must start with node-by-node axes; "
            f"expected ({n_nodes}, {n_nodes}, ...), got {values.shape}. A multivariate result "
            "must be summarized explicitly before plotting."
        )
    extra_dims = _dense_extra_dims(connectivity, values.ndim - 2)
    declared_symmetric, symmetry_basis = _declared_symmetry(connectivity)
    metadata: dict[str, Any] = {
        "source_type": type(connectivity).__name__,
        "reduction": reduction,
        "symmetry_basis": symmetry_basis,
    }
    matrix = _reduce_mne_dimensions(
        values,
        connectivity,
        extra_dims,
        freq=freq,
        time=time,
        epoch=epoch,
        reduction=reduction,
        metadata=metadata,
    )
    if matrix.shape != (n_nodes, n_nodes):
        raise ConnectivityShapeError(
            f"MNE-Connectivity selection did not produce a square matrix; got {matrix.shape}"
        )
    raw_names = getattr(connectivity, "names", None)
    names = (
        tuple(str(value) for value in raw_names)
        if raw_names is not None
        else tuple(str(index) for index in range(int(n_nodes)))
    )
    if len(names) != n_nodes:
        raise ConnectivityShapeError(
            f"MNE-Connectivity names has length {len(names)} but n_nodes is {n_nodes}"
        )
    return _AdaptedMatrix(matrix, names, metadata, declared_symmetric)


def _apply_complex_mode(values: NDArray[np.generic], mode: ComplexMode):
    if mode not in {"raise", "magnitude", "real", "imag"}:
        raise ValueError("complex_mode must be 'raise', 'magnitude', 'real', or 'imag'")
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=np.float64)
    if mode == "raise":
        raise ValueError(
            "Complex connectivity requires explicit complex_mode='magnitude', 'real', or 'imag'"
        )
    if mode == "magnitude":
        return np.asarray(np.abs(values), dtype=np.float64)
    if mode == "real":
        return np.asarray(np.real(values), dtype=np.float64)
    return np.asarray(np.imag(values), dtype=np.float64)


def _triangle_mirror(matrix: NDArray[np.float64], triangle: str) -> NDArray[np.float64]:
    diagonal = np.diag(np.diag(matrix))
    values = np.tril(matrix, -1) if triangle == "lower" else np.triu(matrix, 1)
    return values + values.T + diagonal


def _maxabs_symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(matrix, copy=True)
    for source in range(len(result)):
        for target in range(source + 1, len(result)):
            forward = matrix[source, target]
            reverse = matrix[target, source]
            if np.isfinite(forward) and (
                not np.isfinite(reverse) or abs(forward) >= abs(reverse)
            ):
                selected = forward
            else:
                selected = reverse
            result[source, target] = result[target, source] = selected
    return result


def _recover_declared_triangle(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    if np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-9, equal_nan=True):
        return matrix
    upper_count = int(np.count_nonzero(np.nan_to_num(np.triu(matrix, 1))))
    lower_count = int(np.count_nonzero(np.nan_to_num(np.tril(matrix, -1))))
    if upper_count and not lower_count:
        return _triangle_mirror(matrix, "upper")
    if lower_count and not upper_count:
        return _triangle_mirror(matrix, "lower")
    return matrix


def _symmetrize(
    matrix: NDArray[np.float64], mode: SymmetrizeMode, *, declared_symmetric: bool
) -> NDArray[np.float64]:
    if mode not in {"auto", "none", "mean", "maxabs", "lower", "upper"}:
        raise ValueError("symmetrize must be auto, none, mean, maxabs, lower, or upper")
    if mode == "auto":
        return _recover_declared_triangle(matrix) if declared_symmetric else matrix
    if mode == "none":
        return matrix
    if mode == "mean":
        return (matrix + matrix.T) / 2.0
    if mode == "maxabs":
        return _maxabs_symmetrize(matrix)
    return _triangle_mirror(matrix, mode)


def _parse_threshold(value: float | str | None) -> tuple[str, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)%", value.strip())
        if match is None:
            raise ValueError("edge_threshold string must be a percentile such as '95%'")
        percentile = float(match.group(1))
        if not 0 <= percentile <= 100:
            raise ValueError("edge_threshold percentile must be between 0% and 100%")
        return "percentile", percentile
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("edge_threshold must be a non-negative number, percentile string, or None")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError("edge_threshold numeric value must be finite and non-negative")
    return "absolute", numeric


def _candidate_edges(
    matrix: NDArray[np.float64], *, directed: bool
) -> list[tuple[int, int, float]]:
    node_count = len(matrix)
    if directed:
        indices = (
            (source, target)
            for source in range(node_count)
            for target in range(node_count)
            if source != target
        )
    else:
        indices = (
            (source, target)
            for source in range(node_count)
            for target in range(source + 1, node_count)
        )
    return [
        (source, target, float(matrix[source, target]))
        for source, target in indices
        if np.isfinite(matrix[source, target])
    ]


def _node_strength(visible: NDArray[np.float64], *, directed: bool, mode: StrengthMode):
    if mode not in {"total", "in", "out"}:
        raise ValueError("strength_mode must be 'total', 'in', or 'out'")
    absolute = np.abs(visible)
    if not directed:
        return np.sum(absolute, axis=1)
    incoming = np.sum(absolute, axis=0)
    outgoing = np.sum(absolute, axis=1)
    if mode == "in":
        return incoming
    if mode == "out":
        return outgoing
    return incoming + outgoing


def prepare_connectome(
    connectivity: Any,
    *,
    geometry: ConnectomeGeometry | None = None,
    freq: Any = None,
    time: Any = None,
    epoch: Any = None,
    reduction: Reduction = "mean",
    node_axes: tuple[int, int] | None = None,
    reduce_axes: Sequence[int] | None = None,
    complex_mode: ComplexMode = "raise",
    symmetrize: SymmetrizeMode = "auto",
    directed: Literal["auto"] | bool = "auto",
    edge_mask: Any = None,
    keep_sign: KeepSign = "both",
    min_distance_mm: float | None = None,
    edge_threshold: float | str | None = None,
    max_edges: int | None = None,
    strength_mode: StrengthMode = "total",
    symmetry_rtol: float = 1e-7,
    symmetry_atol: float = 1e-9,
) -> PreparedConnectome:
    """Prepare one immutable, auditable edge set for every plotting backend."""

    reduction = _validate_reduction(reduction)
    if isinstance(connectivity, np.ndarray) or not hasattr(connectivity, "get_data"):
        if any(selector is not None for selector in (freq, time, epoch)):
            raise ValueError("freq, time, and epoch selectors require MNE-Connectivity input")
        adapted = _adapt_numpy(
            connectivity, node_axes=node_axes, reduce_axes=reduce_axes, reduction=reduction
        )
    else:
        if node_axes is not None or reduce_axes is not None:
            raise ValueError("node_axes and reduce_axes apply only to NumPy input")
        adapted = _adapt_mne_connectivity(
            connectivity, freq=freq, time=time, epoch=epoch, reduction=reduction
        )
    matrix = _apply_complex_mode(adapted.values, complex_mode)
    matrix = np.array(
        _symmetrize(matrix, symmetrize, declared_symmetric=adapted.declared_symmetric),
        dtype=np.float64,
        copy=True,
    )
    node_count = matrix.shape[0]
    if geometry is not None and len(geometry.node_names) != node_count:
        raise ConnectivityShapeError(
            f"geometry has {len(geometry.node_names)} nodes but connectivity has {node_count}"
        )
    names = geometry.node_names if geometry is not None else adapted.node_names

    symmetric = np.allclose(
        matrix, matrix.T, rtol=symmetry_rtol, atol=symmetry_atol, equal_nan=True
    )
    if directed == "auto":
        is_directed = not symmetric
    elif isinstance(directed, (bool, np.bool_)):
        is_directed = bool(directed)
        if not is_directed and not symmetric:
            raise ValueError(
                "directed=False requires a symmetric matrix; choose an explicit symmetrize mode"
            )
    else:
        raise ValueError("directed must be True, False, or 'auto'")

    if keep_sign not in {"both", "positive", "negative"}:
        raise ValueError("keep_sign must be 'both', 'positive', or 'negative'")
    if min_distance_mm is not None:
        if not isinstance(min_distance_mm, (int, float, np.number)) or isinstance(
            min_distance_mm, bool
        ):
            raise ValueError("min_distance_mm must be a finite non-negative number")
        min_distance_mm = float(min_distance_mm)
        if not np.isfinite(min_distance_mm) or min_distance_mm < 0:
            raise ValueError("min_distance_mm must be a finite non-negative number")
        if geometry is None:
            raise ValueError("min_distance_mm requires geometry with surface_coords")
    if max_edges is not None:
        if isinstance(max_edges, bool) or not isinstance(max_edges, (int, np.integer)):
            raise ValueError("max_edges must be a non-negative integer or None")
        max_edges = int(max_edges)
        if max_edges < 0:
            raise ValueError("max_edges must be a non-negative integer or None")
    parsed_threshold = _parse_threshold(edge_threshold)

    mask = None
    if edge_mask is not None:
        mask = np.asarray(edge_mask, dtype=bool)
        if mask.shape != matrix.shape:
            raise ValueError(
                f"edge_mask must have shape {matrix.shape}; got {mask.shape}"
            )
        if not is_directed and not np.array_equal(mask, mask.T):
            raise ValueError("edge_mask must be symmetric for an undirected connectome")

    candidates = _candidate_edges(matrix, directed=is_directed)
    if mask is not None:
        candidates = [edge for edge in candidates if mask[edge[0], edge[1]]]
    if keep_sign == "positive":
        candidates = [edge for edge in candidates if edge[2] > 0]
    elif keep_sign == "negative":
        candidates = [edge for edge in candidates if edge[2] < 0]

    distances: dict[tuple[int, int], float] = {}
    if geometry is not None:
        for source, target, _ in candidates:
            distances[(source, target)] = float(
                np.linalg.norm(geometry.surface_coords[source] - geometry.surface_coords[target])
            )
    if min_distance_mm is not None:
        candidates = [
            edge
            for edge in candidates
            if distances[(edge[0], edge[1])] >= min_distance_mm
        ]

    resolved_threshold: float | None = None
    if parsed_threshold is not None:
        threshold_kind, threshold_value = parsed_threshold
        if threshold_kind == "percentile":
            if candidates:
                resolved_threshold = float(
                    np.percentile([abs(weight) for _, _, weight in candidates], threshold_value)
                )
        else:
            resolved_threshold = threshold_value
        if resolved_threshold is not None:
            if threshold_kind == "percentile":
                candidates = [
                    edge for edge in candidates if abs(edge[2]) > resolved_threshold
                ]
            else:
                candidates = [
                    edge for edge in candidates if abs(edge[2]) >= resolved_threshold
                ]

    # Zero weights participate in percentile population semantics, as they do
    # in Nilearn's dense connectome thresholding, but never become visual edges.
    candidates = [edge for edge in candidates if edge[2] != 0]
    candidates.sort(key=lambda value: (-abs(value[2]), value[0], value[1]))
    if max_edges is not None:
        candidates = candidates[:max_edges]
    edges = tuple(
        Edge(source, target, weight, distances.get((source, target)))
        for source, target, weight in candidates
    )
    visible = np.zeros_like(matrix, dtype=np.float64)
    for edge in edges:
        visible[edge.source, edge.target] = edge.weight
        if not is_directed:
            visible[edge.target, edge.source] = edge.weight
    strength = _node_strength(visible, directed=is_directed, mode=strength_mode)
    metadata = dict(adapted.metadata)
    metadata.update(
        {
            "complex_mode": complex_mode,
            "symmetrize": symmetrize,
            "directed": is_directed,
            "keep_sign": keep_sign,
            "edge_threshold": edge_threshold,
            "resolved_edge_threshold": resolved_threshold,
            "max_edges": max_edges,
            "strength_mode": strength_mode,
        }
    )
    if min_distance_mm is not None:
        metadata["min_distance_mm"] = min_distance_mm
        metadata["distance_coordinate"] = "surface_coords"
    return PreparedConnectome(
        matrix=matrix,
        visible_matrix=visible,
        edges=edges,
        node_strength=strength,
        directed=is_directed,
        node_names=names,
        metadata=metadata,
    )
