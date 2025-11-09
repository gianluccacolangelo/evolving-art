from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

from shapes import Shape, UnionN


def _as_shape(shape_or_shapes: Union[Shape, Iterable[Shape]]) -> Shape:
    if isinstance(shape_or_shapes, Shape):
        return shape_or_shapes
    shapes = list(shape_or_shapes)
    if not shapes:
        raise ValueError("No shapes provided")
    return UnionN(*shapes)


def autosize_bounds(
    shape: Union[Shape, Iterable[Shape]],
    initial_bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0),
    coarse_resolution: int = 100,
    margin: float = 0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Estimate tight x/y limits by coarse sampling of sdf/evaluate.
    Returns (xlim, ylim).
    """
    shape = _as_shape(shape)
    xmin, xmax, ymin, ymax = initial_bounds
    xs = np.linspace(xmin, xmax, coarse_resolution)
    ys = np.linspace(ymin, ymax, coarse_resolution)
    X, Y = np.meshgrid(xs, ys)
    inside_points = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            d, _ = shape.evaluate(np.array([X[i, j], Y[i, j]], dtype=float))
            if d <= 0.0:
                inside_points.append((X[i, j], Y[i, j]))
    if not inside_points:
        return (xmin, xmax), (ymin, ymax)
    pts = np.array(inside_points, dtype=float)
    pxmin, pymin = pts.min(axis=0)
    pxmax, pymax = pts.max(axis=0)
    return (float(pxmin - margin), float(pxmax + margin)), (float(pymin - margin), float(pymax + margin))


def sample_shape_rgba(
    shape: Union[Shape, Iterable[Shape]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a shape (or list of shapes) over a grid, returning X, Y, Z(sdf), RGBA.
    """
    shape = _as_shape(shape)
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X, dtype=float)
    RGBA = np.zeros((resolution, resolution, 4), dtype=float)
    fallback = np.array([0.6, 0.6, 0.6], dtype=float)
    for i in range(resolution):
        row_pts = np.stack([X[i, :], Y[i, :]], axis=1)
        evals = [shape.evaluate(pt) for pt in row_pts]
        d_row = np.fromiter((ev[0] for ev in evals), dtype=float, count=row_pts.shape[0])
        Z[i, :] = d_row
        inside_mask = d_row <= 0.0
        if inside_mask.any():
            # Build color row with fallback
            c_list = []
            for _, c in evals:
                if c is None:
                    c_list.append(fallback)
                else:
                    c_arr = np.asarray(c, dtype=float).reshape(3,)
                    c_list.append(np.clip(c_arr, 0.0, 1.0))
            c_row = np.vstack(c_list)
            RGBA[i, :, :3] = np.where(inside_mask[:, None], c_row, 0.0)
            RGBA[i, :, 3] = inside_mask.astype(float)
        else:
            RGBA[i, :, :3] = 0.0
            RGBA[i, :, 3] = 0.0
    return X, Y, Z, RGBA


# ---- Multiprocessing helpers ----
_SHAPE = None
_XS = None
_YS = None
_FALLBACK = None


def _init_worker(shape_obj: Shape, xlim: Tuple[float, float], ylim: Tuple[float, float], resolution: int, fallback: np.ndarray):
    global _SHAPE, _XS, _YS, _FALLBACK
    _SHAPE = shape_obj
    _XS = np.linspace(xlim[0], xlim[1], resolution)
    _YS = np.linspace(ylim[0], ylim[1], resolution)
    _FALLBACK = fallback


def _eval_row(i: int):
    xs = _XS
    y = _YS[i]
    row_pts = np.stack([xs, np.full_like(xs, y)], axis=1)
    evals = [_SHAPE.evaluate(pt) for pt in row_pts]
    d_row = np.fromiter((ev[0] for ev in evals), dtype=float, count=row_pts.shape[0])
    inside_mask = d_row <= 0.0
    c_list = []
    for _, c in evals:
        if c is None:
            c_list.append(_FALLBACK)
        else:
            c_arr = np.asarray(c, dtype=float).reshape(3,)
            c_list.append(np.clip(c_arr, 0.0, 1.0))
    c_row = np.vstack(c_list)
    alpha_row = inside_mask.astype(np.float64)
    return i, d_row, alpha_row, c_row


def render_to_axes(
    ax,
    shape: Union[Shape, Iterable[Shape]],
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 512,
    title: Optional[str] = None,
    draw_edges: bool = False,
    edge_color: str = "black",
    edge_width: float = 1.5,
    interpolation: str = "none",
    parallel: bool = True,
    workers: Optional[int] = None,
) -> None:
    shape = _as_shape(shape)
    if xlim is None or ylim is None:
        xlim, ylim = autosize_bounds(shape)
    # Choose between single-process and parallel sampling
    if parallel:
        if workers is None:
            workers = max(1, (os.cpu_count() or 2) - 0)
        xs = np.linspace(xlim[0], xlim[1], resolution)
        ys = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = np.empty_like(X, dtype=float)
        RGBA = np.zeros((resolution, resolution, 4), dtype=float)
        fallback = np.array([0.6, 0.6, 0.6], dtype=float)
        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(shape, xlim, ylim, resolution, fallback)) as pool:
            for i, d_row, alpha_row, c_row in pool.imap_unordered(_eval_row, range(resolution), chunksize=max(1, resolution // (workers * 8))):
                Z[i, :] = d_row
                RGBA[i, :, :3] = np.where(alpha_row[:, None] > 0.0, c_row, 0.0)
                RGBA[i, :, 3] = alpha_row
    else:
        X, Y, Z, RGBA = sample_shape_rgba(shape, xlim, ylim, resolution=resolution)
    ax.imshow(
        RGBA,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        interpolation=interpolation,
        aspect="equal",
    )
    if draw_edges:
        ax.contour(X, Y, Z, levels=[0.0], colors=[edge_color], linewidths=edge_width, antialiased=True)
    if title:
        ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2, linestyle="--")


def render_to_file(
    shape: Union[Shape, Iterable[Shape]],
    out_path: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    resolution: int = 600,
    figsize: Tuple[float, float] = (6.0, 6.0),
    title: Optional[str] = None,
    draw_edges: bool = False,
    edge_color: str = "black",
    edge_width: float = 1.5,
    interpolation: str = "none",
    parallel: bool = True,
    workers: Optional[int] = None,
    dpi: int = 220,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    render_to_axes(
        ax,
        shape,
        xlim=xlim,
        ylim=ylim,
        resolution=resolution,
        title=title,
        draw_edges=draw_edges,
        edge_color=edge_color,
        edge_width=edge_width,
        interpolation=interpolation,
        parallel=parallel,
        workers=workers,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


