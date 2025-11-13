from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from evolution.genome import CompositionGenome
from shapes import Shape, UnionN

from plotting.vectorizer import save_genome_as_svg, save_genome_as_png, draw_genome_on_axis

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
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists, _ = shape.evaluate_many(pts)
    inside_mask = dists <= 0.0
    if not np.any(inside_mask):
        return (xmin, xmax), (ymin, ymax)
    inside_pts = pts[inside_mask]
    pxmin, pymin = inside_pts.min(axis=0)
    pxmax, pymax = inside_pts.max(axis=0)
    return (float(pxmin - margin), float(pxmax + margin)), (float(pymin - margin), float(pymax + margin))


def sample_shape_rgba(
    shape: Union[Shape, Iterable[Shape]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a shape (or list of shapes) over a grid, returning X, Y, Z(sdf), RGBA.
    Vectorized across the full grid using Shape.evaluate_many.
    """
    shape = _as_shape(shape)
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists, colors = shape.evaluate_many(pts)
    Z = dists.reshape(X.shape)
    RGBA = np.zeros((resolution, resolution, 4), dtype=float)
    inside_mask = (Z <= 0.0)
    if colors is None:
        c = np.tile(np.array([0.6, 0.6, 0.6], dtype=float).reshape(1, 3), (pts.shape[0], 1))
    else:
        c = np.clip(np.asarray(colors, dtype=float).reshape(-1, 3), 0.0, 1.0)
    Cimg = c.reshape(resolution, resolution, 3)
    RGBA[:, :, :3] = np.where(inside_mask[:, :, None], Cimg, 0.0)
    RGBA[:, :, 3] = inside_mask.astype(float)
    return X, Y, Z, RGBA


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
    show_axes: bool = False,
    show_grid: bool = False,
    frame_only: bool = True,
) -> None:
    shape = _as_shape(shape)
    if xlim is None or ylim is None:
        xlim, ylim = autosize_bounds(shape)

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
    if show_axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(show_grid, alpha=0.2, linestyle="--")
        if frame_only:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    else:
        ax.axis("off")

def render_to_file(
    genome: CompositionGenome,
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
    show_axes: bool = False,
    show_grid: bool = False,
    frame_only: bool = False,
    dpi: int = 220,
    format: Optional[str] = None,
    transparent: bool = False,
    return_image: bool = False,
) -> Optional[Image.Image]:
    
 
    
    if format == "svg":
        if out_path is None: 
            raise ValueError("SVG rendering requires an output path (out_path).")
        save_genome_as_svg(genome, filename=out_path)
        return None
    
    if format == "png":
        if return_image:
            return save_genome_as_png(genome, filename=None, resolution=resolution) 
        else:
            # Renderizar a archivo
            if out_path is None: 
                raise ValueError("PNG (vector) rendering requires an output path (out_path) when return_image is False.")
            save_genome_as_png(genome, filename=out_path, resolution=resolution)
            return None
    
    return None
        # lo dejo por las dudas
        #fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        #render_to_axes(
        #    ax,
        #    genome.to_shape(),
        #   xlim=xlim,
        #    ylim=ylim,
        #    resolution=resolution,
        #    title=title,
        #    draw_edges=draw_edges,
        #    edge_color=edge_color,
        #    edge_width=edge_width,
        #    interpolation=interpolation,
        #    parallel=parallel,
        #    workers=workers,
        #    show_axes=show_axes,
        #    show_grid=show_grid,
        #    frame_only=frame_only,
        #fig.savefig(out_path, dpi=dpi, format=format, transparent=transparent)
        #plt.close(fig)

def render_population_grid(
    population: Sequence[CompositionGenome],
    out_path: str,
    cols: int = 4,
    resolution: int = 100,
    figsize_per_cell: Tuple[float, float] = (3.0, 3.0),
    draw_edges: bool = False,
    use_vector: bool = True,
) -> None:
    """
    Renders a grid of genomes.
    If use_vector=True, uses Shapely vectorizer.
    If use_vector=False, uses pixel-based renderer.
    """
    n = len(population)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols
    fig_w = figsize_per_cell[0] * cols
    fig_h = figsize_per_cell[1] * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor('white')
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for idx, genome in enumerate(population):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        
        ax.set_facecolor('white')
        
        try:
            if use_vector:
                draw_genome_on_axis(ax, genome)
                
                ax.axis("on")

                ax.set_xticks([])
                ax.set_yticks([])

                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')
                    spine.set_linewidth(1.0)
                
                if idx < n:
                    ax.set_title(f"{idx}", fontsize=10, color='black')
            else:
                shape = genome.to_shape()
                render_to_axes(
                    ax,
                    shape,
                    resolution=resolution,
                    title=f"{idx}",
                    draw_edges=draw_edges,
                    show_axes=True,   
                    show_grid=False,
                    frame_only=True, 
                )
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
            ax.axis("off")
            
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fmt = "svg" if out_path.endswith(".svg") else "png"
    
    fig.savefig(out_path, dpi=200, format=fmt, transparent=False, facecolor='white')
    plt.close(fig)