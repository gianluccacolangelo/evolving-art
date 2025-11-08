from __future__ import annotations

import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from shapes import (
    UnitSquare,
    UnitDisk,
    Colored,
    HalfSpace,
    Affine2D,
    Transformed,
    Union,  # kept for compatibility
    Intersection,  # kept for compatibility
    Difference,  # binary
    UnionN,
    IntersectionN,
    sample_sdf_grid,
)


def bbox_for_transformed_square(T: Affine2D, margin: float = 0.2) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    corners = np.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5],
            [-0.5, 0.5],
        ]
    )
    pts = np.stack([T.apply(c) for c in corners])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return (xmin - margin, xmax + margin), (ymin - margin, ymax + margin)


def bbox_for_transformed_disk(T: Affine2D, margin: float = 0.2) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # The image of the unit disk under x -> A x + t is an ellipse centered at t
    # Bounding box extents are given by |A[:,0]| + |A[:,1]| per coordinate
    A = T.A
    t = T.t
    extent = np.abs(A[:, 0]) + np.abs(A[:, 1])
    xmin, xmax = (t[0] - extent[0] - margin, t[0] + extent[0] + margin)
    ymin, ymax = (t[1] - extent[1] - margin, t[1] + extent[1] + margin)
    return (xmin, xmax), (ymin, ymax)


def build_parallelogram():
    # Anchor (translation)
    c = np.array([1.5, 0.8])
    # Edge vectors (columns of A)
    u = np.array([2.2, 0.4])
    v = np.array([0.6, 1.7])
    A = np.column_stack([u, v])
    T = Affine2D(A=A, t=c)
    return Transformed(UnitSquare(), T), T


def build_ellipse():
    center = np.array([-0.5, 0.0])
    a1 = np.array([3.2, 0.2])   # semi-axis vector 1
    a2 = np.array([0.8, 2.3])   # semi-axis vector 2 (tilted)
    A = np.column_stack([a1, a2])
    T = Affine2D(A=A, t=center)
    return Transformed(UnitDisk(), T), T


def triangle_from_vertices(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Intersection:
    verts = [np.array(v0, dtype=float), np.array(v1, dtype=float), np.array(v2, dtype=float)]
    def inward_normal(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
        # Edge from pa->pb, outward normal is rotate +90: (dy, -dx), inward points toward pc
        edge = pb - pa
        n_out = np.array([edge[1], -edge[0]], dtype=float)
        # Flip if pointing toward pc (dot > 0) to get inward
        if n_out @ (pc - pa) > 0:
            n_in = -n_out
        else:
            n_in = n_out
        return n_in
    n0 = inward_normal(verts[0], verts[1], verts[2])
    n1 = inward_normal(verts[1], verts[2], verts[0])
    n2 = inward_normal(verts[2], verts[0], verts[1])
    H0 = HalfSpace(n0, c=-n0 @ verts[0])
    H1 = HalfSpace(n1, c=-n1 @ verts[1])
    H2 = HalfSpace(n2, c=-n2 @ verts[2])
    return Intersection(Intersection(H0, H1), H2)


def bbox_for_triangle(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, margin: float = 0.2):
    pts = np.stack([v0, v1, v2])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return (xmin - margin, xmax + margin), (ymin - margin, ymax + margin)


def plot_shape(ax, shape, xlim, ylim, title: str, color="#1f77b4", resolution=400):
    X, Y, Z = sample_sdf_grid(shape, xlim, ylim, resolution=resolution)
    # Filled interior
    ax.contourf(X, Y, Z, levels=[-1e9, 0.0], colors=[color], alpha=0.25, antialiased=True)
    # Boundary
    ax.contour(X, Y, Z, levels=[0.0], colors=[color], linewidths=2.0, antialiased=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2, linestyle="--")


def plot_shape_colored(ax, shape, xlim, ylim, title: str, resolution=400, edge_color="black"):
    # Sample SDF and colors
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X, dtype=float)
    RGBA = np.zeros((resolution, resolution, 4), dtype=float)
    fallback = np.array([0.6, 0.6, 0.6], dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            p = np.array([X[i, j], Y[i, j]])
            d, c = shape.evaluate(p)
            Z[i, j] = d
            if d <= 0.0:
                col = c if c is not None else fallback
                RGBA[i, j, :3] = np.clip(col, 0.0, 1.0)
                RGBA[i, j, 3] = 1.0
            else:
                RGBA[i, j, :3] = 0.0
                RGBA[i, j, 3] = 0.0
    # Draw filled color using imshow
    ax.imshow(
        RGBA,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="lower",
        interpolation="bilinear",
        aspect="equal",
    )
    # Boundary
    ax.contour(X, Y, Z, levels=[0.0], colors=[edge_color], linewidths=1.5, antialiased=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.2, linestyle="--")


class UserCompositeShape:
    """
    Example of a user-defined composite shape using the DSL:
      - Build with primitives
      - Apply transforms via .translate/.scale/.rotate
      - Combine via | (union), & (intersection), - (difference) or UnionN/IntersectionN
    """
    def __init__(self):
        # Base shapes with colors
        ellipse = UnitDisk().scale(1.0, 0.6).rotate(0.4).translate(-0.3, 0.0).scale(2.0).with_color(0.2, 0.8, 0.5)
        square = UnitSquare().scale(1.4, 0.9).rotate(-0.2).translate(1.2, 0.6).with_color(0.9, 0.2, 0.2)
        tri = triangle_from_vertices(np.array([0.0, 1.4]), np.array([-1.2, -0.6]), np.array([1.3, -0.7])).with_color(0.2, 0.4, 0.9)
        # Compose: (ellipse ∪ tri) − square
        self.shape = (ellipse | tri) - square

    def get(self):
        return self.shape


def main():
    os.makedirs("plots", exist_ok=True)

    # Build shapes
    parallelogram, T_par = build_parallelogram()
    ellipse, T_ell = build_ellipse()
    v0 = np.array([0.0, 1.4])
    v1 = np.array([-1.2, -0.6])
    v2 = np.array([1.3, -0.7])
    triangle = triangle_from_vertices(v0, v1, v2)

    # Bboxes
    pxlim, pylim = bbox_for_transformed_square(T_par, margin=0.3)
    exlim, eylim = bbox_for_transformed_disk(T_ell, margin=0.3)
    txlim, tylim = bbox_for_triangle(v0, v1, v2, margin=0.3)

    # Individual figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_shape(axs[0], parallelogram, pxlim, pylim, "Parallelogram (affine of unit square)", color="#1f77b4")
    plot_shape(axs[1], ellipse, exlim, eylim, "Ellipse (affine of unit disk)", color="#d62728")
    plot_shape(axs[2], triangle, txlim, tylim, "Triangle (intersection of half-spaces)", color="#2ca02c")
    fig.suptitle("Minimal geometric primitives via SDF + affine + set ops")
    fig.savefig("plots/shapes_basic.png", dpi=200)
    plt.close(fig)

    # Composite example using DSL: (ellipse ∪ triangle) \ parallelogram
    composite = (ellipse | triangle) - parallelogram
    # Choose a bbox that covers all three
    xmin = min(pxlim[0], exlim[0], txlim[0]) - 0.2
    xmax = max(pxlim[1], exlim[1], txlim[1]) + 0.2
    ymin = min(pylim[0], eylim[0], tylim[0]) - 0.2
    ymax = max(pylim[1], eylim[1], tylim[1]) + 0.2
    cxlim, cylim = (xmin, xmax), (ymin, ymax)

    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    plot_shape(ax2, composite, cxlim, cylim, "(Ellipse ∪ Triangle) − Parallelogram", color="#9467bd", resolution=500)
    fig2.savefig("plots/shapes_composite.png", dpi=220)
    plt.close(fig2)

    # User-defined composite class example
    user_comp = UserCompositeShape().get()
    # Autosize bbox for a general shape by sampling a coarse grid around origin
    # For demo simplicity, reuse bbox covering earlier shapes
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    plot_shape_colored(ax3, user_comp, cxlim, cylim, "UserCompositeShape example (colored)", resolution=500)
    fig3.savefig("plots/shapes_user_composite.png", dpi=220)
    plt.close(fig3)

    print("Saved plots to:")
    print(" - plots/shapes_basic.png")
    print(" - plots/shapes_composite.png")
    print(" - plots/shapes_user_composite.png")


if __name__ == "__main__":
    main()


