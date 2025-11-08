from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import os
import math
import numpy as np


class Shape:
    def sdf(self, point_xy: np.ndarray) -> float:
        raise NotImplementedError

    def contains(self, point_xy: np.ndarray) -> bool:
        return self.sdf(point_xy) <= 0.0

    # ---- Composition DSL ----
    def union(self, other: "Shape") -> "UnionN":
        return UnionN(self, other)

    def __or__(self, other: "Shape") -> "UnionN":
        return self.union(other)

    def intersect(self, other: "Shape") -> "IntersectionN":
        return IntersectionN(self, other)

    def __and__(self, other: "Shape") -> "IntersectionN":
        return self.intersect(other)

    def difference(self, other: "Shape") -> "Difference":
        return Difference(self, other)

    def __sub__(self, other: "Shape") -> "Difference":
        return self.difference(other)

    # ---- Transform helpers ----
    def transformed(self, T: "Affine2D") -> "Transformed":
        return Transformed(self, T)

    def translate(self, dx: float, dy: float) -> "Transformed":
        return self.transformed(Affine2D.from_translate(dx, dy))

    def scale(self, sx: float, sy: float | None = None) -> "Transformed":
        return self.transformed(Affine2D.from_scale(sx, sy))

    def rotate(self, theta_radians: float) -> "Transformed":
        return self.transformed(Affine2D.from_rotation(theta_radians))

    # ---- Color / evaluation ----
    def color_at(self, point_xy: np.ndarray) -> Optional[np.ndarray]:
        """
        Optional per-point color in RGB [0,1]. Default: None (no color).
        """
        return None

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Return (sdf, color) at a point. Default uses sdf() and color_at().
        """
        return self.sdf(point_xy), self.color_at(point_xy)

    def with_color(self, r: float, g: float, b: float) -> "Colored":
        return Colored(self, np.array([r, g, b], dtype=float))


@dataclass(frozen=True)
class Affine2D:
    """
    2D affine transform x -> A x + t
    """
    A: np.ndarray  # shape (2, 2)
    t: np.ndarray  # shape (2,)

    def __post_init__(self):
        if self.A.shape != (2, 2):
            raise ValueError("A must be 2x2")
        if self.t.shape != (2,):
            raise ValueError("t must be length-2")
        # Precompute inverse for efficient inverse application
        object.__setattr__(self, "_Ainv", np.linalg.inv(self.A))

    def apply(self, point_xy: np.ndarray) -> np.ndarray:
        return self.A @ point_xy + self.t

    def inverse_apply(self, point_xy: np.ndarray) -> np.ndarray:
        return self._Ainv @ (point_xy - self.t)

    # ---- Constructors and composition ----
    @staticmethod
    def identity() -> "Affine2D":
        return Affine2D(A=np.eye(2), t=np.zeros(2))

    @staticmethod
    def from_translate(dx: float, dy: float) -> "Affine2D":
        return Affine2D(A=np.eye(2), t=np.array([dx, dy], dtype=float))

    @staticmethod
    def from_scale(sx: float, sy: float | None = None) -> "Affine2D":
        if sy is None:
            sy = sx
        return Affine2D(A=np.array([[sx, 0.0], [0.0, sy]], dtype=float), t=np.zeros(2))

    @staticmethod
    def from_rotation(theta_radians: float) -> "Affine2D":
        c = math.cos(theta_radians)
        s = math.sin(theta_radians)
        return Affine2D(A=np.array([[c, -s], [s, c]], dtype=float), t=np.zeros(2))

    def then(self, after: "Affine2D") -> "Affine2D":
        """
        First apply self, then apply 'after'.
        y = after.apply(self.apply(x))
        """
        A_new = after.A @ self.A
        t_new = after.A @ self.t + after.t
        return Affine2D(A=A_new, t=t_new)


class Transformed(Shape):
    def __init__(self, shape: Shape, transform: Affine2D):
        self.shape = shape
        self.transform = transform

    def sdf(self, point_xy: np.ndarray) -> float:
        # Pullback: evaluate base shape at inverse-mapped point
        return self.shape.sdf(self.transform.inverse_apply(point_xy))

    def color_at(self, point_xy: np.ndarray) -> Optional[np.ndarray]:
        return self.shape.color_at(self.transform.inverse_apply(point_xy))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        pin = self.transform.inverse_apply(point_xy)
        d = self.shape.sdf(pin)
        c = self.shape.color_at(pin)
        return d, c


class UnitSquare(Shape):
    """
    Axis-aligned square centered at origin with side length 1.
    """
    def sdf(self, point_xy: np.ndarray) -> float:
        q = np.abs(point_xy) - 0.5
        outside = np.maximum(q, 0.0)
        # distance to outside + inside max component (if inside)
        return float(np.linalg.norm(outside, ord=2) + min(max(q[0], q[1]), 0.0))


class UnitDisk(Shape):
    """
    Unit circle centered at origin.
    """
    def sdf(self, point_xy: np.ndarray) -> float:
        return float(np.linalg.norm(point_xy) - 1.0)

class Colored(Shape):
    """
    Wrapper that assigns a constant RGB color to a shape.
    """
    def __init__(self, shape: Shape, rgb: np.ndarray):
        self.shape = shape
        rgb = np.array(rgb, dtype=float).reshape(3,)
        self.rgb = np.clip(rgb, 0.0, 1.0)

    def sdf(self, point_xy: np.ndarray) -> float:
        return self.shape.sdf(point_xy)

    def color_at(self, point_xy: np.ndarray) -> Optional[np.ndarray]:
        return self.rgb

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        return self.shape.sdf(point_xy), self.rgb


class HalfSpace(Shape):
    """
    { x : n·x + c <= 0 }, with n normalized
    """
    def __init__(self, normal_xy: np.ndarray, c: float):
        n = np.array(normal_xy, dtype=float)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            raise ValueError("HalfSpace normal must be non-zero")
        self.n = n / norm
        self.c = float(c)

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(self.n @ point_xy + self.c)


class UnionN(Shape):
    def __init__(self, *shapes: Shape):
        # flatten nested UnionN
        flat: list[Shape] = []
        for s in shapes:
            if isinstance(s, UnionN):
                flat.extend(s.shapes)
            else:
                flat.append(s)
        if len(flat) == 0:
            raise ValueError("UnionN requires at least one shape")
        self.shapes = flat

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(min(s.sdf(point_xy) for s in self.shapes))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        # Choose the contributor with minimal SDF
        best_d = None
        best_c = None
        for s in self.shapes:
            d, c = s.evaluate(point_xy)
            if best_d is None or d < best_d:
                best_d, best_c = d, c
        return float(best_d), best_c


class Union(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(min(self.a.sdf(point_xy), self.b.sdf(point_xy)))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        da, ca = self.a.evaluate(point_xy)
        db, cb = self.b.evaluate(point_xy)
        if da <= db:
            return da, ca
        else:
            return db, cb


class IntersectionN(Shape):
    def __init__(self, *shapes: Shape):
        # flatten nested IntersectionN
        flat: list[Shape] = []
        for s in shapes:
            if isinstance(s, IntersectionN):
                flat.extend(s.shapes)
            else:
                flat.append(s)
        if len(flat) == 0:
            raise ValueError("IntersectionN requires at least one shape")
        self.shapes = flat

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(max(s.sdf(point_xy) for s in self.shapes))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        # Choose the contributor with maximal SDF
        best_d = None
        best_c = None
        for s in self.shapes:
            d, c = s.evaluate(point_xy)
            if best_d is None or d > best_d:
                best_d, best_c = d, c
        return float(best_d), best_c


class Intersection(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(max(self.a.sdf(point_xy), self.b.sdf(point_xy)))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        da, ca = self.a.evaluate(point_xy)
        db, cb = self.b.evaluate(point_xy)
        if da >= db:
            return da, ca
        else:
            return db, cb


class Difference(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a = a
        self.b = b

    def sdf(self, point_xy: np.ndarray) -> float:
        return float(max(self.a.sdf(point_xy), -self.b.sdf(point_xy)))

    def evaluate(self, point_xy: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        da, ca = self.a.evaluate(point_xy)
        db, cb = self.b.evaluate(point_xy)
        d = float(max(da, -db))
        # Difference region is subset of A; color with A's color for inside
        return d, ca


def sample_sdf_grid(
    shape: Shape,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    resolution: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample an SDF on a regular grid suitable for contouring.
    """
    xs = np.linspace(xlim[0], xlim[1], resolution)
    ys = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X, dtype=float)
    # Vectorized-ish per-grid evaluation (loop across first axis to control memory)
    for i in range(X.shape[0]):
        row = np.stack([X[i, :], Y[i, :]], axis=1)
        Z[i, :] = np.array([shape.sdf(p) for p in row], dtype=float)
    return X, Y, Z


