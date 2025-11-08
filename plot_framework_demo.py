from __future__ import annotations

import numpy as np

from shapes import UnitDisk, UnitSquare, Polygon
from plotting import render_to_file


def build_composition():
    circle = UnitDisk().scale(1.2).rotate(0.3).translate(-0.6, 0.1).with_color(0.2, 0.8, 0.5)
    square = UnitSquare().scale(1.3, 0.8).rotate(-0.2).translate(0.9, 0.7).with_color(0.9, 0.2, 0.2)
    quad = Polygon(np.array([[0.0, 1.3], [-1.3, -0.4], [0.2, -1.2], [1.2, 0.2], [0.0, 2.5]], dtype=float)).with_color(0.2, 0.45, 0.95)
    return (circle | square) | quad


def main():
    shape = build_composition()
    shape2 = shape.scale(0.5).rotate(np.pi/4)
    render_to_file(
        shape2,
        out_path="plots/framework_demo.png",
        title="Framework demo: (Circle ∪ Square) − Triangle",
        resolution=300,
        figsize=(6, 6),
        dpi=220,
    )
    print("Saved: plots/framework_demo.png")


if __name__ == "__main__":
    main()


