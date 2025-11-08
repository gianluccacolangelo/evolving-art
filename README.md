# evolving-art
A framework to evolve art pieces with a variety of optimizers 

## Minimal geometric primitives + plotting

This repo includes a minimal SDF-based geometry core with affine transforms and set operations, plus a plotting script that renders examples:

- Parallelogram (affine image of the unit square)
- Ellipse (affine image of the unit disk)
- Triangle (intersection of three half-spaces)
- A composite: (Ellipse ∪ Triangle) − Parallelogram

## Composition framework (DSL)

You can build complex shapes by composing primitives with operators and transform helpers:

- Transforms on any `Shape`:
  - `shape.translate(dx, dy)`, `shape.scale(sx[, sy])`, `shape.rotate(theta)`
- Set operations (all return a `Shape`):
  - `a | b` (union), `a & b` (intersection), `a - b` (difference)
  - `UnionN(a, b, c, ...)` and `IntersectionN(...)` for n-ary combos

Example user-defined composite:

```python
from shapes import UnitSquare, UnitDisk, UnionN, IntersectionN
import numpy as np

class MyComposite:
    def __init__(self):
        circle = UnitDisk().scale(1.2).translate(-0.5, 0.0)
        square = UnitSquare().scale(1.0, 0.6).rotate(0.3).translate(0.8, 0.6)
        self.shape = (circle | square) - UnitDisk().scale(0.5).translate(0.2, 0.1)

    def get(self):
        return self.shape
```

See `plot_shapes.py` for a working example (`UserCompositeShape`) and how to render it.

### Setup (conda)

```bash
conda env create -f environment.yml
```

### Generate plots

```bash
conda run -n evolving-art python plot_shapes.py
```

Outputs:
- `plots/shapes_basic.png`
- `plots/shapes_composite.png`
 - `plots/shapes_user_composite.png`

Tweaks:
- Edit `plot_shapes.py` to change the affine transforms, triangle vertices, or composite expression.
- The core SDF and transforms live in `shapes/geometry.py`.
