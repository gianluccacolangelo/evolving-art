# evolving-art
A framework to evolve art pieces with a variety of optimizers 

![Composition grid example](plots/shapes_composition.png)
## Framework essentials

- **Primitives** (2D, SDF-based):
  - `UnitDisk()` — unit circle at origin
  - `UnitSquare()` — unit square centered at origin
  - `Polygon(vertices)` — arbitrary simple polygon (N≥3)
  - Optional: `HalfSpace(n, c)` for half-plane constructions

- **Chainable transforms** on any `Shape`:
  - `shape.scale(sx[, sy]).rotate(theta).translate(dx, dy)`
  - Colors (optional, RGB in [0,1]): `shape.with_color(r, g, b)`

- **Set composition** (returns a `Shape`):
  - `A | B` union, `A & B` intersection, `A - B` difference
  - Variadic: `UnionN(A, B, C, ...)`, `IntersectionN(...)`

Minimal example:

```python
from shapes import UnitDisk, UnitSquare, Polygon

circle = UnitDisk().scale(1.2).translate(-0.6, 0.1).with_color(0.2, 0.8, 0.5)
square = UnitSquare().scale(1.0, 0.6).rotate(0.3).translate(0.8, 0.6).with_color(0.9, 0.2, 0.2)
tri = Polygon([[0.0, 1.3], [-1.3, -0.6], [1.2, -0.7]]).with_color(0.2, 0.45, 0.95)
shape = (circle | square) - tri
```

### Composability
- Every transform and set operator returns a `Shape`. That means you can apply transforms to already-composed shapes and keep nesting indefinitely.
- Colors are preserved through transforms and set operations via a general color algebra.

Example (transforming a composite):

```python
small_rotated = shape.scale(0.5).rotate(0.6).translate(0.2, -0.3)
```

### Setup (conda)

```bash
conda env create -f environment.yml
```

## Plotting

- Render any shape (or list of shapes) with auto-bounds:

```python
from plotting import render_to_file
render_to_file(shape, out_path="plots/my_shape.png", title="My composition")
```

- No outline by default; enable if desired: `draw_edges=True`.
- Pass explicit `xlim`/`ylim` to skip auto-bounds.
- You can also pass a list: `render_to_file([shape1, shape2], ...)`.

## Plotting framework

A small plotting module provides auto-bounds and rendering for any composed `Shape` (or a list of shapes):

```python
from plotting import render_to_file, autosize_bounds
from shapes import UnitDisk, UnitSquare, Polygon

shape = (UnitDisk().scale(1.2).with_color(0.2, 0.8, 0.5) | UnitSquare().translate(0.8, 0.4).with_color(0.9, 0.2, 0.2))
render_to_file(shape, out_path="plots/framework_demo.png", title="My composition")
```

- Auto-bounds: If you don’t pass `xlim`/`ylim`, the renderer samples a coarse grid to choose a tight extent.
- You can also pass a list of shapes; it will be unioned automatically for rendering.

See `plot_framework_demo.py` for a runnable example. It outputs `plots/framework_demo.png`.
