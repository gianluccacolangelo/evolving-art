from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapes import Shape, UnitDisk, UnitSquare, Polygon
from plotting.renderer import sample_shape_rgba


@dataclass
class TreeNode:
    kind: str  # "op" or "primitive"
    label: str
    color_rgb: Optional[Tuple[float, float, float]] = None
    children: List["TreeNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    primitive_payload: Optional[dict] = None


def _parse_tree(node_obj: dict) -> TreeNode:
    """
    Parse a node object from the params JSON into a TreeNode.
    - Operator nodes have keys: "op", "children"
    - Primitive nodes appear wrapped as {"primitive": {...}}
    """
    if "primitive" in node_obj:
        prim = node_obj["primitive"]
        kind = prim.get("kind", "primitive")
        color = prim.get("color_rgb", [0.7, 0.7, 0.7])
        try:
            color_tuple = (float(color[0]), float(color[1]), float(color[2]))
        except Exception:
            color_tuple = (0.7, 0.7, 0.7)
        label = kind
        # Optional: add extra hints to label (e.g., polygon vertex count)
        if kind == "polygon" and prim.get("polygon_vertices") is not None:
            try:
                n = len(prim["polygon_vertices"])
                label = f"{kind}({n})"
            except Exception:
                pass
        return TreeNode(kind="primitive", label=label, color_rgb=color_tuple, children=[], primitive_payload=prim)
    # Otherwise it's an operator node
    op = node_obj["op"]
    ch = node_obj.get("children", [])
    children = [_parse_tree(child) for child in ch]
    return TreeNode(kind="op", label=str(op), color_rgb=None, children=children)


def _compute_depth(root: TreeNode) -> int:
    if not root.children:
        return 1
    return 1 + max(_compute_depth(ch) for ch in root.children)


def _assign_positions(root: TreeNode, y_step: float = 1.6) -> None:
    """
    Assign x/y positions to nodes for a tidy tree layout using a simple
    in-order leaf indexing. x spacing is uniform per leaf; y is depth-based.
    """
    # First pass: assign x to leaves incrementally, return next_x and collect leaf xs
    def assign_x(node: TreeNode, depth: int, next_x: float) -> Tuple[float, List[float]]:
        if not node.children:
            node.x = next_x
            node.y = -depth * y_step
            return next_x + 1.0, [node.x]
        xs: List[float] = []
        for ch in node.children:
            next_x, child_xs = assign_x(ch, depth + 1, next_x)
            xs.extend(child_xs)
        node.x = sum(xs) / len(xs)
        node.y = -depth * y_step
        return next_x, [node.x]

    assign_x(root, 0, 0.0)


def _gather_edges(node: TreeNode) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for ch in node.children:
        edges.append(((node.x, node.y), (ch.x, ch.y)))
        edges.extend(_gather_edges(ch))
    return edges


def _draw_tree(root: TreeNode,
               node_radius: float = 0.22,
               op_facecolor: Tuple[float, float, float] = (0.9, 0.9, 0.9),
               op_edgecolor: Tuple[float, float, float] = (0.2, 0.2, 0.2),
               prim_edgecolor: Tuple[float, float, float] = (0.2, 0.2, 0.2),
               font_size: int = 7,
               figsize_scale: float = 0.9) -> plt.Figure:
    # Compute figure bounds
    leaves: List[TreeNode] = []
    def collect_leaves(n: TreeNode):
        if not n.children:
            leaves.append(n)
        for c in n.children:
            collect_leaves(c)
    collect_leaves(root)
    if leaves:
        x_min = min(l.x for l in leaves) - 1.0
        x_max = max(l.x for l in leaves) + 1.0
    else:
        x_min, x_max = -1.0, 1.0
    depth = _compute_depth(root)
    y_min = -depth * 1.6 - 0.6
    y_max = 0.6
    width = x_max - x_min
    height = y_max - y_min
    # Make figure size proportional to content, but clamp reasonably
    fig_w = max(4.0, min(18.0, figsize_scale * width * 1.2))
    fig_h = max(3.0, min(12.0, figsize_scale * height * 0.9))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Draw edges
    for (x0, y0), (x1, y1) in _gather_edges(root):
        ax.plot([x0, x1], [y0, y1], color=(0.5, 0.5, 0.5), linewidth=1.0, zorder=1)

    # Helper: build a Shape from primitive payload (ignoring translation for thumbnail)
    def _shape_from_primitive_payload(prim: dict) -> Shape:
        kind = prim.get("kind", "disk")
        if kind == "disk":
            base = UnitDisk()
        elif kind == "square":
            base = UnitSquare()
        elif kind == "polygon":
            verts = prim.get("polygon_vertices")
            base = Polygon(verts) if verts is not None else UnitSquare()
        else:
            base = UnitSquare()
        color = prim.get("color_rgb", [0.7, 0.7, 0.7])
        try:
            r, g, b = float(color[0]), float(color[1]), float(color[2])
        except Exception:
            r, g, b = 0.7, 0.7, 0.7
        s = base.with_color(r, g, b)
        tf = prim.get("transform", {})
        sx = float(tf.get("sx", 1.0))
        sy = float(tf.get("sy", 1.0))
        theta = float(tf.get("theta", 0.0))
        # For thumbnails, omit translation so the primitive is centered in its node
        s = s.scale(sx, sy).rotate(theta)
        return s

    # Draw nodes
    def draw_node(n: TreeNode):
        if n.kind == "op":
            # Render operator as labeled circle node
            circ = Circle((n.x, n.y), node_radius, facecolor=op_facecolor, edgecolor=op_edgecolor, linewidth=1.0, zorder=3)
            ax.add_patch(circ)
            ax.text(n.x, n.y, n.label, ha="center", va="center", fontsize=font_size + 4, fontweight="bold", zorder=4)
        else:
            # Render actual primitive thumbnail
            # Sample a generous local frame so scaled primitives fit
            xlim_local = (-2.2, 2.2)
            ylim_local = (-2.2, 2.2)
            try:
                shape = _shape_from_primitive_payload(n.primitive_payload or {})
                _, _, _, RGBA = sample_shape_rgba(shape, xlim_local, ylim_local, resolution=96)
                # Place the thumbnail centered at (n.x, n.y)
                half = node_radius * 2.2
                extent = (n.x - half, n.x + half, n.y - half, n.y + half)
                ax.imshow(RGBA, extent=extent, origin="lower", interpolation="bilinear", zorder=3)
            except Exception:
                # Fallback: label only
                ax.text(n.x, n.y, n.label, ha="center", va="center", fontsize=font_size, zorder=4)
        # Recurse to children
        for c in n.children:
            draw_node(c)
    draw_node(root)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _load_composition_from_params(params_path: str, index: int) -> dict:
    with open(params_path, "r") as f:
        data = json.load(f)
    # Expect structure: data -> population -> list of items with {"index": i, "composition": {...}}
    pop = data.get("data", {}).get("population", [])
    if not pop:
        raise ValueError("No population found in params JSON.")
    # Prefer item with matching 'index' if present, otherwise treat 'index' as position
    by_index = {int(item.get("index", i)): item for i, item in enumerate(pop)}
    if index in by_index:
        comp = by_index[index].get("composition")
    else:
        if index < 0 or index >= len(pop):
            raise IndexError(f"Index {index} out of range for population of size {len(pop)}")
        comp = pop[index].get("composition")
    if comp is None:
        raise ValueError("Selected population item missing 'composition'.")
    return comp


def visualize_params_tree(params_path: str,
                          index: int = 0,
                          out_path: Optional[str] = None,
                          dpi: int = 200,
                          show: bool = False) -> str:
    comp = _load_composition_from_params(params_path, index)
    root = _parse_tree(comp)
    _assign_positions(root)
    fig = _draw_tree(root)
    if out_path is None:
        base_dir = os.path.dirname(params_path)
        base_name = os.path.splitext(os.path.basename(params_path))[0]
        out_path = os.path.join(base_dir, f"{base_name}_tree_idx_{index}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    if show:
        # Optional immediate view: re-render to screen
        fig2 = _draw_tree(root)
        plt.show()
        plt.close(fig2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize composition tree from params JSON.")
    parser.add_argument("params_json", help="Path to *_params.json")
    parser.add_argument("--index", type=int, default=0, help="Population index to visualize (default: 0)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: next to JSON)")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI (default: 200)")
    parser.add_argument("--show", action="store_true", help="Also display the tree window")
    args = parser.parse_args()

    out_path = visualize_params_tree(
        params_path=args.params_json,
        index=args.index,
        out_path=args.out,
        dpi=args.dpi,
        show=args.show,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


