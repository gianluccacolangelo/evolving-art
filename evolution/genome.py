from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional, List, Tuple, Union
import math
import numpy as np

from shapes import Shape, UnitDisk, UnitSquare, Polygon, UnionN, IntersectionN, Difference


PrimitiveKind = Literal["disk", "square", "polygon"]


@dataclass(frozen=True)
class TransformParams:
    sx: float
    sy: float
    theta: float
    dx: float
    dy: float

    def apply_to(self, shape: Shape) -> Shape:
        transformed = shape.scale(self.sx, self.sy).rotate(self.theta).translate(self.dx, self.dy)
        return transformed


@dataclass(frozen=True)
class PrimitiveGene:
    kind: PrimitiveKind
    transform: TransformParams
    color_rgb: np.ndarray
    polygon_vertices: Optional[np.ndarray] = None

    def to_shape(self) -> Shape:
        if self.kind == "disk":
            base = UnitDisk()
        elif self.kind == "square":
            base = UnitSquare()
        elif self.kind == "polygon":
            if self.polygon_vertices is None:
                raise ValueError("polygon gene missing vertices")
            base = Polygon(self.polygon_vertices)
        else:
            raise ValueError(f"unknown primitive kind: {self.kind}")
        colored = base.with_color(float(self.color_rgb[0]), float(self.color_rgb[1]), float(self.color_rgb[2]))
        return self.transform.apply_to(colored)


@dataclass(frozen=True)
class PrimitiveNode:
    gene: PrimitiveGene

    def to_shape(self) -> Shape:
        return self.gene.to_shape()


OperatorKind = Literal["union", "intersection", "difference"]


@dataclass(frozen=True)
class OpNode:
    kind: OperatorKind
    children: List["CompositionNode"]

    def to_shape(self) -> Shape:
        if not self.children:
            raise ValueError("OpNode has no children")
        if self.kind == "union":
            return UnionN(*[c.to_shape() for c in self.children])
        if self.kind == "intersection":
            return IntersectionN(*[c.to_shape() for c in self.children])
        # difference: fold left
        if len(self.children) < 2:
            # Degenerate: just return single child
            return self.children[0].to_shape()
        cur = self.children[0].to_shape()
        for nxt in self.children[1:]:
            cur = Difference(cur, nxt.to_shape())
        return cur


CompositionNode = Union[PrimitiveNode, OpNode]


@dataclass(frozen=True)
class CompositionGenome:
    root: CompositionNode

    def to_shape(self) -> Shape:
        return self.root.to_shape()


def _rand_color(rng: np.random.Generator) -> np.ndarray:
    return np.clip(rng.uniform(0.0, 1.0, size=3), 0.0, 1.0)


def _rand_transform(rng: np.random.Generator,
                    scale_range: tuple[float, float],
                    translate_range: tuple[float, float]) -> TransformParams:
    smin, smax = scale_range
    tmin, tmax = translate_range
    sx = rng.uniform(smin, smax)
    sy = rng.uniform(smin, smax)
    theta = rng.uniform(-math.pi, math.pi)
    dx = rng.uniform(tmin, tmax)
    dy = rng.uniform(tmin, tmax)
    return TransformParams(sx=sx, sy=sy, theta=theta, dx=dx, dy=dy)


def _rand_polygon_vertices(rng: np.random.Generator,
                           num_vertices: int,
                           radius_range: tuple[float, float] = (0.4, 1.0)) -> np.ndarray:
    # Generate a simple star-like polygon around origin by random angles sorted
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=num_vertices))
    rmin, rmax = radius_range
    radii = rng.uniform(rmin, rmax, size=num_vertices)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    verts = np.stack([xs, ys], axis=1)
    return verts


def random_gene(rng: np.random.Generator,
                kind_probs: dict[PrimitiveKind, float] | None = None,
                scale_range: tuple[float, float] = (0.2, 1.4),
                translate_range: tuple[float, float] = (-1.6, 1.6),
                polygon_vertices_range: tuple[int, int] = (3, 7)) -> PrimitiveGene:
    if kind_probs is None:
        kind_probs = {"disk": 0.4, "square": 0.4, "polygon": 0.2}
    kinds = list(kind_probs.keys())
    probs = np.array([kind_probs[k] for k in kinds], dtype=float)
    probs = probs / probs.sum()
    kind = rng.choice(kinds, p=probs).item()
    transform = _rand_transform(rng, scale_range, translate_range)
    color_rgb = _rand_color(rng)
    if kind == "polygon":
        nmin, nmax = polygon_vertices_range
        n = int(rng.integers(nmin, nmax + 1))
        verts = _rand_polygon_vertices(rng, n)
        return PrimitiveGene(kind="polygon", transform=transform, color_rgb=color_rgb, polygon_vertices=verts)
    if kind == "disk":
        return PrimitiveGene(kind="disk", transform=transform, color_rgb=color_rgb)
    return PrimitiveGene(kind="square", transform=transform, color_rgb=color_rgb)


def random_genome(rng: np.random.Generator,
                  num_genes: int,
                  **kwargs) -> CompositionGenome:
    # Create initial primitive leaves
    leaves: List[CompositionNode] = [PrimitiveNode(random_gene(rng, **kwargs)) for _ in range(max(1, num_genes))]
    # Randomly combine into a tree
    nodes: List[CompositionNode] = leaves
    # While more than one node, merge two or more nodes with a random operator
    while len(nodes) > 1:
        # pick operator
        op_kind: OperatorKind = rng.choice(["union", "intersection", "difference"]).item()
        if op_kind == "difference":
            # pick two indices
            idxs = list(rng.choice(len(nodes), size=2, replace=False))
        else:
            k = int(rng.integers(2, min(4, len(nodes)) + 1))
            idxs = list(rng.choice(len(nodes), size=k, replace=False))
        idxs.sort(reverse=True)
        picked = [nodes[i] for i in idxs]
        # remove picked from nodes
        for i in idxs:
            nodes.pop(i)
        nodes.append(OpNode(kind=op_kind, children=picked))
    return CompositionGenome(root=nodes[0])


@dataclass(frozen=True)
class MutationConfig:
    gene_mutation_prob: float = 0.6
    translate_sigma: float = 0.15
    rotate_sigma: float = 0.25
    scale_sigma: float = 0.15
    color_sigma: float = 0.1
    translate_range: tuple[float, float] = (-1.8, 1.8)
    scale_range: tuple[float, float] = (0.1, 2.0)
    add_primitive_prob: float = 0.2
    remove_primitive_prob: float = 0.1
    operator_mutation_prob: float = 0.15


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _walk_nodes(node: CompositionNode, path: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], CompositionNode]]:
    out = [(path, node)]
    if isinstance(node, OpNode):
        for i, ch in enumerate(node.children):
            out.extend(_walk_nodes(ch, path + (i,)))
    return out


def _get_by_path(node: CompositionNode, path: Tuple[int, ...]) -> CompositionNode:
    cur = node
    for idx in path:
        assert isinstance(cur, OpNode)
        cur = cur.children[idx]
    return cur


def _replace_path(node: CompositionNode, path: Tuple[int, ...], new: CompositionNode) -> CompositionNode:
    if not path:
        return new
    assert isinstance(node, OpNode)
    idx = path[0]
    new_children = list(node.children)
    new_children[idx] = _replace_path(node.children[idx], path[1:], new)
    return OpNode(kind=node.kind, children=new_children)


def _mutate_primitive(rng: np.random.Generator, g: PrimitiveGene, cfg: MutationConfig) -> PrimitiveGene:
    if rng.random() > cfg.gene_mutation_prob:
        return g
    t = g.transform
    dx = t.dx + rng.normal(0.0, cfg.translate_sigma)
    dy = t.dy + rng.normal(0.0, cfg.translate_sigma)
    dx = _clip(dx, cfg.translate_range[0], cfg.translate_range[1])
    dy = _clip(dy, cfg.translate_range[0], cfg.translate_range[1])
    theta = t.theta + rng.normal(0.0, cfg.rotate_sigma)
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    sx = _clip(t.sx + rng.normal(0.0, cfg.scale_sigma), cfg.scale_range[0], cfg.scale_range[1])
    sy = _clip(t.sy + rng.normal(0.0, cfg.scale_sigma), cfg.scale_range[0], cfg.scale_range[1])
    t_new = TransformParams(sx=sx, sy=sy, theta=theta, dx=dx, dy=dy)
    c = np.clip(g.color_rgb + rng.normal(0.0, cfg.color_sigma, size=3), 0.0, 1.0)
    return replace(g, transform=t_new, color_rgb=c)


def mutate_genome(rng: np.random.Generator,
                  genome: CompositionGenome,
                  cfg: MutationConfig) -> CompositionGenome:
    root = genome.root
    nodes = _walk_nodes(root)
    # Possibly mutate an operator kind
    if any(isinstance(n, OpNode) for _, n in nodes) and rng.random() < cfg.operator_mutation_prob:
        op_paths = [p for p, n in nodes if isinstance(n, OpNode)]
        path = op_paths[int(rng.integers(0, len(op_paths)))]
        op: OpNode = _get_by_path(root, path)  # type: ignore
        kinds = ["union", "intersection", "difference"]
        kinds.remove(op.kind)
        new_kind: OperatorKind = rng.choice(kinds).item()
        # If switching to difference, keep first two children
        new_children = op.children if new_kind != "difference" else op.children[:2]
        root = _replace_path(root, path, OpNode(kind=new_kind, children=new_children))
        nodes = _walk_nodes(root)
    # Possibly add a primitive
    if rng.random() < cfg.add_primitive_prob:
        new_leaf = PrimitiveNode(random_gene(rng))
        # If there is an OpNode with union/intersection, append to it; else wrap root with a union
        op_paths = [p for p, n in nodes if isinstance(n, OpNode) and n.kind in ("union", "intersection")]
        if op_paths:
            # pick random eligible op
            # resolve path again
            pick = op_paths[int(rng.integers(0, len(op_paths)))]
            op: OpNode = _get_by_path(root, pick)  # type: ignore
            new_children = list(op.children) + [new_leaf]
            root = _replace_path(root, pick, OpNode(kind=op.kind, children=new_children))
        else:
            root = OpNode(kind="union", children=[root, new_leaf])
        nodes = _walk_nodes(root)
    # Possibly remove a primitive (but keep at least one)
    if rng.random() < cfg.remove_primitive_prob:
        leaf_paths = [p for p, n in nodes if isinstance(n, PrimitiveNode)]
        if len(leaf_paths) > 1:
            pick = leaf_paths[int(rng.integers(0, len(leaf_paths)))]
            # Replace parent structure accordingly
            # Find parent path
            parent_path = pick[:-1]
            if not parent_path:
                # Removing root primitive: replace with a new random primitive to keep non-empty
                root = PrimitiveNode(random_gene(rng))
            else:
                parent: OpNode = _get_by_path(root, parent_path)  # type: ignore
                idx = pick[-1]
                new_children = list(parent.children)
                new_children.pop(idx)
                if parent.kind in ("union", "intersection"):
                    if len(new_children) == 1:
                        # collapse parent
                        root = _replace_path(root, parent_path, new_children[0])
                    else:
                        root = _replace_path(root, parent_path, OpNode(kind=parent.kind, children=new_children))
                else:
                    # difference: A - B (children[0] - children[1] - ...)
                    if not new_children:
                        # replace with random primitive
                        root = _replace_path(root, parent_path, PrimitiveNode(random_gene(rng)))
                    elif len(new_children) == 1:
                        root = _replace_path(root, parent_path, new_children[0])
                    else:
                        root = _replace_path(root, parent_path, OpNode(kind="difference", children=new_children[:2]))
        nodes = _walk_nodes(root)
    # Mutate primitive parameters
    def _mutate_node(n: CompositionNode) -> CompositionNode:
        if isinstance(n, PrimitiveNode):
            return PrimitiveNode(_mutate_primitive(rng, n.gene, cfg))
        if isinstance(n, OpNode):
            return OpNode(kind=n.kind, children=[_mutate_node(ch) for ch in n.children])
        return n
    root = _mutate_node(root)
    return CompositionGenome(root=root)


def breed_uniform(rng: np.random.Generator,
                  parent_a: CompositionGenome,
                  parent_b: CompositionGenome) -> CompositionGenome:
    # Subtree crossover: pick random subtree from each and swap
    nodes_a = _walk_nodes(parent_a.root)
    nodes_b = _walk_nodes(parent_b.root)
    path_a = nodes_a[int(rng.integers(0, len(nodes_a)))][0]
    path_b = nodes_b[int(rng.integers(0, len(nodes_b)))][0]
    sub_a = _get_by_path(parent_a.root, path_a)
    sub_b = _get_by_path(parent_b.root, path_b)
    child_root = _replace_path(parent_a.root, path_a, sub_b)
    return CompositionGenome(root=child_root)

