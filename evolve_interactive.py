from __future__ import annotations

import argparse
import os
import json
import numpy as np
from typing import Optional

from evolution import (
    GAConfig,
    initialize_population,
    ask_user_likes,
    evolve_one_generation,
)
from evolution.genome import PrimitiveNode, OpNode, PrimitiveGene, TransformParams, CompositionGenome
from plotting.renderer import render_to_file, render_population_grid

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive evolution of geometric compositions.")
    p.add_argument("--pop", type=int, default=16, help="population size")
    p.add_argument("--genes", type=int, default=4, help="number of primitives per composition")
    p.add_argument("--gens", type=int, default=20, help="number of generations to run")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--outdir", type=str, default="plots/evolve", help="output directory for grid images")
    p.add_argument("--cols", type=int, default=4, help="columns in the grid")
    p.add_argument("--tag", type=str, default="", help="experiment tag (subfolder) to organize outputs")
    p.add_argument("--save", type=int, nargs="*", default=[], help="indices to save in high quality each generation (e.g., --save 1 3)")
    p.add_argument("--hq_format", type=str, default="svg", choices=["png", "svg"], help="file format for high-quality saves")
    p.add_argument("--save-res", type=int, default=1200, help="resolution for high-quality individual renders")
    p.add_argument("--save-dpi", type=int, default=300, help="DPI for high-quality individual renders")
    p.add_argument("--checkpoint", type=str, default="", help="path to a *_params.json file to initialize population from")
    return p.parse_args()


def _gene_to_dict(g) -> dict:
    d = {
        "kind": g.kind,
        "transform": {
            "sx": float(g.transform.sx),
            "sy": float(g.transform.sy),
            "theta": float(g.transform.theta),
            "dx": float(g.transform.dx),
            "dy": float(g.transform.dy),
        },
        "color_rgb": [float(x) for x in (g.color_rgb.tolist() if hasattr(g.color_rgb, "tolist") else g.color_rgb)],
        "polygon_vertices": None,
    }
    if g.polygon_vertices is not None:
        d["polygon_vertices"] = [[float(a), float(b)] for a, b in g.polygon_vertices.tolist()]
    return d


def _node_to_dict(node):
    if isinstance(node, PrimitiveNode):
        return {"primitive": _gene_to_dict(node.gene)}
    if isinstance(node, OpNode):
        return {
            "op": node.kind,
            "children": [_node_to_dict(ch) for ch in node.children],
        }
    return {"unknown": True}


def _population_to_dict(population) -> dict:
    return {
        "population": [
            {
                "index": i,
                "composition": _node_to_dict(genome.root),
            }
            for i, genome in enumerate(population)
        ]
    }

def _dict_to_node(d) -> PrimitiveNode | OpNode:
    if "primitive" in d:
        g = d["primitive"]
        t = g["transform"]
        gene = PrimitiveGene(
            kind=g["kind"],
            transform=TransformParams(
                sx=float(t["sx"]),
                sy=float(t["sy"]),
                theta=float(t["theta"]),
                dx=float(t["dx"]),
                dy=float(t["dy"]),
            ),
            color_rgb=np.array(g.get("color_rgb", [0.6, 0.6, 0.6]), dtype=float),
            polygon_vertices=None if g.get("polygon_vertices") is None else np.array(g["polygon_vertices"], dtype=float),
        )
        return PrimitiveNode(gene=gene)
    if "op" in d and "children" in d:
        return OpNode(kind=str(d["op"]), children=[_dict_to_node(c) for c in d["children"]])
    raise ValueError("Invalid composition node in checkpoint JSON")


def main() -> None:
    args = parse_args()
    outdir = args.outdir if not args.tag else os.path.join(args.outdir, args.tag)
    os.makedirs(outdir, exist_ok=True)
    # Initialize from checkpoint if provided, otherwise random population
    if args.checkpoint:
        with open(args.checkpoint, "r", encoding="utf-8") as f:
            meta = json.load(f)
        pop_items = meta.get("data", {}).get("population", [])
        population = []
        for item in pop_items:
            comp = item.get("composition")
            if comp is None:
                continue
            node = _dict_to_node(comp)
            population.append(CompositionGenome(root=node))
        if not population:
            raise ValueError(f"Checkpoint {args.checkpoint} contained no valid population")
        cfg = GAConfig(population_size=len(population), num_genes=args.genes, random_seed=args.seed)
        rng = np.random.default_rng(cfg.random_seed)
        print(f"Loaded checkpoint with {len(population)} individuals from {args.checkpoint}")
    else:
        cfg = GAConfig(population_size=args.pop, num_genes=args.genes, random_seed=args.seed)
        rng, population = initialize_population(cfg)
    for g in range(args.gens):
        out_path = os.path.join(outdir, f"gen_{g:03d}.png")
        params_path = os.path.join(outdir, f"gen_{g:03d}_params.json")
        print(f"Rendering generation {g} grid -> {out_path}")
        render_population_grid(population, out_path=out_path, cols=args.cols)
        # Save parameters for later analysis
        meta = {
            "tag": args.tag,
            "generation": g,
            "config": {
                "population_size": cfg.population_size,
                "num_genes": cfg.num_genes,
                "random_seed": cfg.random_seed,
            },
            "data": _population_to_dict(population),
        }
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Open the image and pick your favorites: {out_path}")
        likes, saves = ask_user_likes(len(population))
        print(f"Selected: {likes if likes else 'none'}")
        # Union of CLI --save and on-demand saves for this generation
        to_save = sorted(set(i for i in (list(saves) + list(args.save)) if 0 <= i < len(population)))
        for idx in to_save:
            genome = population[idx]
            hq_path = os.path.join(outdir, f"gen_{g:03d}_idx_{idx:02d}_hq.{args.hq_format}")
            print(f"  Saving HQ individual #{idx} -> {hq_path}")
            render_to_file(
                genome=genome,
                out_path=hq_path,
                resolution=args.save_res,
                dpi=args.save_dpi,
                title=None,
                draw_edges=False,
                show_axes=True,
                show_grid=False,
                frame_only=True,
                format=args.hq_format,
            )
        population = evolve_one_generation(rng, population, likes, cfg)
    # Final render
    out_path = os.path.join(outdir, f"gen_{args.gens:03d}_final.png")
    params_path = os.path.join(outdir, f"gen_{args.gens:03d}_final_params.json")
    print(f"Rendering final generation grid -> {out_path}")
    render_population_grid(population, out_path=out_path, cols=args.cols)
    # Save final parameters
    final_meta = {
        "tag": args.tag,
        "generation": args.gens,
        "final": True,
        "config": {
            "population_size": cfg.population_size,
            "num_genes": cfg.num_genes,
            "random_seed": cfg.random_seed,
        },
        "data": _population_to_dict(population),
    }
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()


