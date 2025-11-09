from __future__ import annotations

import argparse
import os
import json
from typing import Optional

from evolution import (
    GAConfig,
    initialize_population,
    render_population_grid,
    ask_user_likes,
    evolve_one_generation,
)
from evolution.genome import PrimitiveNode, OpNode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive evolution of geometric compositions.")
    p.add_argument("--pop", type=int, default=16, help="population size")
    p.add_argument("--genes", type=int, default=4, help="number of primitives per composition")
    p.add_argument("--gens", type=int, default=20, help="number of generations to run")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--outdir", type=str, default="plots/evolve", help="output directory for grid images")
    p.add_argument("--cols", type=int, default=4, help="columns in the grid")
    p.add_argument("--tag", type=str, default="", help="experiment tag (subfolder) to organize outputs")
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


def main() -> None:
    args = parse_args()
    outdir = args.outdir if not args.tag else os.path.join(args.outdir, args.tag)
    os.makedirs(outdir, exist_ok=True)
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
        likes = ask_user_likes(len(population))
        print(f"Selected: {likes if likes else 'none'}")
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


