from __future__ import annotations

import argparse
import os

from evolution import (
    GAConfig,
    initialize_population,
    evolve_one_generation,
)
from plotting.renderer import render_population_grid

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch non-interactive evolution with random selection.")
    p.add_argument("--pop", type=int, default=16, help="population size (default: 16)")
    p.add_argument("--genes", type=int, default=4, help="number of primitives per composition")
    p.add_argument(
        "--gens",
        type=int,
        nargs="*",
        default=[10, 100, 1000, 10000, 100000],
        help="list of generation counts to simulate (default: 10 100 1000 10000 100000)",
    )
    p.add_argument("--seed", type=int, default=42, help="base random seed (each run offsets this)")
    p.add_argument("--outdir", type=str, default="plots/evolve", help="output directory for result grids")
    p.add_argument("--cols", type=int, default=4, help="columns in the grid")
    p.add_argument("--tag", type=str, default="random", help="subfolder tag under outdir")
    return p.parse_args()


def run_simulation(num_generations: int, cfg: GAConfig) -> list:
    rng, population = initialize_population(cfg)
    for _ in range(num_generations):
        # Random selection each iteration: pass empty likes so parents are chosen randomly
        population = evolve_one_generation(rng, population, liked_indices=[], cfg=cfg)
    return population


def main() -> None:
    args = parse_args()
    # Ensure output directory
    outdir = args.outdir if not args.tag else os.path.join(args.outdir, args.tag)
    os.makedirs(outdir, exist_ok=True)
    # Sort and de-dup generation counts, keep as positive integers
    gen_counts = sorted(set(int(g) for g in args.gens if int(g) > 0))
    if not gen_counts:
        raise ValueError("No valid positive generation counts provided.")
    for i, gens in enumerate(gen_counts):
        # Offset seed per run for diversity while keeping reproducibility
        cfg = GAConfig(population_size=args.pop, num_genes=args.genes, random_seed=int(args.seed) + i)
        print(f"[Run {i+1}/{len(gen_counts)}] Evolving {gens} generations (pop={cfg.population_size}, genes={cfg.num_genes}, seed={cfg.random_seed})")
        population = run_simulation(gens, cfg)
        # Save final grid for this simulation (zero-pad to 6 to accommodate 100000)
        out_path = os.path.join(outdir, f"random_{gens:06d}.png")
        print(f"Rendering final population grid -> {out_path}")
        render_population_grid(population, out_path=out_path, cols=args.cols)
    print("All simulations completed.")


if __name__ == "__main__":
    main()


