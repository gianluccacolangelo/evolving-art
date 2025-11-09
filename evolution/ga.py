from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

from shapes import Shape
from plotting.renderer import render_to_axes
from .genome import (
    CompositionGenome,
    MutationConfig,
    random_genome,
    mutate_genome,
    breed_uniform,
)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 16
    num_genes: int = 4
    random_seed: int = 42
    mutation: MutationConfig = MutationConfig()


def initialize_population(cfg: GAConfig) -> Tuple[np.random.Generator, List[CompositionGenome]]:
    rng = np.random.default_rng(cfg.random_seed)
    pop = [random_genome(rng, cfg.num_genes) for _ in range(cfg.population_size)]
    return rng, pop


def render_population_grid(population: Sequence[CompositionGenome],
                           out_path: str,
                           cols: int = 4,
                           resolution: int = 100,
                           figsize_per_cell: Tuple[float, float] = (3.0, 3.0),
                           draw_edges: bool = False) -> None:
    n = len(population)
    cols = max(1, cols)
    rows = (n + cols - 1) // cols
    fig_w = figsize_per_cell[0] * cols
    fig_h = figsize_per_cell[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.expand_dims(axes, axis=1)
    for idx, genome in enumerate(population):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        try:
            shape: Shape = genome.to_shape()
            render_to_axes(
                ax,
                shape,
                resolution=resolution,
                title=f"#{idx}",
                draw_edges=draw_edges,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Error\n{e}", ha="center", va="center")
            ax.axis("off")
    # Hide any extra axes
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def ask_user_likes(n_items: int) -> List[int]:
    print(f"Enter indices you like (0 to {n_items - 1}), separated by spaces (or leave empty):")
    s = input("> ").strip()
    if not s:
        return []
    picks: List[int] = []
    for tok in s.replace(",", " ").split():
        try:
            v = int(tok)
            if 0 <= v < n_items:
                picks.append(v)
        except ValueError:
            pass
    return sorted(set(picks))


def evolve_one_generation(rng: np.random.Generator,
                          population: Sequence[CompositionGenome],
                          liked_indices: Sequence[int],
                          cfg: GAConfig) -> List[CompositionGenome]:
    n = len(population)
    if n == 0:
        return []
    if liked_indices:
        parents = [population[i] for i in liked_indices]
    else:
        # If nothing is liked, softly keep a random subset as "parents"
        k = max(1, n // 4)
        parents = list(rng.choice(population, size=k, replace=False))
    # Elitism: carry over parents (trim to avoid explosion)
    next_pop: List[CompositionGenome] = []
    for p in parents[: min(len(parents), max(1, n // 4))]:
        next_pop.append(p)
    # Breed to fill the rest
    while len(next_pop) < n:
        if len(parents) == 1:
            child = mutate_genome(rng, parents[0], cfg.mutation)
        else:
            pa, pb = rng.choice(parents, size=2, replace=True)
            child = breed_uniform(rng, pa, pb)
            child = mutate_genome(rng, child, cfg.mutation)
        next_pop.append(child)
    return next_pop


