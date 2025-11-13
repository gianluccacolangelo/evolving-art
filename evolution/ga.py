from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np

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



def ask_user_likes(n_items: int) -> Tuple[List[int], List[int]]:
    print(f"Enter indices you like (0 to {n_items - 1}), optionally add --save i j to save HQ:")
    s = input("> ").strip()
    if not s:
        return [], []
    likes: List[int] = []
    saves: List[int] = []
    save_mode = False
    for tok in s.replace(",", " ").split():
        if tok.lower() in ("--save", "save", "--save:"):
            save_mode = True
            continue
        try:
            v = int(tok)
            if 0 <= v < n_items:
                (saves if save_mode else likes).append(v)
        except ValueError:
            pass
    likes = sorted(set(likes))
    saves = sorted(set(saves))
    return likes, saves


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
    
    for p in parents:
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


