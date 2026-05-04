from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from optimizers.base import BaseOptimizer

if TYPE_CHECKING:
    from core.config import AppConfig
    from core.function_handler import FunctionHandler
    from core.solutions_handler import SolutionsHandler

SelectionMethod = Literal["Best", "Roulette", "Tournament"]


@dataclass
class Individual:
    x1: float
    x2: float
    f_value: float
    slot: int


class EvolutiveSearchOptimizer(BaseOptimizer):
    """Population-based optimizer using mutation, crossover, and selection."""

    def __init__(self) -> None:
        self.population_size: int = 12
        self.mutation_sigma: float = 0.5
        self.crossover_temperature: float = 0.2
        self.selection_temperature: float = 1.0
        self.selection_method: SelectionMethod = "Best"
        self.iteration: int = 0
        self.population: list[Individual] = []
        self.history: list[dict] = []

    @property
    def name(self) -> str:
        return "Evolutive search"

    @property
    def description(self) -> str:
        return "Population-based search using mutation, crossover, and selection."

    def configure(
        self,
        n: int,
        sigma: float,
        crossover_temperature: float,
        selection_temperature: float,
        method: SelectionMethod,
    ) -> None:
        self.population_size = max(2, int(n))
        self.mutation_sigma = max(1e-6, float(sigma))
        self.crossover_temperature = max(1e-6, float(crossover_temperature))
        self.selection_temperature = max(1e-6, float(selection_temperature))
        self.selection_method = method

    def initialize_population(
        self,
        function_handler: FunctionHandler,
        config: AppConfig,
    ) -> list[Individual]:
        self.population = []
        for slot in range(self.population_size):
            x1, x2 = config.sample_uniform_point()
            self.population.append(
                Individual(x1=x1, x2=x2, f_value=function_handler.evaluate(x1, x2), slot=slot)
            )
        self.iteration = 0
        self.history = [self._history_row(self.population)]
        return list(self.population)

    def preview_next_generation(
        self,
        function_handler: FunctionHandler,
        config: AppConfig,
    ) -> dict | None:
        if not self.population:
            return None
        rng = config.clone_rng()
        return self._build_generation(function_handler, config, rng)

    def advance_generation(
        self,
        function_handler: FunctionHandler,
        config: AppConfig,
    ) -> list[Individual] | None:
        prepared = self._build_generation(function_handler, config, config.rng)
        if prepared is None:
            return None

        selected: list[Individual] = prepared["selected_points"]
        new_population: list[Individual] = []
        for slot, p in enumerate(selected):
            new_population.append(Individual(x1=p.x1, x2=p.x2, f_value=p.f_value, slot=slot))
        self.population = new_population
        self.iteration += 1
        self.history.append(self._history_row(self.population))
        return list(self.population)

    def _build_generation(
        self,
        function_handler: FunctionHandler,
        config: AppConfig,
        rng: np.random.Generator,
    ) -> dict | None:
        if not self.population:
            return None

        mutate_n = max(1, self.population_size // 2)
        mut_idx = rng.choice(len(self.population), size=mutate_n, replace=False).tolist()
        mut_points = []
        for idx in mut_idx:
            p = self.population[idx]
            nx1, nx2 = config.clamp(
                p.x1 + float(rng.normal(0.0, self.mutation_sigma)),
                p.x2 + float(rng.normal(0.0, self.mutation_sigma)),
            )
            mut_points.append(
                Individual(nx1, nx2, function_handler.evaluate(nx1, nx2), slot=p.slot)
            )

        cross_pairs: list[tuple[int, int]] = []
        cross_points: list[Individual] = []
        for _ in range(self.population_size):
            i, j = rng.choice(len(self.population), size=2, replace=False).tolist()
            p1 = self.population[i]
            p2 = self.population[j]
            cross_pairs.append((i, j))
            z = float(rng.normal(0.0, self.crossover_temperature))
            w = 1.0 / (1.0 + np.exp(-z))
            nx1, nx2 = config.clamp(w * p1.x1 + (1 - w) * p2.x1, w * p1.x2 + (1 - w) * p2.x2)
            cross_points.append(
                Individual(nx1, nx2, function_handler.evaluate(nx1, nx2), slot=p1.slot)
            )

        candidates = list(self.population) + mut_points + cross_points
        selected, selection_details = self._select(candidates, self.population_size, rng)
        return {
            "mutated_indices": mut_idx,
            "mutated_points": copy.deepcopy(mut_points),
            "crossover_pairs": cross_pairs,
            "crossover_points": copy.deepcopy(cross_points),
            "selected_points": copy.deepcopy(selected),
            "candidate_points": [
                {
                    "x1": p.x1,
                    "x2": p.x2,
                    "f_value": p.f_value,
                    "slot": p.slot,
                }
                for p in candidates
            ],
            "selection": selection_details,
            "sigma": self.mutation_sigma,
        }

    def _select(
        self, candidates: list[Individual], n: int, rng: np.random.Generator
    ) -> tuple[list[Individual], dict]:
        if self.selection_method == "Best":
            sorted_indices = sorted(
                range(len(candidates)),
                key=lambda idx: candidates[idx].f_value,
                reverse=True,
            )[:n]
            return (
                [candidates[idx] for idx in sorted_indices],
                {
                    "mode": "best",
                    "selected_indices": sorted_indices,
                    "candidate_probabilities": None,
                },
            )

        if self.selection_method == "Roulette":
            f = np.array([p.f_value for p in candidates], dtype=float)
            f = f - f.min() + 1e-9
            weights = np.power(f, 1.0 / self.selection_temperature)
            probs = weights / weights.sum()
            idx = rng.choice(len(candidates), size=n, replace=not n <= len(candidates), p=probs)
            selected_indices = [int(i) for i in idx]
            return (
                [candidates[i] for i in selected_indices],
                {
                    "mode": "roulette",
                    "selected_indices": selected_indices,
                    "candidate_probabilities": probs.tolist(),
                },
            )

        picked = []
        selected_indices = []
        matchup_probabilities = []
        tournament_pairs = []
        for _ in range(n):
            i, j = rng.choice(len(candidates), size=2, replace=False).tolist()
            tournament_pairs.append((i, j))
            better_idx, worse_idx = (
                (i, j)
                if candidates[i].f_value >= candidates[j].f_value
                else (j, i)
            )
            delta = abs(candidates[better_idx].f_value - candidates[worse_idx].f_value)
            p_better = 1.0 / (1.0 + float(np.exp(-delta / self.selection_temperature)))
            chosen_idx = better_idx if rng.random() < p_better else worse_idx
            matchup_probabilities.append(p_better)
            selected_indices.append(chosen_idx)
            picked.append(candidates[chosen_idx])
        return (
            picked,
            {
                "mode": "tournament",
                "selected_indices": selected_indices,
                "candidate_probabilities": None,
                "tournament_pairs": tournament_pairs,
                "best_pick_probabilities": matchup_probabilities,
            },
        )

    def _history_row(self, population: list[Individual]) -> dict:
        values = [p.f_value for p in population]
        return {
            "iteration": self.iteration,
            "population": [(p.slot, p.x1, p.x2, p.f_value) for p in population],
            "best": max(values) if values else float("nan"),
            "mean": float(np.mean(values)) if values else float("nan"),
        }

    def suggest_next(
        self,
        function_handler: FunctionHandler,  # noqa: ARG002
        solutions_handler: SolutionsHandler,  # noqa: ARG002
        config: AppConfig,  # noqa: ARG002
    ) -> tuple[float, float] | None:
        return None

    def reset(self) -> None:
        self.population = []
        self.history = []
        self.iteration = 0
