from __future__ import annotations
import numpy as np
from typing import List


def compute_tta(
    times: List[float], energies: List[float], target_energy: float
) -> float | None:
    """Time-to-approximate-solution (TTA): first time reaching target."""
    for t, e in zip(times, energies):
        if e <= target_energy:
            return t
    return None


def compute_success_probability(
    final_energies: List[float], target_energy: float
) -> float:
    """Success probability: fraction of runs reaching target."""
    if not final_energies:
        return 0.0
    return sum(1 for e in final_energies if e <= target_energy) / len(final_energies)


def compute_approximation_ratio(energy: float, best_known: float) -> float:
    """Approximation ratio: energy / best_known."""
    if best_known == 0:
        return 1.0 if energy == 0 else float("inf")
    return energy / best_known
