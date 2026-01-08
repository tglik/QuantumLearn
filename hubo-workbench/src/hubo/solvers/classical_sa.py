from __future__ import annotations
import time
import numpy as np
from typing import Any, Dict

from .base import Solver, SolveResult, SolverTrace


class SimulatedAnnealingSolver:
    """Classical simulated annealing solver for HUBO/QUBO."""

    name = "sa"

    def solve(self, model: Any, *, seed: int, budget: Dict[str, float]) -> SolveResult:
        """
        SA with exponential cooling schedule.
        Budget keys: 'iters' (default 10000), 'wall_time_s' (default 5.0)
        """
        rng = np.random.default_rng(seed)
        variables = model.variables
        n = len(variables)

        # Initial random assignment
        assignment = {v: rng.integers(0, 2) for v in variables}
        current_energy = model.energy(assignment)
        best_assignment = assignment.copy()
        best_energy = current_energy

        max_iters = int(budget.get("iters", 10000))
        max_time = budget.get("wall_time_s", 5.0)

        # Temperature schedule: T = T0 * alpha^iteration
        T0 = abs(current_energy) * 2.0 if current_energy != 0 else 100.0
        alpha = 0.99

        start_time = time.perf_counter()
        trace_times = []
        trace_energies = []

        for iteration in range(max_iters):
            elapsed = time.perf_counter() - start_time
            if elapsed > max_time:
                break

            # Temperature
            T = T0 * (alpha**iteration)
            if T < 1e-9:
                T = 1e-9

            # Flip random variable
            var = variables[rng.integers(0, n)]
            old_val = assignment[var]
            assignment[var] = 1 - old_val
            new_energy = model.energy(assignment)
            delta = new_energy - current_energy

            # Metropolis acceptance
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment.copy()
            else:
                # Reject
                assignment[var] = old_val

            # Record trace every 100 iters
            if iteration % 100 == 0:
                trace_times.append(elapsed)
                trace_energies.append(best_energy)

        end_time = time.perf_counter()
        wall_time = end_time - start_time

        # Final trace point
        trace_times.append(wall_time)
        trace_energies.append(best_energy)

        return SolveResult(
            best_assignment=best_assignment,
            best_energy=best_energy,
            wall_time_s=wall_time,
            solver_time_s=wall_time,
            trace=SolverTrace(times=trace_times, energies=trace_energies),
            metadata={"iterations": iteration + 1, "T0": T0, "alpha": alpha},
        )
