from __future__ import annotations
import time
from typing import Any, Dict

from .base import Solver, SolveResult, SolverTrace

try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


class ILPSolver:
    """Integer linear programming solver using PuLP."""

    name = "ilp"

    def solve(self, model: Any, *, seed: int, budget: Dict[str, float]) -> SolveResult:
        """
        Solve HUBO/QUBO as ILP.
        Budget keys: 'wall_time_s' (default 30.0)
        NOTE: PuLP does not support quadratic objectives; skips quadratic models.
        """
        if not PULP_AVAILABLE:
            raise ImportError(
                "PuLP not installed; cannot use ILP solver. Install with: pip install pulp"
            )

        # Check if model has quadratic terms
        has_quadratic = False
        if hasattr(model, "terms"):
            # HUBO: check for any term with degree > 1
            has_quadratic = any(len(term.vars) > 1 for term in model.terms)
        else:
            # QUBO: check for any Q terms
            has_quadratic = bool(model.Q)

        if has_quadratic:
            # Skip quadratic models (PuLP ILP is linear only)
            return SolveResult(
                best_assignment={v: 0 for v in model.variables},
                best_energy=float("inf"),
                wall_time_s=0.0,
                solver_time_s=0.0,
                trace=SolverTrace(times=[0.0], energies=[float("inf")]),
                success=False,
                metadata={
                    "status": "SKIPPED",
                    "reason": "PuLP ILP does not support quadratic terms",
                },
            )

        start_time = time.perf_counter()
        max_time = budget.get("wall_time_s", 30.0)

        # Create LP problem (minimize)
        prob = pulp.LpProblem("HUBO_QUBO_ILP", pulp.LpMinimize)

        # Variables
        lp_vars = {v: pulp.LpVariable(v, cat="Binary") for v in model.variables}

        # Build objective (linear only)
        objective = model.constant
        if hasattr(model, "terms"):
            # HUBO (linear only)
            for term in model.terms:
                if len(term.vars) == 1:
                    objective += term.coeff * lp_vars[term.vars[0]]
        else:
            # QUBO (linear only)
            for var, coeff in model.linear.items():
                objective += coeff * lp_vars[var]

        prob += objective

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=max_time, msg=False)
        status = prob.solve(solver)

        end_time = time.perf_counter()
        wall_time = end_time - start_time

        if status == pulp.LpStatusOptimal or status == pulp.LpStatusNotSolved:
            # Extract assignment
            assignment = {
                v: int(lp_vars[v].varValue) if lp_vars[v].varValue is not None else 0
                for v in model.variables
            }
            best_energy = model.energy(assignment)
            success = status == pulp.LpStatusOptimal
        else:
            # Infeasible or error
            assignment = {v: 0 for v in model.variables}
            best_energy = float("inf")
            success = False

        return SolveResult(
            best_assignment=assignment,
            best_energy=best_energy,
            wall_time_s=wall_time,
            solver_time_s=wall_time,
            trace=SolverTrace(times=[wall_time], energies=[best_energy]),
            success=success,
            metadata={"status": pulp.LpStatus[status], "optimal": success},
        )
