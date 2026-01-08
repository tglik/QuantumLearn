from __future__ import annotations

__all__ = [
    "Solver",
    "SolveResult",
    "SolverTrace",
    "SimulatedAnnealingSolver",
    "ILPSolver",
    "get_solver",
]

from .base import Solver, SolveResult, SolverTrace
from .classical_sa import SimulatedAnnealingSolver
from .classical_ilp import ILPSolver


SOLVER_REGISTRY = {
    "sa": SimulatedAnnealingSolver,
    "ilp": ILPSolver,
}


def get_solver(name: str) -> Solver:
    """Get solver by name."""
    if name not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver: {name}. Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name]()
