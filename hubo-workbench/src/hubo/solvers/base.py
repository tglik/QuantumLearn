from __future__ import annotations
from typing import Protocol, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class SolverTrace:
    """Energy trajectory over time."""

    times: List[float] = field(default_factory=list)  # wall seconds
    energies: List[float] = field(default_factory=list)


@dataclass
class SolveResult:
    """Result of a single solve run."""

    best_assignment: Dict[str, int]
    best_energy: float
    wall_time_s: float
    solver_time_s: float  # may differ from wall time (compilation, etc.)
    trace: SolverTrace = field(default_factory=SolverTrace)
    success: bool = False  # whether target threshold met
    metadata: Dict[str, Any] = field(default_factory=dict)


class Solver(Protocol):
    """Solver plugin interface."""

    name: str

    def solve(self, model: Any, *, seed: int, budget: Dict[str, float]) -> SolveResult:
        """
        Solve the given model (HUBOModel or QUBOModel).
        Budget keys: 'wall_time_s', 'iters', 'shots', etc.
        Returns SolveResult with best assignment, energy, timing, trace.
        """
        ...
