from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ReductionConfig:
    enable_reduction: bool = True
    penalty_scale: float = 10.0
    max_degree: int = 2


@dataclass
class SolverConfig:
    name: str
    budget: Dict[str, float] = field(
        default_factory=lambda: {"wall_time_s": 5.0, "iters": 10000}
    )
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    family: str = "assignment"
    sizes: List[int] = field(default_factory=lambda: [10])
    repeats: int = 3
    seeds: Optional[List[int]] = None
    solvers: List[SolverConfig] = field(
        default_factory=lambda: [
            SolverConfig(name="sa", budget={"iters": 20000, "wall_time_s": 5.0}),
            SolverConfig(name="ilp", budget={"wall_time_s": 30.0}),
        ]
    )
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    target: Optional[Dict[str, float]] = None  # e.g., {"approx_ratio": 1.05}
    notes: str = ""
