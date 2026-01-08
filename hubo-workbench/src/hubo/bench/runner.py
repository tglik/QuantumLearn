from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from hubo.config import RunnerConfig, SolverConfig
from hubo.problems import AssignmentProblem
from hubo.solvers import get_solver
from hubo.encoding import rosenberg_reduction


def run_benchmark(cfg: RunnerConfig, output_dir: Path):
    """Run benchmark for given config, write JSONL results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(
            {
                "family": cfg.family,
                "sizes": cfg.sizes,
                "repeats": cfg.repeats,
                "solvers": [{"name": s.name, "budget": s.budget} for s in cfg.solvers],
                "reduction": cfg.reduction.__dict__,
                "target": cfg.target,
            },
            f,
        )

    with open(results_file, "w") as f:
        for size in cfg.sizes:
            for repeat in range(cfg.repeats):
                seed = cfg.seeds[repeat] if cfg.seeds else (1000 + repeat)
                problem = _generate_problem(cfg.family, size, seed)

                # Encode
                hubo = problem.encode_hubo()
                if (
                    cfg.reduction.enable_reduction
                    and hubo.degree() > cfg.reduction.max_degree
                ):
                    result = rosenberg_reduction(
                        hubo, penalty_scale=cfg.reduction.penalty_scale
                    )
                    model = result.qubo
                    model_type = "qubo"
                    reduction_metadata = {
                        "ancilla_count": result.ancilla_count,
                        "penalty_strength": result.penalty_strength,
                        "overhead_factor": result.overhead_factor,
                    }
                else:
                    model = hubo
                    model_type = "hubo"
                    reduction_metadata = {}

                # Solve with each solver
                for solver_cfg in cfg.solvers:
                    solver = get_solver(solver_cfg.name)
                    solve_result = solver.solve(
                        model, seed=seed, budget=solver_cfg.budget
                    )

                    # Check success vs target
                    success = False
                    if cfg.target and "approx_ratio" in cfg.target:
                        # Assume we have best_known from ILP or heuristic (for now, just check energy)
                        success = solve_result.best_energy < float("inf")

                    record = {
                        "family": cfg.family,
                        "size": size,
                        "repeat": repeat,
                        "seed": seed,
                        "solver": solver_cfg.name,
                        "model_type": model_type,
                        "best_energy": solve_result.best_energy,
                        "wall_time_s": solve_result.wall_time_s,
                        "solver_time_s": solve_result.solver_time_s,
                        "success": success,
                        "trace_times": solve_result.trace.times,
                        "trace_energies": solve_result.trace.energies,
                        "reduction": reduction_metadata,
                        "solver_metadata": solve_result.metadata,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()


def run_benchmark_from_config(config_path: str | Path, output_dir: str | Path):
    """Load config YAML and run benchmark."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    cfg = _runner_from_dict(data)
    run_benchmark(cfg, Path(output_dir))


def _runner_from_dict(d: dict) -> RunnerConfig:
    from hubo.config import ReductionConfig

    solvers = [
        SolverConfig(**s) if isinstance(s, dict) else s for s in d.get("solvers", [])
    ]
    reduction = d.get("reduction", {})
    cfg = RunnerConfig(
        family=d.get("family", "assignment"),
        sizes=d.get("sizes", [10]),
        repeats=d.get("repeats", 3),
        seeds=d.get("seeds"),
        solvers=solvers
        or [
            SolverConfig(name="sa", budget={"iters": 20000, "wall_time_s": 5.0}),
            SolverConfig(name="ilp", budget={"wall_time_s": 30.0}),
        ],
        reduction=(
            ReductionConfig(**reduction) if isinstance(reduction, dict) else reduction
        ),
        target=d.get("target"),
        notes=d.get("notes", ""),
    )
    return cfg


def _generate_problem(family: str, size: int, seed: int):
    if family == "assignment":
        return AssignmentProblem.random(n=size, seed=seed)
    else:
        raise ValueError(f"Unknown family: {family}")
