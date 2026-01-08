from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

import yaml

from hubo.config import RunnerConfig


def generate_config_yaml(
    family: str,
    sizes: list[int],
    repeats: int,
    output_path: str | Path,
    solvers: list[str] | None = None,
):
    """Generate a YAML config for instance generation."""
    if solvers is None:
        solvers = ["sa", "ilp"]

    cfg = {
        "family": family,
        "sizes": sizes,
        "repeats": repeats,
        "solvers": [
            {"name": s, "budget": {"iters": 20000, "wall_time_s": 5.0}} for s in solvers
        ],
        "reduction": {"enable_reduction": True, "penalty_scale": 10.0, "max_degree": 2},
        "target": {"approx_ratio": 1.05},
        "notes": f"Auto-generated config for {family} problem family",
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def write_instance_config(output_path: Path, cfg: RunnerConfig):
    """Write RunnerConfig to YAML."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "family": cfg.family,
        "sizes": cfg.sizes,
        "repeats": cfg.repeats,
        "seeds": cfg.seeds,
        "solvers": [
            {"name": s.name, "budget": s.budget, "params": s.params}
            for s in cfg.solvers
        ],
        "reduction": {
            "enable_reduction": cfg.reduction.enable_reduction,
            "penalty_scale": cfg.reduction.penalty_scale,
            "max_degree": cfg.reduction.max_degree,
        },
        "target": cfg.target,
        "notes": cfg.notes,
    }
    with open(output_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
