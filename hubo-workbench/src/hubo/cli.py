from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import RunnerConfig, SolverConfig, ReductionConfig
from .bench.runner import run_benchmark
from .bench.reports import summarize_runs
from .bench.instances import write_instance_config

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("gen-instances")
def gen_instances(
    family: str = typer.Option("assignment", help="Instance family"),
    size: int = typer.Option(10, help="Problem size"),
    repeats: int = typer.Option(3, help="Number of repeats/seeds"),
    out: Path = typer.Option(
        Path("experiments/configs/assignment.yaml"), help="Path to save config YAML"
    ),
):
    cfg = RunnerConfig(family=family, sizes=[size], repeats=repeats)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_instance_config(out, cfg)
    typer.echo(f"Wrote config to {out}")


@app.command("run-bench")
def run_bench(
    config: Path = typer.Option(..., exists=True, help="Config YAML path"),
    out: Path = typer.Option(
        Path("experiments/runs/out"), help="Output directory for results"
    ),
):
    with open(config, "r") as f:
        data = yaml.safe_load(f)
    # Support both dict (single) and list (batch) configs
    if isinstance(data, dict):
        cfg = _runner_from_dict(data)
        run_benchmark(cfg, out)
    elif isinstance(data, list):
        for item in data:
            cfg = _runner_from_dict(item)
            run_benchmark(cfg, out)
    else:
        raise typer.BadParameter("Invalid config format")


@app.command("report")
def report(
    runs: Path = typer.Option(..., exists=True, help="Runs directory"),
    out: Path = typer.Option(
        Path("experiments/runs/summary"), help="Summary output directory"
    ),
):
    out.mkdir(parents=True, exist_ok=True)
    summarize_runs(runs, out)
    typer.echo(f"Wrote summary to {out}")


def _runner_from_dict(d: dict) -> RunnerConfig:
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
