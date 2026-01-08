from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict


def compute_tta(
    times: List[float], energies: List[float], target_energy: float
) -> float | None:
    """
    Compute time-to-approximate-solution: first time reaching target_energy.
    Returns None if never reached.
    """
    for t, e in zip(times, energies):
        if e <= target_energy:
            return t
    return None


def compute_success_probability(energies: List[float], target_energy: float) -> float:
    """Fraction of runs reaching target."""
    successes = sum(1 for e in energies if e <= target_energy)
    return successes / len(energies) if energies else 0.0


def compute_approximation_ratio(energy: float, best_known: float) -> float:
    """Approximation ratio: energy / best_known (lower is better)."""
    if best_known == 0:
        return 1.0 if energy == 0 else float("inf")
    return energy / best_known


def summarize_runs(runs_dir: Path, output_dir: Path):
    """Generate summary CSV from JSONL results."""
    results_file = runs_dir / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    records = []
    with open(results_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Compute best_known per instance (size, repeat)
    best_known = (
        df.groupby(["family", "size", "repeat"])["best_energy"].min().reset_index()
    )
    best_known.rename(columns={"best_energy": "best_known_energy"}, inplace=True)
    df = df.merge(best_known, on=["family", "size", "repeat"])

    # Approximation ratio
    df["approx_ratio"] = df.apply(
        lambda row: compute_approximation_ratio(
            row["best_energy"], row["best_known_energy"]
        ),
        axis=1,
    )

    # Per-solver summary
    summary = (
        df.groupby(["family", "size", "solver"])
        .agg(
            {
                "best_energy": ["mean", "median", "std"],
                "wall_time_s": ["mean", "median"],
                "approx_ratio": ["mean", "median"],
                "success": "mean",
            }
        )
        .reset_index()
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Write metadata
    with open(output_dir / "summary_metadata.json", "w") as f:
        json.dump(
            {
                "total_runs": len(df),
                "families": df["family"].unique().tolist(),
                "sizes": df["size"].unique().tolist(),
                "solvers": df["solver"].unique().tolist(),
            },
            f,
            indent=2,
        )


def generate_summary_report(runs_dir: str | Path, output_dir: str | Path):
    """Wrapper for summarize_runs."""
    summarize_runs(Path(runs_dir), Path(output_dir))
