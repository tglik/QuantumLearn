Methodology & Reproducibility

Core Principles

1. **Deterministic instances**: All problem instances are seeded and hashed to ensure consistency.
2. **Fixed budgets**: Wall time and iteration limits are specified per solver in config YAML.
3. **Standardized encodings**: Each problem encodes to HUBO, then optionally reduces to QUBO via Rosenberg.
4. **Reproducible seeds**: Each repeat uses a fixed seed (either provided or derived from repeat index).

Target Definitions

- **Energy threshold**: Target energy based on best-known (ILP optimum or best heuristic).
- **Approximation ratio**: `energy / best_known` (lower is better; 1.0 = optimal).
- **Success criterion**: Whether the solver reached the target within budget.

Fairness

- Same instance encoding for all solvers (HUBO or reduced QUBO).
- Comparable stopping conditions (time budgets may differ for classical vs quantum solvers).
- Deterministic benchmarks with repeats (N ≥ 3) to capture variance.

Metrics Computed

- **Best energy**: Lowest energy found across repeats.
- **Success probability**: Fraction of repeats reaching target within budget.
- **TTA (time-to-approximate-solution)**: Expected time to reach target energy with p=0.9 success (if enough repeats).
- **Approximation ratio**: Quality vs best-known solution.

Experiment Tracking

- Config YAML saved with each run.
- JSONL traces record energy vs time for post-hoc analysis.
- Git commit hash and Python environment info (optional future feature).
