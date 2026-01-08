Metrics Definitions

Time-to-Approximate-Solution (TTA)

**Definition**: Wall-clock time to first reach a target energy threshold (or approximation ratio).

**Use case**: Measuring runtime advantage. If Solver A reaches target in 0.5s and Solver B in 5s, A is 10x faster for this instance.

**Computation**: For each repeat, find the first time `t` where `energy(t) ≤ target_energy`. Report median TTA across repeats, or p90 (time by which 90% of repeats succeed).

Success Probability

**Definition**: Fraction of repeats that reach the target energy within the budget.

**Use case**: Reliability metric. A solver with 90% success is more reliable than one with 50%.

**Computation**: `success_prob = (# successful repeats) / (total repeats)`

Approximation Ratio

**Definition**: `energy / best_known_energy` where `best_known` is the best energy found across all solvers (or ILP optimum).

**Use case**: Solution quality. Ratio of 1.0 = optimal; 1.05 = 5% from optimal.

**Computation**: Per instance, compute best_known across solvers (or use ILP as reference). Then for each solver: `approx_ratio = final_energy / best_known`.

Resource Proxies (optional, future)

- **Circuit depth**: For QAOA/quantum solvers.
- **2Q gate count**: Hardware connectivity / noise sensitivity.
- **Shots**: Number of circuit evaluations (quantum sampling).
- **QPU wall time**: Actual hardware execution time.
- **Embedding overhead**: For annealing solvers (graph minor embedding).

Notes

- **TTA curves**: Plot time vs energy for all solvers; show when each crosses target threshold.
- **Success@budget scatter**: Plot success probability vs time budget to see tradeoffs.
- **Approximation ratio box plots**: Show distribution of solution quality across instances.
