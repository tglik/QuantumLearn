HUBO Benchmarking & Optimization Workbench 

Overview
- Encode real optimization problems into HUBO/QUBO
- Run standardized benchmarks across multiple solvers (classical + quantum-ish)
- Report advantage-shaped metrics: TTA, success probability, approximation ratio

Quickstart
1) Install (Python 3.11+):
   pip install -e .
2) Generate a sample assignment config:
   hubo gen-instances --family assignment --size 10 --repeats 3 --out experiments/configs/assignment_10.yaml
3) Run benchmark:
   hubo run-bench --config experiments/configs/assignment_10.yaml --out experiments/runs/assignment_10
4) Summarize results:
   hubo report --runs experiments/runs/assignment_10 --out experiments/runs/assignment_10/summary

Notes
- Deterministic instances and seeds enable reproducibility.
- ILP serves as reference for small instances; SA provides scalable baseline.
- Optional solvers (QAOA, annealing) import lazily if installed.
