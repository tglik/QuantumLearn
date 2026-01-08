# QML Platform Project - Context

## Project Overview
Building a **developer-first Quantum Machine Learning platform** that makes QML practical, testable, and comparable. Focus on data encoding, trainability, hardware-aware evaluation, and reproducible benchmarks with strong classical baselines.

**Vision**: Do for QML what Hugging Face + PyTorch did for classical ML — but starting from the hard parts, not toy demos.

## Current Status
**Last Updated**: 2025-01-07

### Active Phase: Research & Discovery + HUBO Workbench Development
- Surveying publicly available QML algorithms, encodings, and benchmarks
- Identifying production-ready components vs. research-stage techniques
- Cataloging platforms (Kipu, IBM Quantum, PennyLane, etc.) for reference
- Finding the right algorithms and components to start experimenting with
- **NEW**: Built HUBO/QUBO benchmarking workbench for optimization (Kipu-aligned)
  - Rosenberg reduction, SA/ILP solvers, advantage-shaped metrics (TTA, success prob)
  - Reproducible benchmark runner with JSONL traces and summary reports
  - See: `hubo-workbench/` subdirectory

## Core Mission & Thesis

### Mission
Build a **developer-first QML platform** focusing on:
- **Data encoding & data re-uploading**
- **Trainability & barren plateau avoidance**
- **Hardware-aware evaluation**
- **Reproducible benchmarks with strong classical baselines**

### Core Thesis
- QML will **not replace GPUs**; it will enable **new hybrid / quantum-native models**
- **Data encoding is the bottleneck**: bad encoding erases any quantum advantage
- Near-term value comes from:
  - Structured data
  - Shallow, trainable circuits
  - Hybrid classical–quantum workflows
- Hardware reality (noise, connectivity, shot budgets) must be **first-class citizens**

## Primary Product Directions

### A. Quantum Data Encoding Platform (CORE)
- Automated classical → quantum feature map generation
- Support for:
  - Angle, amplitude, IQP, Hamiltonian, and data re-uploading encodings
- Dataset-aware encoding selection:
  - Sparsity
  - Periodicity
  - Correlation structure
- Output circuits usable in **Qiskit / PennyLane**
- Benchmark encodings side-by-side

### B. Trainability Analyzer
- Detect barren plateau risk
- Circuit depth & entanglement analysis
- Initialization & cost-function recommendations
- Trainability score (heuristic, not proof)

### C. QML Benchmarking Layer
- Curated **QML-relevant datasets**
- Strong classical baselines required by default
- Metrics for:
  - Accuracy / AUC
  - Sample efficiency
  - Circuit depth
  - Hardware noise sensitivity
- Reproducibility & experiment tracking

## Near-Term Technical Focus
- Variational Quantum Classifiers (VQC)
- Quantum Kernels (QKE)
- Data re-uploading circuits
- Hybrid classical → quantum pipelines
- Empirical (not theoretical-only) quantum advantage demos

## Research Deliverables

### Completed
1. **kipu_urls.md** - Kipu Quantum platform reference catalog
   - 20+ use cases across finance, logistics, healthcare, cybersecurity
   - 25+ quantum algorithms (QAOA, DCQS, QGAN, Grover's, etc.)
   - Marketplace services and documentation links
   - Research paper references (one reference platform among many)

2. **kipu_wget_urls.txt** - URL list for batch processing

3. **kipu_analysis.md** - (Status: To be verified)

4. **hubo-workbench/** - HUBO/QUBO benchmarking workbench (2025-01-07)
   - Full Python package with Typer CLI (`hubo gen-instances`, `run-bench`, `report`)
   - HUBO/QUBO models with Rosenberg reduction (higher-order → quadratic)
   - Problem encoders: AssignmentProblem (bipartite matching with one-hot constraints)
   - Solvers: SA (simulated annealing), ILP (via PuLP), extensible plugin system
   - Benchmark runner: deterministic seeds, JSONL traces, advantage-shaped metrics
   - Metrics: TTA (time-to-approximate-solution), success probability, approximation ratio
   - Docs: methodology.md, metrics.md; tests for energy consistency and determinism
   - See README in `hubo-workbench/` for quickstart

### In Progress
- Surveying publicly available QML resources across platforms
- Identifying production-ready encodings and circuits
- Evaluating trainability analysis tools and techniques

## Technical Stack

### Target Frameworks (Platform Outputs)
- **Qiskit** - IBM's quantum SDK
- **PennyLane** - Xanadu's differentiable quantum programming
- Output circuits should be portable across these frameworks

### Reference Platforms (Research Sources)
- **Kipu Quantum Hub** (https://dashboard.hub.kipu-quantum.com/)
- **IBM Quantum** / Qiskit ecosystem
- **PennyLane** tutorials & demos
- **TensorFlow Quantum**
- **Amazon Braket**
- Academic implementations (arXiv, GitHub)

### Core QML Techniques (Priority)
1. **Data Encoding Methods**
   - Angle encoding (Rx, Ry, Rz rotations)
   - Amplitude encoding (state preparation)
   - IQP (Instantaneous Quantum Polynomial) encoding
   - Hamiltonian evolution encoding
   - Data re-uploading architectures

2. **QML Algorithms**
   - Variational Quantum Classifiers (VQC)
   - Quantum Kernel Estimation (QKE)
   - Quantum Neural Networks (QNN)
   - Hybrid classical-quantum models

3. **Trainability Considerations**
   - Barren plateau detection & mitigation
   - Parameter initialization strategies
   - Cost function design
   - Circuit expressibility vs. trainability tradeoffs

### Supporting Technologies
- Classical ML frameworks (PyTorch, scikit-learn) for baselines
- Experiment tracking (MLflow, Weights & Biases potential)
- Dataset handling & preprocessing
- Hardware noise simulators

## Target Use Cases & Applications

### Priority Application Domains (QML-Relevant)
1. **Structured Data Classification**
   - Financial fraud detection
   - Network intrusion detection
   - Medical diagnostics with tabular data
   - Customer segmentation

2. **Kernel-Based Learning**
   - Small-sample learning problems
   - Feature space expansion via quantum kernels
   - Domains where classical kernels struggle

3. **Optimization with ML Components**
   - Combinatorial optimization (QAOA-style)
   - Portfolio optimization with learned objectives
   - Scheduling & assignment with learned costs

4. **Hybrid Quantum-Classical Workflows**
   - Classical preprocessing → quantum encoding → classical post-processing
   - Quantum feature extraction for classical models
   - Quantum layers in classical neural networks

### QML Algorithms of Interest
1. **Core Focus (Priority)**
   - Variational Quantum Classifiers (VQC)
   - Quantum Kernel Estimation (QKE)
   - Data re-uploading circuits
   - Quantum Neural Networks (QNN)

2. **Supporting Techniques**
   - QAOA (for optimization problems)
   - Quantum Generative Models (QGAN)
   - Quantum Boltzmann Machines (if trainable)
   - Hybrid Transfer Learning

3. **Foundational (Reference Only)**
   - Grover's Search
   - HHL Algorithm
   - Quantum Fourier Transform
   - Deutsch-Josza (pedagogical)

## Design Principles

### Platform Philosophy
1. **Hardware-aware by default** - Noise, connectivity, and shot budgets are first-class citizens
2. **Reproducibility over novelty** - If it can't be benchmarked, it doesn't exist
3. **Opinionated but configurable** - Smart defaults, but allow customization
4. **Classical baselines are mandatory** - No quantum advantage claims without proof
5. **Developer-first experience** - Focus on usability, not academic purity

### Explicit Non-Goals
- ❌ General quantum algorithms marketplace
- ❌ Pure theory without runnable code
- ❌ Claims of advantage without baselines
- ❌ Deep circuits requiring fault-tolerant quantum computing (FTQC)
- ❌ Toy demos that don't scale to real problems

## Target Users

### Primary Personas
1. **QML Researchers** - Need tools for reproducible experiments
2. **Quantum Software Engineers** - Building production QML pipelines
3. **Applied ML Engineers exploring QC** - Want practical QML entry point
4. **Early-adopter industry labs** - Finance, materials, sensing, drug discovery

### User Needs
- Fast iteration on encoding strategies
- Confidence in trainability before hardware execution
- Fair comparison between quantum and classical approaches
- Portability across quantum hardware vendors

## Key Research Insights

### From Platform Survey
- **Kipu Quantum**: Strong on optimization, QAOA variants, some QGAN work
- **PennyLane**: Best differentiable programming, excellent tutorials
- **Qiskit**: Most mature ecosystem, extensive hardware access
- **TensorFlow Quantum**: Good hybrid models, but less active development

### Critical Findings
- **Data encoding is under-tooled**: Most platforms expect users to hand-code encodings
- **Barren plateaus are real**: Deep circuits often untrainable on NISQ hardware
- **Classical baselines often missing**: Many QML papers lack proper comparisons
- **Hardware noise matters more than expected**: Simulated advantage often disappears on real devices
- **HUBO benchmarking gap**: Kipu and others focus on HUBO (higher-order) problems, but tooling for reproducible benchmarks with advantage-shaped metrics is sparse; we built `hubo-workbench` to address this

## Research Resources

### Key Papers (Data Encoding Focus)
1. **Data Re-uploading** - https://arxiv.org/abs/1907.02085 (Pérez-Salinas et al.)
2. **Quantum Feature Maps** - Various IQP, Hamiltonian evolution papers
3. **Encoding Comparison Studies** - TBD (ongoing research)

### Key Papers (Trainability)
1. **Barren Plateaus in QNNs** - https://arxiv.org/abs/1803.11173 (McClean et al.)
2. **Cost Function Dependent Barren Plateaus** - https://arxiv.org/abs/2001.00550
3. **Parameter Initialization Strategies** - https://arxiv.org/abs/2108.13969

### Key Papers (QML Algorithms)
1. **QAOA** - https://arxiv.org/abs/1411.4028 (Farhi et al.)
2. **Quantum Kernels** - https://arxiv.org/abs/1906.10467 (Havlíček et al.)
3. **Variational Quantum Classifiers** - Multiple sources, PennyLane tutorials
4. **QGAN** - https://arxiv.org/abs/1901.00848
5. **Quantum Transfer Learning** - https://arxiv.org/abs/1902.01083

### Benchmarking References
1. **QML Benchmarks Paper** - https://arxiv.org/abs/2111.05292 (LaRose et al.)
2. **NISQ Algorithm Performance** - Various comparative studies

### Platform Documentation
- **PennyLane Docs**: https://pennylane.ai/
- **Qiskit Machine Learning**: https://qiskit.org/ecosystem/machine-learning/
- **Kipu Quantum**: https://docs.hub.kipu-quantum.com/
- **TensorFlow Quantum**: https://www.tensorflow.org/quantum

## Project Files Structure

```
quantum/
├── PROJECT_CONTEXT.md          # This file - project scope and decisions
├── kipu_urls.md                # Kipu platform reference (one of many sources)
├── kipu_wget_urls.txt          # URL list for batch processing
├── kipu_analysis.md            # Platform analysis (to verify)
├── hubo-workbench/             # HUBO/QUBO benchmarking workbench
│   ├── README.md
│   ├── pyproject.toml
│   ├── src/hubo/               # Core package
│   │   ├── problems/           # Problem encoders (assignment, scheduling, etc.)
│   │   ├── encoding/           # HUBO/QUBO models + Rosenberg reduction
│   │   ├── solvers/            # SA, ILP, + extensible plugin system
│   │   ├── bench/              # Runner, metrics, reports
│   │   └── cli.py              # Typer CLI
│   ├── tests/                  # Energy, determinism, reduction tests
│   ├── docs/                   # Methodology, metrics definitions
│   └── experiments/            # Configs, runs, results
└── [QML research & experiments TBD]
```

## Next Steps & Research Questions

### Immediate Research Priorities
1. **Encoding Survey**
   - Compare angle vs. amplitude vs. IQP encoding on simple datasets
   - Identify dataset characteristics that favor each encoding
   - Document tradeoffs: circuit depth, expressibility, trainability

2. **Trainability Analysis**
   - Test barren plateau detection heuristics
   - Evaluate existing tools (PennyLane, Qiskit)
   - Define "trainability score" methodology

3. **Benchmark Dataset Selection**
   - Identify QML-appropriate datasets (structured, small-sample)
   - Establish classical baselines for each
   - Define metrics: accuracy, sample efficiency, circuit resources

4. **Platform/Tool Evaluation**
   - Which framework best supports encoding experimentation? (PennyLane likely)
   - What existing libraries can be leveraged vs. built from scratch?
   - How to make outputs portable across Qiskit/PennyLane?

### Key Questions to Answer
- **Encoding**: Which encoding strategies are production-ready? Which need better tooling?
- **Trainability**: Can we reliably predict trainability before hardware execution?
- **Advantage**: Where has empirical quantum advantage been demonstrated with proper baselines?
- **Hardware**: Which NISQ devices are practical for QML today? (IBM, IonQ, Rigetti, etc.)
- **Gaps**: What's missing from existing platforms that we need to build?
- **HUBO/QUBO benchmarking**: Can we replicate Kipu-style optimization benchmarks with reproducible metrics and open classical baselines? (Answered: Yes, see `hubo-workbench`)

## Notes for Future Sessions

### Core Context to Remember
- **This is NOT a Kipu-focused project** - Kipu is one reference among many
- **Focus**: Building a developer-first QML platform (encoding, trainability, benchmarking) + optimization workbenches
- **Current Phase**: Research & discovery + prototype workbench development
- **Philosophy**: Hardware-aware, reproducible, classical baselines mandatory, developer-first
- **HUBO workbench**: Completed v0.1.0 with HUBO/QUBO encodings, SA/ILP solvers, benchmark runner, and metrics
- **Target Output**: Circuits usable in Qiskit & PennyLane

### Key Terminology (QML-Specific)
- **VQC**: Variational Quantum Classifier
- **QKE**: Quantum Kernel Estimation
- **QAOA**: Quantum Approximate Optimization Algorithm
- **NISQ**: Noisy Intermediate-Scale Quantum (current hardware era)
- **Barren Plateaus**: Trainability problem where gradients vanish
- **Data Re-uploading**: Encoding technique that repeats data encoding with trainable gates
- **IQP**: Instantaneous Quantum Polynomial (encoding method)
- **Ansatz**: Parameterized quantum circuit structure

### Repository Context
- **Git Status**: Currently on `main` branch
- **Untracked files**: kipu_analysis.md, kipu_urls.md, kipu_wget_urls.txt, PROJECT_CONTEXT.md
- Not yet committed to version control

### Critical Decision Points Ahead
1. **Which encoding methods to implement first?** (Angle, amplitude, or data re-uploading?)
2. **Which framework to prototype in?** (Likely PennyLane for flexibility)
3. **What datasets to use for benchmarking?** (Need small, structured, QML-appropriate)
4. **How to measure trainability?** (Gradient variance? Cost landscape analysis?)

---

## Session History Log

### 2025-01-07 - Project Definition & Context Setup
- **Clarified project mission**: Building developer-first QML platform, NOT Kipu documentation
- Established core thesis: encoding is bottleneck, hardware-aware by default, baselines mandatory
- Defined three product directions: Encoding Platform (core), Trainability Analyzer, Benchmarking Layer
- Created comprehensive PROJECT_CONTEXT.md reflecting actual goals
- Catalogued reference resources:
  - Kipu platform (one of many sources)
  - Key papers on encoding, trainability, QML algorithms
  - Platform documentation (PennyLane, Qiskit, TFQ)
- Identified immediate research priorities: encoding survey, trainability analysis, dataset selection
- **Key Insight**: Most platforms under-tool data encoding; users must hand-code feature maps

### 2025-01-07 (later) - HUBO Workbench v0.1.0 Built
- **Created `hubo-workbench/`**: Full Python package for HUBO/QUBO optimization benchmarking
- **Core capabilities**:
  - HUBO/QUBO models with energy eval, Rosenberg reduction (k-local → quadratic)
  - Problem encoders: AssignmentProblem (bipartite matching with penalties)
  - Solvers: SA (simulated annealing), ILP (via PuLP), extensible plugin system
  - Benchmark runner: deterministic seeds, JSONL traces, advantage-shaped metrics
  - Metrics: TTA (time-to-approximate-solution), success probability, approximation ratio
  - Typer CLI: `hubo gen-instances`, `run-bench`, `report`
  - Docs (methodology.md, metrics.md) + tests (energy, determinism, reduction)
- **Design philosophy**: Kipu-aligned (HUBO-first), reproducible, classical baselines mandatory
- **Next**: Add scheduling/ATFM problems, QAOA/SQA solvers, run first benchmarks
- **Key Insight**: Building reproducible benchmark infra is harder than individual solvers; automation + traceability are critical

---

## Long-Term Vision

A **neutral, open QML platform** that:
- Makes data encoding as easy as `model.encode(data, method='auto')`
- Predicts trainability before expensive hardware execution
- Requires classical baselines for any quantum advantage claim
- Outputs portable circuits (Qiskit, PennyLane, etc.)
- Enables fair, reproducible QML research and development

**North Star**: Do for QML what Hugging Face did for transformers - make best practices accessible, reproducible, and developer-friendly.

---

*Update this file after significant research findings, architectural decisions, or shifts in project direction.*
