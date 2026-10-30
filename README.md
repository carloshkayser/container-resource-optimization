# Container Resource & Replica Optimization: From Queueing Theory to Machine Learning

> **A Three-Part Compendium on Cloud Systems and Mathematical Sizing**  
> *Theoretical and Practical Optimization of CPU, RAM, and Microservice Replicas under Latency Service Level Agreements (SLAs)*

This repository, divided into **three parts**, explores the progression of resource allocation and autoscaling techniques for containerized workloads in modern Linux cloud environments (e.g., Kubernetes, Docker).

---

## Repository Structure and Compendium Parts

The codebase is organized into **three self-contained directories**, each equipped with its dedicated textbook part (`README.md`), mathematical model scripts (`.py`), interactive Jupyter notebooks (`.ipynb`), datasets, and persisted artifacts:

### [Part 1: Single-Container Vertical Scaling via Queueing Theory (M/M/1)](./part1/README.md)
* **Directory:** [`part1/`](./part1/)
* **Focus:** Optimal sizing of **CPU** (in millicores) and **RAM** (in Megabytes) for an isolated container instance.
* **Mathematical Model:** Analytical M/M/1 Queueing Theory with processor-dependent service capacity $\mu = \alpha_{\text{cpu}} \cdot \text{CPU}$ and mean residence time $W = \frac{1}{\mu - \lambda} \le W_{\max}$.
* **Optimization Formulation:** Proves and implements an exact linear reformulation of the hyperbolic SLA constraint ($\mu - \lambda \ge 1/W_{\max}$), yielding a continuous Linear Program (LP) solved by HiGHS/Pyomo in sub-millisecond time.
* **Contents:**
  - [`part1/README.md`](./part1/README.md): Complete textbook chapter with mathematical proofs, Little's Law derivations, and asymptotic analysis.
  - [`part1/container_queue_model.py`](./part1/container_queue_model.py): Pyomo optimization model (supporting exact LP and raw NLP via Ipopt).
  - [`part1/container_queue_analysis.ipynb`](./part1/container_queue_analysis.ipynb): Interactive notebook with 3D cost surfaces, SLA sensitivity curves, and traffic sweeps.

---

### [Part 2: Joint Horizontal and Vertical Scaling (Parallel M/M/1 Queues & MILP)](./part2/README.md)
* **Directory:** [`part2/`](./part2/)
* **Focus:** Simultaneous optimization of **how many container replicas to run ($N \in \{1, \dots, 5\}$)** and **how much hardware to allocate per replica ($\text{CPU}, \text{RAM}$)** behind an HTTP load balancer.
* **Mathematical Model:** Employs the Poisson Splitting Theorem to model $N$ homogeneous parallel M/M/1 queues receiving load-balanced arrival rates $\lambda_{\text{rep}} = \lambda / N$.
* **Optimization Formulation:** Formulates a Multi-Objective Mixed-Integer Linear Program (MILP) that exacts a multiple-choice linearization of $\lambda / N$ using binary indicators ($z_n$). Balances per-pod resource footprint against cluster coordination and replica multiplicity costs.
* **Contents:**
  - [`part2/README.md`](./part2/README.md): Complete textbook chapter analyzing scale-up vs. scale-out, the "replica tax" on headroom, and the discrete Pareto front.
  - [`part2/container_replicas_queue_model.py`](./part2/container_replicas_queue_model.py): Pyomo MILP optimization model.
  - [`part2/container_replicas_queue_analysis.ipynb`](./part2/container_replicas_queue_analysis.ipynb): Interactive notebook with stacked cost breakdowns, replica sweeps, and aggregate cluster footprint plots.

---

### [Part 3: Data-Driven Resource Optimization via Machine Learning (ML Surrogates)](./part3/README.md)
* **Directory:** [`part3/`](./part3/)
* **Focus:** Overcoming the real-world breakdown of analytical queueing models caused by Linux **CFS Bandwidth Throttling** (`cpu.cfs_quota_us`), memory pressure, Garbage Collection, and payload heterogeneity.
* **Mathematical Model:** Surrogate-Based Optimization (Metamodeling). Trains non-linear Multi-Layer Perceptrons (MLP), Random Forests, and Ridge regressors on empirical microservice telemetry (UPB / Nginx benchmark) to serve as a high-fidelity constraint oracle:
  $$\min_{\text{CPU}, \text{RAM}} \quad \text{Cost}(\text{CPU}, \text{RAM}) \quad \text{subject to} \quad \hat{W}_{\text{MLP}}(\text{CPU}, \text{RAM}, \lambda, \text{payload}) \le W_{\max}$$
* **Core Empirical Finding:** Theoretical M/M/1 severely under-provisions CPU (allocating $100\text{ mCPU}$, producing $2126\text{ ms}$ latency and **violating the $500\text{ ms}$ SLA by 425%** due to Linux CFS throttling), whereas the MLP surrogate allocates $440\text{ mCPU}$, achieving $490\text{ ms}$ and **guaranteeing SLA compliance in production**.
* **Contents:**
  - [`part3/README.md`](./part3/README.md): Complete textbook chapter detailing Linux cgroups scheduling, surrogate optimization theory, and the M/M/1 vs. MLP showdown.
  - [`part3/container_ml_model.py`](./part3/container_ml_model.py): Production training and optimization script.
  - [`part3/container_ml_training.ipynb`](./part3/container_ml_training.ipynb): Notebook 1 (EDA, log-transform engineering, neural architecture search, cross-validation).
  - [`part3/container_ml_analysis.ipynb`](./part3/container_ml_analysis.ipynb): Notebook 2 (2D response surface visualization, SLA sensitivity curves, M/M/1 vs. MLP showdown).
  - [`part3/data/`](./part3/data/): Empirical benchmark telemetry dataset (`container_performance_web1.csv`).
  - [`part3/models/`](./part3/models/): Persisted joblib artifacts (`models.joblib`, `scaler.joblib`, `metrics.joblib`).

---

## Environment Setup and Execution

Dependencies are managed using **uv** (an ultra-fast Python package and project manager written in Rust).

### 1. Install & Sync Virtual Environment
From the repository root:

```bash
uv sync
```

### 2. Execute Python Scripts

Each script solves the optimization model and can optionally render and save the figures directly to `assets/` by passing `--plot`:

```bash
# Part 1: Single-Container Analytical M/M/1 Optimization (add --plot to save figures)
uv run python part1/container_queue_model.py --plot

# Part 2: Multi-Objective Parallel M/M/1 Replicas Optimization (add --plot to save figures)
uv run python part2/container_replicas_queue_model.py --plot

# Part 3: ML Surrogate Optimization & Comparison with M/M/1 (add --plot to save figures)
uv run python part3/container_ml_model.py --w_max 500 --compare_mm1 --plot
```

### 3. Launch Jupyter Lab for Interactive Notebooks

```bash
uv run jupyter lab
```

Navigate to any notebook from the Jupyter file browser:
- `part1/container_queue_analysis.ipynb`
- `part2/container_replicas_queue_analysis.ipynb`
- `part3/container_ml_training.ipynb`
- `part3/container_ml_analysis.ipynb`

---

## Cross-Part Comparative Synthesis

| Dimension | Part 1: Single M/M/1 Queue | Part 2: Parallel M/M/1 Replicas | Part 3: Machine Learning Surrogate |
| :--- | :---: | :---: | :---: |
| **Directory** | `part1/` | `part2/` | `part3/` |
| **Optimization Variables** | Vertical $\text{CPU}, \text{RAM}$ | Vertical $\text{CPU}, \text{RAM}$ + Horizontal $N$ | Continuous $\text{CPU}, \text{RAM}$ under cgroups |
| **Foundational Paradigm** | Classical Queueing Theory | Queueing Theory + Integer Programming | Empirical Telemetry + Deep Learning |
| **Problem Class** | Linear Programming (LP) | Mixed-Integer Linear Programming (MILP) | Non-Linear Black-Box Optimization |
| **Solver Algorithm** | HiGHS / Simplex / Ipopt | HiGHS (Branch-and-Bound) | Adaptive Grid Search / SLSQP |
| **Captures CFS Throttling?** | ❌ No (assumes linear capacity) | ❌ No (assumes pure $\lambda/N$ division) | ✅ Yes (learns real Linux kernel behavior) |
| **Dedicated Textbook** | [Part 1 Textbook](./part1/README.md) | [Part 2 Textbook](./part2/README.md) | [Part 3 Textbook](./part3/README.md) |
