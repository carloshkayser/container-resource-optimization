# Container Resource & Replica Optimization (Queueing Theory & Machine Learning)

This directory contains mathematical and data-driven optimization models to right-size containerized workloads (**CPU**, **RAM**, and **Replicas**) under latency Service Level Agreements (**SLA**), comparing analytical **M/M/1 Queueing Theory** against **Machine Learning Surrogate Models**.

---

## Overview of the Cases

- **Case 1: Single-Container Resource Optimization (Vertical Scaling)**
  - **Approach:** Analytical M/M/1 Queueing Theory ($\mu = \alpha \cdot \text{CPU}$, $W = 1 / (\mu - \lambda)$).
  - **Problem Class:** Linear Programming (LP) / Non-linear Programming (NLP).
  - **Objectives:** Minimize allocated container resources ($\text{CPU}, \text{RAM}$).
  - **Script:** [`container_queue_model.py`](./container_queue_model.py)
  - **Notebook:** [`container_queue_analysis.ipynb`](./container_queue_analysis.ipynb)

- **Case 2: Multi-Objective Container & Replica Optimization (Horizontal + Vertical Scaling)**
  - **Approach:** Load-Balanced Parallel M/M/1 Queues ($\lambda_{\text{rep}} = \lambda / N$).
  - **Problem Class:** Multi-Objective Mixed-Integer Linear Programming (MILP) / MINLP.
  - **Objectives:**
    - **Objective 1:** Minimize per-container $\text{CPU}$ & $\text{RAM}$.
    - **Objective 2:** Minimize active container replicas $N \in [1, 5]$.
  - **Script:** [`container_replicas_queue_model.py`](./container_replicas_queue_model.py)
  - **Notebook:** [`container_replicas_queue_analysis.ipynb`](./container_replicas_queue_analysis.ipynb)

- **Case 3: Data-Driven Resource Optimization using Machine Learning (ML Surrogate)**
  - **Approach:** Empirical ML Surrogate Models (Multi-Layer Perceptron / MLP, Random Forest, Ridge) trained on real microservice telemetry to capture non-linear CPU throttling and saturation cliffs.
  - **Problem Class:** Surrogate-Based Optimization / Non-linear Optimization.
  - **Objectives:** Minimize container resources while strictly satisfying empirical latency SLAs under real-world runtime behavior.
  - **Script:** [`container_ml_model.py`](./container_ml_model.py)
  - **Notebooks:**
    1. **Iterative Model Training & Architecture Search:** [`container_ml_training.ipynb`](./container_ml_training.ipynb)
    2. **Surrogate Optimization & Queueing Comparison:** [`container_ml_analysis.ipynb`](./container_ml_analysis.ipynb)

---

## Case 1: Single-Container Resource Optimization (Vertical Scaling)

### Problem Description
We model a single container instance as an **M/M/1 queue** receiving Poisson arrivals at rate $\lambda$ ($\text{req/s}$) and processing requests at rate $\mu = \alpha_{\text{cpu}} \cdot \text{CPU}$ ($\text{req/s}$).

### Mathematical Formulation

**Decision Variables:**
- $\text{CPU} \in [500, 2000]\text{ millicores}$
- $\text{RAM} \in [128, 4096]\text{ MB}$
- $\mu \ge 0\text{ req/s}$

**Objective Function:**

$$
\min_{\text{CPU}, \text{RAM}, \mu} \quad w_{\text{cpu}} \frac{\text{CPU}}{\text{CPU}_{\max}} + w_{\text{ram}} \frac{\text{RAM}}{\text{RAM}_{\max}}
$$

**Constraints:**
- **Service Capacity:** $\mu = \alpha_{\text{cpu}} \cdot \text{CPU}$
- **Memory Provisioning:** $\text{RAM} \ge \text{RAM}_{\text{base}} + \beta_{\text{ram}} \cdot \mu$
- **M/M/1 Response Time SLA ($W \le W_{\max}$ and $\mu > \lambda$):**

$$
W = \frac{1}{\mu - \lambda} \le W_{\max} \iff \mu - \lambda \ge \frac{1}{W_{\max}}
$$

---

## Case 2: Multi-Objective Container & Replica Optimization (Horizontal + Vertical Scaling)

### Problem Description
Case 2 extends Case 1 by allowing the system to add or remove homogeneous container replicas $N \in \{1, 2, 3, 4, 5\}$ behind an even load balancer ($\lambda_{\text{rep}} = \lambda / N$).

### Mathematical Formulation (Exact MILP)

**Decision Variables:**
- $\text{CPU} \in [500, 2000]\text{ mCPU}$ (millicores allocated per container replica)
- $\text{RAM} \in [128, 4096]\text{ MB}$ (Megabytes allocated per container replica)
- $N \in \{1, 2, 3, 4, 5\}$ (integer active container replicas)
- $z_n \in \{0, 1\}$ for $n \in \{1, \dots, 5\}$ (binary indicator: $z_n = 1 \iff N = n$)

**Objective 1 (Per-Container Resources):**

$$
f_1(\text{CPU}, \text{RAM}) = w_{\text{cpu}} \frac{\text{CPU}}{\text{CPU}_{\max}} + w_{\text{ram}} \frac{\text{RAM}}{\text{RAM}_{\max}}
$$

**Objective 2 (Number of Replicas):**

$$
f_2(N) = \frac{N}{N_{\max}}
$$

**Combined Weighted-Sum Objective:**

$$
\min_{\text{CPU}, \text{RAM}, N} \quad w_{\text{resources}} \cdot f_1(\text{CPU}, \text{RAM}) + w_{\text{replicas}} \cdot f_2(N)
$$

**Constraints:**
- **Replica Selection:** $\sum_{n=1}^{5} z_n = 1, \quad N = \sum_{n=1}^{5} n \cdot z_n$
- **Exact Linearized Load Balancing:** $\lambda_{\text{rep}} = \sum_{n=1}^{5} \left(\frac{\lambda}{n}\right) z_n$
- **Per-Replica Service Capacity & RAM:** $\mu = \alpha_{\text{cpu}} \cdot \text{CPU}, \quad \text{RAM} \ge \text{RAM}_{\text{base}} + \beta_{\text{ram}} \cdot \mu$
- **Per-Replica M/M/1 Latency SLA:** $\mu - \lambda_{\text{rep}} \ge \frac{1}{W_{\max}}$

---

## Case 3: Data-Driven Resource Optimization using Machine Learning (ML Surrogate)

### Problem Description
Real-world containers running on Linux/Kubernetes experience non-linear phenomenon that analytical M/M/1 models fail to capture:
- **CFS Bandwidth Throttling:** Linux Completely Fair Scheduler enforces container CPU limits through quotas; once exhausted mid-period, execution halts and latency spikes non-linearly.
- **Resource Saturation & GC Pressure:** Memory consumption and cache thrashing create performance cliffs.
- **Payload Heterogeneity:** Heavy I/O or media payloads have radically different response curves than simple HTTP pings.

In Case 3, we use an empirical public benchmark dataset (**UPB Microservices / Nginx Benchmark**) to train regression models that predict response time:

$$
\hat{W} = f_{\text{ML}}(\text{CPU}, \text{RAM}, \lambda, \text{payload})
$$

### Case 3 Workflow: Two Dedicated Notebooks
1. **Iterative Model Training & Architecture Search:** [`container_ml_training.ipynb`](./container_ml_training.ipynb)
   - Exploratory Data Analysis (EDA) of 1,598 container telemetry measurements.
   - Target log-transformation $\log(1 + W)$ and feature standardization.
   - Regularization tuning on Ridge Regression baseline.
   - Tree ensemble depth & estimator tuning on Random Forest.
   - Multi-Layer Perceptron (MLP) architecture search (`(32,)`, `(64, 32)`, `(128, 64, 32)`) and training loss convergence curves.
   - Cross-validation and artifact persistence to `models/`.
2. **Surrogate Optimization & Queueing Comparison:** [`container_ml_analysis.ipynb`](./container_ml_analysis.ipynb)
   - 2D Response Surface visualization and contour map of the MLP surrogate.
   - Solving container sizing $\min \text{Cost}(\text{CPU}, \text{RAM})$ s.t. $\hat{W}_{\text{MLP}} \le W_{\max}$.
   - SLA sensitivity curve demonstrating exponential cost of tail latency.
   - Side-by-side comparison showing that analytical M/M/1 severely under-provisions CPU (allocating $100\text{ mCPU}$ leading to $2126\text{ ms}$ latency), while MLP allocates $440\text{ mCPU}$ achieving $490\text{ ms}$ and guaranteeing SLA compliance.

### Optimization Formulation (ML Surrogate)

$$
\min_{\text{CPU}, \text{RAM}} \quad w_{\text{cpu}} \frac{\text{CPU}}{\text{CPU}_{\max}} + w_{\text{ram}} \frac{\text{RAM}}{\text{RAM}_{\max}}
$$

$$
\begin{aligned}
\text{subject to} \quad & \hat{W}_{\text{MLP}}(\text{CPU}, \text{RAM}, \lambda, \text{payload}) \le W_{\max} \\
& \text{CPU} \in [100, 1000]\text{ mCPU} \\
& \text{RAM} \in [64, 512]\text{ MB}
\end{aligned}
$$

---

## Environment Setup & How to Run

Dependencies are managed using **uv** and declared in [`pyproject.toml`](./pyproject.toml).

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Case 1 (Single Container - M/M/1)
```bash
uv run python3 container_queue_model.py
```

### 3. Run Case 2 (Multi-Objective Replicas - Parallel M/M/1)
```bash
uv run python3 container_replicas_queue_model.py
```

### 4. Run Case 3 (Data-Driven ML Surrogate Optimization)
```bash
# Sizing container with MLP surrogate and comparing against M/M/1
uv run python3 container_ml_model.py --w_max 500 --compare_mm1

# Retrain models from scratch if desired
uv run python3 container_ml_model.py --retrain
```

### 5. Open the Notebooks
```bash
uv run jupyter lab
```
* **Case 1:** [`container_queue_analysis.ipynb`](./container_queue_analysis.ipynb)
* **Case 2:** [`container_replicas_queue_analysis.ipynb`](./container_replicas_queue_analysis.ipynb)
* **Case 3 (Training):** [`container_ml_training.ipynb`](./container_ml_training.ipynb)
* **Case 3 (Optimization):** [`container_ml_analysis.ipynb`](./container_ml_analysis.ipynb)
