"""
Container Resource Allocation Optimization using Queueing Theory (M/M/1) and Pyomo.

Mathematical Formulation:
-------------------------
- Decision Variables:
    * CPU in [500, 2000] (millicores, mCPU)
    * RAM in [ram_min, ram_max] (Megabytes, MB)
    * mu >= 0 (Container service capacity in requests/second)

- Queueing System (M/M/1):
    * Arrival rate: lambda_req (requests/second)
    * Service rate: mu = alpha_cpu * CPU (requests/second)
    * Utilization: rho = lambda_req / mu < 1
    * Average Response Time (Wait + Service): W = 1 / (mu - lambda_req)
    * Average Queue Length (Little's Law): L = lambda_req * W

- Constraints:
    1. Service Capacity: mu == alpha_cpu * CPU
    2. Latency SLA (W <= W_max) & Queue Stability (mu > lambda_req):
       1 / (mu - lambda_req) <= W_max  <==>  mu - lambda_req >= 1 / W_max
    3. Memory Provisioning:
       RAM >= ram_base + beta_ram * mu

- Objective:
    Minimize weighted sum of allocated resources (CPU and RAM):
        min  w_cpu * (CPU / CPU_max) + w_ram * (RAM / RAM_max)
"""

import pyomo.environ as pyo


def build_container_queue_model(
    lambda_req: float = 50.0,
    w_max: float = 0.05,
    alpha_cpu: float = 0.08,
    ram_base: float = 128.0,
    beta_ram: float = 4.0,
    cpu_bounds: tuple[float, float] = (500.0, 2000.0),
    ram_bounds: tuple[float, float] = (128.0, 4096.0),
    w_cpu: float = 1.0,
    w_ram: float = 1.0,
    nonlinear: bool = False,
) -> pyo.ConcreteModel:
    """
    Builds the Pyomo optimization model for an M/M/1 container system.

    Parameters
    ----------
    lambda_req : float
        Mean arrival rate of requests (req/s).
    w_max : float
        Maximum allowed mean response time (SLA in seconds, e.g., 0.05s = 50ms).
    alpha_cpu : float
        Service capacity per millicore of CPU (req/s per mCPU).
    ram_base : float
        Base memory footprint required by the container runtime (MB).
    beta_ram : float
        Additional memory required per unit of service capacity (MB per req/s).
    cpu_bounds : tuple[float, float]
        Lower and upper bounds for CPU allocation in millicores (500 to 2000 mCPU).
    ram_bounds : tuple[float, float]
        Lower and upper bounds for RAM allocation in MB.
    w_cpu : float
        Weight for normalized CPU allocation in the objective function.
    w_ram : float
        Weight for normalized RAM allocation in the objective function.
    nonlinear : bool
        If True, models W = 1 / (mu - lambda) directly as a nonlinear constraint
        (requires an NLP solver like 'ipopt'). If False, uses the exact linear
        reformulation mu - lambda >= 1 / w_max (compatible with LP/NLP solvers).
    """
    model = pyo.ConcreteModel(name="Container_MM1_Resource_Optimization")

    # =========================================================================
    # 1. Parameters
    # =========================================================================
    model.lambda_req = pyo.Param(initialize=lambda_req, doc="Arrival rate (req/s)")
    model.w_max = pyo.Param(initialize=w_max, doc="Max average response time SLA (s)")
    model.alpha_cpu = pyo.Param(
        initialize=alpha_cpu, doc="Throughput per CPU millicore (req/s/mCPU)"
    )
    model.ram_base = pyo.Param(initialize=ram_base, doc="Base RAM overhead (MB)")
    model.beta_ram = pyo.Param(
        initialize=beta_ram, doc="RAM required per unit of capacity (MB/(req/s))"
    )
    model.w_cpu = pyo.Param(initialize=w_cpu, doc="Weight for normalized CPU in objective")
    model.w_ram = pyo.Param(initialize=w_ram, doc="Weight for normalized RAM in objective")

    # =========================================================================
    # 2. Decision Variables
    # =========================================================================
    model.cpu = pyo.Var(
        bounds=cpu_bounds,
        initialize=cpu_bounds[0],
        doc="Allocated CPU (millicores)",
    )
    model.ram = pyo.Var(
        bounds=ram_bounds,
        initialize=ram_bounds[0],
        doc="Allocated RAM (MB)",
    )
    model.mu = pyo.Var(
        within=pyo.NonNegativeReals,
        initialize=alpha_cpu * cpu_bounds[0],
        doc="Container service rate (req/s)",
    )

    # =========================================================================
    # 3. Constraints
    # =========================================================================
    # (a) Container capacity as a function of CPU allocated
    @model.Constraint(doc="Service rate proportional to allocated CPU")
    def service_capacity_rule(m):
        return m.mu == m.alpha_cpu * m.cpu

    # (b) Memory capacity requirement to support container service capacity / queue
    @model.Constraint(doc="Minimum RAM needed to sustain the service capacity")
    def ram_requirement_rule(m):
        return m.ram >= m.ram_base + m.beta_ram * m.mu

    # (c) M/M/1 Queueing SLA Constraint: W = 1 / (mu - lambda) <= w_max
    if nonlinear:
        model.W = pyo.Var(
            bounds=(1e-6, w_max),
            initialize=w_max,
            doc="Mean response time in M/M/1 system (s)",
        )

        @model.Constraint(doc="Nonlinear M/M/1 response time relation: W * (mu - lambda) == 1")
        def mm1_response_time_rule(m):
            return m.W * (m.mu - m.lambda_req) == 1.0
    else:
        @model.Constraint(
            doc="Linearized M/M/1 SLA and stability constraint: mu - lambda >= 1 / w_max"
        )
        def mm1_sla_rule(m):
            return m.mu - m.lambda_req >= 1.0 / m.w_max

    # =========================================================================
    # 4. Queueing Performance Expressions (for reporting/inspection)
    # =========================================================================
    @model.Expression(doc="Server utilization (rho = lambda / mu)")
    def utilization(m):
        return m.lambda_req / m.mu

    @model.Expression(doc="Mean response time in system W (s)")
    def response_time(m):
        return 1.0 / (m.mu - m.lambda_req)

    @model.Expression(doc="Mean waiting time in queue Wq (s)")
    def waiting_time_queue(m):
        return m.lambda_req / (m.mu * (m.mu - m.lambda_req))

    @model.Expression(doc="Mean number of requests in system L (Little's Law)")
    def avg_requests_in_system(m):
        return m.lambda_req / (m.mu - m.lambda_req)

    @model.Expression(doc="Mean number of requests in queue Lq")
    def avg_requests_in_queue(m):
        return (m.lambda_req**2) / (m.mu * (m.mu - m.lambda_req))

    # =========================================================================
    # 5. Objective Function: Minimize Allocated Resources
    # =========================================================================
    cpu_max = cpu_bounds[1]
    ram_max = ram_bounds[1]

    @model.Objective(sense=pyo.minimize, doc="Minimize normalized CPU and RAM allocation")
    def total_resource_cost(m):
        return w_cpu * (m.cpu / cpu_max) + w_ram * (m.ram / ram_max)

    return model


def solve_model(model: pyo.ConcreteModel, solver_name: str | None = None) -> pyo.SolverResults:
    """
    Solves the Pyomo model using the specified solver or automatically detects an available one.
    """
    candidate_solvers = (
        [solver_name]
        if solver_name
        else ["glpk", "highs", "appsi_highs", "cbc", "ipopt", "scipy"]
    )

    for name in candidate_solvers:
        solver = pyo.SolverFactory(name)
        if solver.available(exception_flag=False):
            results = solver.solve(model, tee=False)
            model._solver_name = name
            model._solver_results = results
            return results

    raise RuntimeError(
        f"No compatible Pyomo solver found among {candidate_solvers}. "
        "Install 'glpk', 'highs' (highspy), or 'ipopt'."
    )


def print_solution(
    model: pyo.ConcreteModel, results: pyo.SolverResults | None = None
) -> None:
    """
    Prints:
    1. Problem Title
    2. Description of the Parameters Used
    3. Solver Status / Whether an Optimal Solution was Found
    4. Details of the Solution Found
    """
    res = results if results is not None else getattr(model, "_solver_results", None)
    solver_name = getattr(model, "_solver_name", "auto")

    lam_val = pyo.value(model.lambda_req)
    w_max_val = pyo.value(model.w_max)
    alpha_val = pyo.value(model.alpha_cpu)
    ram_base_val = pyo.value(model.ram_base)
    beta_ram_val = pyo.value(model.beta_ram)
    w_cpu_val = pyo.value(model.w_cpu)
    w_ram_val = pyo.value(model.w_ram)
    cpu_lb, cpu_ub = model.cpu.lb, model.cpu.ub
    ram_lb, ram_ub = model.ram.lb, model.ram.ub

    # 1. Problem Title
    print("=" * 72)
    print("  CASE 1: SINGLE-CONTAINER M/M/1 RESOURCE OPTIMIZATION PROBLEM")
    print("=" * 72)

    # 2. Description of Parameters Used
    print("Workload, SLA & System Parameters:")
    print(f"  Arrival Rate (lambda)         : {lam_val:.2f} req/s")
    print(f"  Max Response Time SLA (W_max) : {w_max_val * 1000:.2f} ms ({w_max_val:.4f} s)")
    print(f"  CPU Throughput (alpha_cpu)    : {alpha_val:.4f} req/s per millicore")
    print(f"  Base RAM Overhead (ram_base)  : {ram_base_val:.2f} MB")
    print(f"  Working RAM Factor (beta_ram) : {beta_ram_val:.2f} MB per (req/s)")
    print(f"  CPU Allocation Bounds         : [{cpu_lb:.0f}, {cpu_ub:.0f}] millicores")
    print(f"  RAM Allocation Bounds         : [{ram_lb:.0f}, {ram_ub:.0f}] MB")
    print(f"  Objective Weights (CPU, RAM)  : w_cpu={w_cpu_val:.2f}, w_ram={w_ram_val:.2f}")
    print("=" * 72)

    # 3. Solution Status Banner
    is_optimal = (
        res is not None
        and res.solver.status == pyo.SolverStatus.ok
        and res.solver.termination_condition == pyo.TerminationCondition.optimal
    )

    print("\n")
    print("*" * 72)
    if is_optimal or res is None:
        print("  OPTIMAL SOLUTION FOUND")
    else:
        status_str = f"{res.solver.status} ({res.solver.termination_condition})"
        print(f"  SOLUTION STATUS: {status_str.upper()}")
    print("*" * 72)

    if res is not None and not is_optimal:
        print(f"Solver '{solver_name}' did not converge to an optimal solution.")
        print("=" * 72)
        return

    # 4. Details of the Solution Found
    cpu_val = pyo.value(model.cpu)
    ram_val = pyo.value(model.ram)
    mu_val = pyo.value(model.mu)

    rho_val = pyo.value(model.utilization)
    w_val = pyo.value(model.response_time)
    wq_val = pyo.value(model.waiting_time_queue)
    l_val = pyo.value(model.avg_requests_in_system)
    lq_val = pyo.value(model.avg_requests_in_queue)

    print("Optimal Resource Allocation:")
    print(f"  CPU Allocated                 : {cpu_val:.2f} millicores (bounds: [{cpu_lb:.0f}, {cpu_ub:.0f}])")
    print(f"  RAM Allocated                 : {ram_val:.2f} MB (bounds: [{ram_lb:.0f}, {ram_ub:.0f}])")
    print(f"  Service Capacity (mu)         : {mu_val:.2f} req/s")
    print("-" * 72)
    print("Queueing Metrics (M/M/1 Steady State):")
    print(f"  Container Utilization (rho)   : {rho_val * 100:.2f}%")
    print(f"  Avg Response Time (W)         : {w_val * 1000:.2f} ms")
    print(f"  Avg Queue Waiting Time (Wq)   : {wq_val * 1000:.2f} ms")
    print(f"  Avg Requests in System (L)    : {l_val:.2f} req")
    print(f"  Avg Requests in Queue (Lq)    : {lq_val:.2f} req")
    print(f"  Objective Value               : {pyo.value(model.total_resource_cost):.4f}")
    print("=" * 72)


def plot_and_save_queue_model(
    model: pyo.ConcreteModel,
    assets_dir: str | None = None,
) -> None:
    """
    Generates and saves the analytical queueing plots and search space taxonomy
    for Part 1 into the assets directory.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    if assets_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = (
            os.path.join(base_dir, "..", "assets", "part1")
            if os.path.basename(base_dir) == "part1"
            else os.path.join(base_dir, "assets", "part1")
        )
    os.makedirs(assets_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    lambda_req = pyo.value(model.lambda_req)
    alpha_cpu = pyo.value(model.alpha_cpu)
    w_max = pyo.value(model.w_max)
    w_max_ms = w_max * 1000.0

    opt_cpu = pyo.value(model.cpu)
    opt_ram = pyo.value(model.ram)
    opt_w_ms = pyo.value(model.response_time) * 1000.0

    # 1. Response Time vs CPU
    cpu_stability = lambda_req / alpha_cpu
    cpu_vals = np.linspace(cpu_stability + 15, 2000, 400)
    mu_vals = alpha_cpu * cpu_vals
    w_total_ms = (1.0 / (mu_vals - lambda_req)) * 1000.0
    w_queue_ms = (lambda_req / (mu_vals * (mu_vals - lambda_req))) * 1000.0
    service_time_ms = (1.0 / mu_vals) * 1000.0

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(cpu_vals, w_total_ms, label=r"Total Response Time $W = W_q + 1/\mu$", color="#1f77b4", lw=2.5)
    ax.plot(cpu_vals, w_queue_ms, label=r"Queue Waiting Time $W_q$", color="#ff7f0e", ls="--", lw=2)
    ax.plot(cpu_vals, service_time_ms, label=r"Service Time $1/\mu$", color="#2ca02c", ls=":", lw=2)
    ax.axhline(w_max_ms, color="#d62728", ls="-.", lw=1.8, label=rf"Latency SLA ($W_{{\max}} = {w_max_ms:.0f}$ ms)")
    ax.axvline(cpu_stability, color="black", ls="--", lw=1.5, label=rf"Stability Limit ($\mu = \lambda \Rightarrow {cpu_stability:.0f}$ mCPU)")
    ax.axvspan(opt_cpu, 2000, color="#2ca02c", alpha=0.10, label=rf"SLA Feasible Region (CPU $\geq {opt_cpu:.0f}$ mCPU)")
    ax.scatter([opt_cpu], [opt_w_ms], color="#d62728", s=110, zorder=5, label=f"Optimal Allocation ({opt_cpu:.0f} mCPU, {opt_w_ms:.0f} ms)")
    ax.set_xlim(500, 2000)
    ax.set_ylim(0, 200)
    ax.set_title(rf"M/M/1 Latency vs. Allocated Container CPU ($\lambda = {lambda_req:.0f}$ req/s)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Allocated CPU (millicores)")
    ax.set_ylabel("Time (ms)")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    f1_path = os.path.join(assets_dir, "01_response_time_vs_cpu.png")
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Feasible Region & Contours
    cpu_grid = np.linspace(500, 2000, 300)
    ram_grid = np.linspace(128, 1200, 300)
    CPU_M, RAM_M = np.meshgrid(cpu_grid, ram_grid)
    OBJ = (CPU_M / 2000.0) + (RAM_M / 4096.0)
    cpu_min_sla = (lambda_req + 1.0 / w_max) / alpha_cpu
    ram_min_curve = 128.0 + 4.0 * (alpha_cpu * cpu_grid)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    cs = ax.contour(CPU_M, RAM_M, OBJ, levels=12, cmap="Blues", alpha=0.9, linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt="Obj: %.2f")
    ax.fill_between(cpu_grid, ram_min_curve, 1200, where=(cpu_grid >= cpu_min_sla), color="#2ca02c", alpha=0.18, label="Feasible Region (SLA & Constraints Met)")
    ax.axvline(cpu_min_sla, color="#d62728", lw=2.2, ls="--", label=rf"M/M/1 SLA Boundary (CPU $\geq {cpu_min_sla:.0f}$ mCPU)")
    ax.plot(cpu_grid, ram_min_curve, color="#9467bd", lw=2.2, label=r"RAM Capacity Constraint ($\mathrm{RAM} \geq 128 + 4\mu$)")
    ax.scatter([opt_cpu], [opt_ram], color="#d62728", s=130, zorder=6, edgecolors="black", label=f"Optimal Solution ({opt_cpu:.0f} mCPU, {opt_ram:.0f} MB)")
    ax.set_xlim(500, 2000)
    ax.set_ylim(128, 1100)
    ax.set_title("2D Optimization Space: Feasible Region & Objective Contours", fontweight="bold")
    ax.set_xlabel("Allocated CPU (millicores)")
    ax.set_ylabel("Allocated RAM (MB)")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    f2_path = os.path.join(assets_dir, "04_feasible_region_contours.png")
    plt.savefig(f2_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated and saved Part 1 plots to: {assets_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 1: Single-Container M/M/1 Resource Optimization")
    parser.add_argument("--lambda_req", type=float, default=50.0, help="Arrival rate in req/s (default: 50.0)")
    parser.add_argument("--w_max", type=float, default=0.05, help="Max mean response time SLA in seconds (default: 0.05)")
    parser.add_argument("--plot", action="store_true", help="Generate and save visualization plots to assets/part1/")
    args = parser.parse_args()

    # Build and solve the model for specified scenario
    container_model = build_container_queue_model(
        lambda_req=args.lambda_req,
        w_max=args.w_max,
        alpha_cpu=0.08,
        ram_base=128.0,
        beta_ram=4.0,
        cpu_bounds=(500.0, 2000.0),
    )

    solver_results = solve_model(container_model)
    print_solution(container_model, solver_results)

    if args.plot:
        plot_and_save_queue_model(container_model)
