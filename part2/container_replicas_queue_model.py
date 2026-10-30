"""
Case 2: Multi-Objective Container Resource & Replica Optimization (Horizontal + Vertical Scaling)
using Queueing Theory (Load-Balanced Parallel M/M/1) and Pyomo.

Mathematical Formulation:
-------------------------
- Decision Variables:
    * CPU in [500, 2000] (millicores allocated per container replica)
    * RAM in [ram_min, ram_max] (Megabytes allocated per container replica)
    * N (replicas) in {1, 2, 3, 4, 5} (integer number of active container replicas)
    * mu >= 0 (Service capacity per container replica in requests/second)

- Load-Balanced Queueing System (N Homogeneous Parallel M/M/1 Containers):
    * Total arrival rate: lambda_req (requests/second)
    * Per-replica arrival rate: lambda_rep = lambda_req / N
    * Per-replica service rate: mu = alpha_cpu * CPU
    * Per-replica utilization: rho = (lambda_req / N) / mu < 1
    * Mean Response Time: W = 1 / (mu - lambda_req / N) <= W_max

- Multi-Objective Formulation:
    * Objective 1 (Vertical Resources):
        f_1(CPU, RAM) = w_cpu * (CPU / CPU_max) + w_ram * (RAM / RAM_max)
    * Objective 2 (Horizontal Replicas):
        f_2(N) = N / N_max
    * Combined Weighted-Sum Objective:
        min  w_resources * f_1(CPU, RAM) + w_replicas * f_2(N)
"""

import pyomo.environ as pyo


def build_container_replicas_model(
    lambda_req: float = 120.0,
    w_max: float = 0.05,
    alpha_cpu: float = 0.08,
    ram_base: float = 128.0,
    beta_ram: float = 4.0,
    cpu_bounds: tuple[float, float] = (500.0, 2000.0),
    ram_bounds: tuple[float, float] = (128.0, 4096.0),
    replica_bounds: tuple[int, int] = (1, 5),
    fixed_replicas: int | None = None,
    w_cpu: float = 1.0,
    w_ram: float = 1.0,
    w_resources: float = 1.0,
    w_replicas: float = 1.0,
    nonlinear: bool = False,
) -> pyo.ConcreteModel:
    """
    Builds the Case 2 Pyomo MILP/MINLP model for joint vertical (CPU/RAM) and
    horizontal (replicas) container scaling under an M/M/1 latency SLA.

    Parameters
    ----------
    lambda_req : float
        Total incoming request arrival rate to the load balancer (req/s).
    w_max : float
        Maximum allowed mean response time (SLA in seconds, e.g., 0.05s = 50ms).
    alpha_cpu : float
        Service capacity per millicore of CPU (req/s per mCPU).
    ram_base : float
        Base memory footprint required by each container replica (MB).
    beta_ram : float
        Additional memory required per unit of service capacity (MB per req/s).
    cpu_bounds : tuple[float, float]
        Lower and upper bounds for per-container CPU allocation in millicores (500 to 2000).
    ram_bounds : tuple[float, float]
        Lower and upper bounds for per-container RAM allocation in MB.
    replica_bounds : tuple[int, int]
        Lower and upper bounds for the integer number of replicas (default: 1 to 5).
    w_cpu, w_ram : float
        Internal weights for normalized CPU and RAM in Objective 1.
    w_resources : float
        Weight for Objective 1 (minimizing per-container CPU & RAM resources).
    w_replicas : float
        Weight for Objective 2 (minimizing the number of container replicas).
    nonlinear : bool
        If True, uses direct nonlinear division lambda_req / replicas (MINLP).
        If False, uses exact binary selection variables z[n] so the problem is
        a pure Mixed-Integer Linear Program (MILP) solvable by GLPK/HiGHS/CBC.
    """
    model = pyo.ConcreteModel(name="Container_MultiObjective_Replicas_MM1")

    n_min, n_max = int(replica_bounds[0]), int(replica_bounds[1])
    model.N_SET = pyo.RangeSet(n_min, n_max)

    # =========================================================================
    # 1. Parameters
    # =========================================================================
    model.lambda_req = pyo.Param(
        initialize=lambda_req, doc="Total system arrival rate (req/s)"
    )
    model.w_max = pyo.Param(initialize=w_max, doc="Max average response time SLA (s)")
    model.alpha_cpu = pyo.Param(
        initialize=alpha_cpu, doc="Throughput per CPU millicore (req/s/mCPU)"
    )
    model.ram_base = pyo.Param(
        initialize=ram_base, doc="Base RAM overhead per replica (MB)"
    )
    model.beta_ram = pyo.Param(
        initialize=beta_ram, doc="RAM required per unit of capacity (MB/(req/s))"
    )
    model.w_cpu = pyo.Param(initialize=w_cpu, doc="Weight for normalized CPU in Obj 1")
    model.w_ram = pyo.Param(initialize=w_ram, doc="Weight for normalized RAM in Obj 1")
    model.w_resources = pyo.Param(
        initialize=w_resources, doc="Weight for Objective 1 (Resources)"
    )
    model.w_replicas = pyo.Param(
        initialize=w_replicas, doc="Weight for Objective 2 (Replicas)"
    )

    # =========================================================================
    # 2. Decision Variables
    # =========================================================================
    model.cpu = pyo.Var(
        bounds=cpu_bounds,
        initialize=cpu_bounds[0],
        doc="Allocated CPU per container replica (millicores)",
    )
    model.ram = pyo.Var(
        bounds=ram_bounds,
        initialize=ram_bounds[0],
        doc="Allocated RAM per container replica (MB)",
    )
    model.replicas = pyo.Var(
        bounds=(n_min, n_max),
        within=pyo.Integers,
        initialize=fixed_replicas if fixed_replicas is not None else n_min,
        doc="Number of active container replicas",
    )
    if fixed_replicas is not None:
        model.replicas.fix(int(fixed_replicas))
    model.mu = pyo.Var(
        within=pyo.NonNegativeReals,
        initialize=alpha_cpu * cpu_bounds[0],
        doc="Service capacity per container replica (req/s)",
    )
    model.lambda_per_replica = pyo.Var(
        within=pyo.NonNegativeReals,
        initialize=lambda_req / n_min,
        doc="Arrival rate routed to each replica (req/s)",
    )

    # =========================================================================
    # 3. Constraints
    # =========================================================================
    # (a) Per-container service capacity proportional to CPU
    @model.Constraint(doc="Service rate per replica proportional to allocated CPU")
    def service_capacity_rule(m):
        return m.mu == m.alpha_cpu * m.cpu

    # (b) Per-container memory requirement
    @model.Constraint(doc="Minimum RAM per replica to support service capacity")
    def ram_requirement_rule(m):
        return m.ram >= m.ram_base + m.beta_ram * m.mu

    # (c) Load distribution (lambda_per_replica = lambda_req / replicas) and M/M/1 SLA
    if nonlinear:
        @model.Constraint(doc="Nonlinear load balancing across replicas")
        def load_balance_nonlinear_rule(m):
            return m.lambda_per_replica * m.replicas == m.lambda_req

        @model.Constraint(doc="M/M/1 SLA constraint per replica")
        def mm1_sla_nonlinear_rule(m):
            return m.mu - m.lambda_per_replica >= 1.0 / m.w_max
    else:
        # Exact MILP formulation using unary/binary selector z[n] in {0, 1}
        model.z = pyo.Var(
            model.N_SET,
            within=pyo.Binary,
            initialize=lambda m, n: 1 if n == n_min else 0,
            doc="Binary indicator: z[n] = 1 iff replicas == n",
        )

        @model.Constraint(doc="Exactly one replica count is selected")
        def select_one_replica_count_rule(m):
            return sum(m.z[n] for n in m.N_SET) == 1

        @model.Constraint(doc="Link integer replicas variable to binary selectors")
        def replicas_link_rule(m):
            return m.replicas == sum(n * m.z[n] for n in m.N_SET)

        @model.Constraint(
            doc="Exact linear representation of lambda_per_replica = lambda_req / replicas"
        )
        def load_balance_linear_rule(m):
            return m.lambda_per_replica == sum(
                (m.lambda_req / float(n)) * m.z[n] for n in m.N_SET
            )

        @model.Constraint(
            doc="Linearized M/M/1 SLA and stability constraint per replica: mu - lambda/N >= 1 / w_max"
        )
        def mm1_sla_linear_rule(m):
            return m.mu - m.lambda_per_replica >= 1.0 / m.w_max

    # =========================================================================
    # 4. Queueing & Cluster Expressions (for reporting and analysis)
    # =========================================================================
    cpu_max = cpu_bounds[1]
    ram_max = ram_bounds[1]

    @model.Expression(doc="Objective 1: Normalized per-container resource allocation (CPU + RAM)")
    def obj_resources(m):
        return w_cpu * (m.cpu / cpu_max) + w_ram * (m.ram / ram_max)

    @model.Expression(doc="Objective 2: Normalized number of container replicas")
    def obj_replicas(m):
        return m.replicas / float(n_max)

    @model.Expression(doc="Per-replica utilization (rho = lambda_per_replica / mu)")
    def utilization(m):
        return m.lambda_per_replica / m.mu

    @model.Expression(doc="Mean response time in system W (s)")
    def response_time(m):
        return 1.0 / (m.mu - m.lambda_per_replica)

    @model.Expression(doc="Mean waiting time in queue Wq (s)")
    def waiting_time_queue(m):
        return m.lambda_per_replica / (m.mu * (m.mu - m.lambda_per_replica))

    @model.Expression(doc="Mean requests in each replica L_rep")
    def avg_requests_per_replica(m):
        return m.lambda_per_replica / (m.mu - m.lambda_per_replica)

    @model.Expression(doc="Total mean requests across all replicas in cluster L_total")
    def avg_requests_in_cluster(m):
        return m.replicas * (m.lambda_per_replica / (m.mu - m.lambda_per_replica))

    @model.Expression(doc="Total cluster CPU allocated across all replicas (millicores)")
    def total_cluster_cpu(m):
        return m.replicas * m.cpu

    @model.Expression(doc="Total cluster RAM allocated across all replicas (MB)")
    def total_cluster_ram(m):
        return m.replicas * m.ram

    # =========================================================================
    # 5. Multi-Objective Function (Weighted Sum of Resources and Replicas)
    # =========================================================================
    @model.Objective(
        sense=pyo.minimize,
        doc="Minimize weighted sum of Objective 1 (Resources) and Objective 2 (Replicas)",
    )
    def total_multi_objective_cost(m):
        return w_resources * m.obj_resources + w_replicas * m.obj_replicas

    return model


def solve_replicas_model(
    model: pyo.ConcreteModel, solver_name: str | None = None
) -> pyo.SolverResults:
    """
    Solves the Case 2 MILP/MINLP Pyomo model using an available solver.
    """
    candidate_solvers = (
        [solver_name]
        if solver_name
        else ["glpk", "highs", "appsi_highs", "cbc", "ipopt"]
    )

    for name in candidate_solvers:
        solver = pyo.SolverFactory(name)
        if solver.available(exception_flag=False):
            results = solver.solve(model, tee=False)
            model._solver_name = name
            model._solver_results = results
            return results

    raise RuntimeError(
        f"No compatible MILP solver found among {candidate_solvers}. "
        "Install 'glpk', 'highs' (highspy), or 'cbc'."
    )


def print_replicas_solution(
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

    lam_total = pyo.value(model.lambda_req)
    w_max_val = pyo.value(model.w_max)
    alpha_val = pyo.value(model.alpha_cpu)
    ram_base_val = pyo.value(model.ram_base)
    beta_ram_val = pyo.value(model.beta_ram)
    w_cpu_val = pyo.value(model.w_cpu)
    w_ram_val = pyo.value(model.w_ram)
    w_res_val = pyo.value(model.w_resources)
    w_rep_val = pyo.value(model.w_replicas)

    cpu_lb, cpu_ub = model.cpu.lb, model.cpu.ub
    ram_lb, ram_ub = model.ram.lb, model.ram.ub
    rep_lb, rep_ub = model.replicas.lb, model.replicas.ub

    # 1. Problem Title
    print("=" * 72)
    print("  CASE 2: MULTI-OBJECTIVE CONTAINER & REPLICA OPTIMIZATION PROBLEM")
    print("=" * 72)

    # 2. Description of Parameters Used
    print("Workload, SLA & System Parameters:")
    print(f"  Total Arrival Rate (lambda)       : {lam_total:.2f} req/s")
    print(f"  Max Response Time SLA (W_max)     : {w_max_val * 1000:.2f} ms ({w_max_val:.4f} s)")
    print(f"  CPU Throughput (alpha_cpu)        : {alpha_val:.4f} req/s per millicore")
    print(f"  Base RAM Overhead (ram_base)      : {ram_base_val:.2f} MB per container")
    print(f"  Working RAM Factor (beta_ram)     : {beta_ram_val:.2f} MB per (req/s)")
    print(f"  Allowed Replicas Bounds (N)       : [{rep_lb:.0f}, {rep_ub:.0f}] containers")
    print(f"  Per-Container CPU Bounds          : [{cpu_lb:.0f}, {cpu_ub:.0f}] millicores")
    print(f"  Per-Container RAM Bounds          : [{ram_lb:.0f}, {ram_ub:.0f}] MB")
    print(
        f"  Multi-Objective Weights           : w_resources={w_res_val:.2f} "
        f"(w_cpu={w_cpu_val:.1f}, w_ram={w_ram_val:.1f}), w_replicas={w_rep_val:.2f}"
    )
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
    n_val = int(round(pyo.value(model.replicas)))
    cpu_val = pyo.value(model.cpu)
    ram_val = pyo.value(model.ram)
    mu_val = pyo.value(model.mu)
    lam_rep = pyo.value(model.lambda_per_replica)

    rho_val = pyo.value(model.utilization)
    w_val = pyo.value(model.response_time)
    wq_val = pyo.value(model.waiting_time_queue)
    l_rep = pyo.value(model.avg_requests_per_replica)
    l_tot = pyo.value(model.avg_requests_in_cluster)
    tot_cpu = pyo.value(model.total_cluster_cpu)
    tot_ram = pyo.value(model.total_cluster_ram)

    print("Optimal Scaling & Load Balancing Decision (Horizontal + Vertical):")
    print(f"  Active Replicas (N) [Obj 2]       : {n_val} containers (bounds: [{rep_lb:.0f}, {rep_ub:.0f}])")
    print(
        f"  Arrival Rate per Replica (lam/N)  : {lam_total:.2f} / {n_val} = {lam_rep:.2f} req/s"
    )
    print(f"  CPU per Container   [Obj 1]       : {cpu_val:.2f} millicores (bounds: [{cpu_lb:.0f}, {cpu_ub:.0f}])")
    print(f"  RAM per Container   [Obj 1]       : {ram_val:.2f} MB (bounds: [{ram_lb:.0f}, {ram_ub:.0f}])")
    print(f"  Service Rate per Replica (mu)     : {mu_val:.2f} req/s")
    print("-" * 72)
    print("Total Cluster Footprint (N * Per-Container):")
    print(f"  Total Cluster CPU                 : {tot_cpu:.2f} millicores ({tot_cpu / 1000:.2f} vCPUs)")
    print(f"  Total Cluster RAM                 : {tot_ram:.2f} MB")
    print(f"  Total Cluster Capacity (N*mu)     : {n_val * mu_val:.2f} req/s")
    print("-" * 72)
    print("Queueing Metrics (Steady State per Replica):")
    print(f"  Replica Utilization (rho)         : {rho_val * 100:.2f}%")
    print(f"  Avg Response Time (W)             : {w_val * 1000:.2f} ms")
    print(f"  Avg Queue Waiting Time (Wq)       : {wq_val * 1000:.2f} ms")
    print(f"  Avg Requests per Replica (L)      : {l_rep:.2f} req (Cluster Total: {l_tot:.2f} req)")
    print("-" * 72)
    print("Multi-Objective Breakdown:")
    print(f"  Objective 1 (Normalized Res.)     : {pyo.value(model.obj_resources):.4f}")
    print(f"  Objective 2 (Normalized Rep.)     : {pyo.value(model.obj_replicas):.4f}")
    print(f"  Combined Objective Value          : {pyo.value(model.total_multi_objective_cost):.4f}")
    print("=" * 72)


def print_workload_distribution_table(
    lambda_req: float = 120.0,
    w_max: float = 0.05,
    w_resources: float = 1.0,
    w_replicas: float = 1.0,
) -> None:
    """
    Solves the model for each possible replica count N in {1..5} to show how
    the workload (lambda / N) is distributed and how CPU/RAM respond.
    """
    print("\nWorkload Distribution & Resource Breakdown by Number of Replicas (N):")
    print("-" * 72)
    print(
        f"{'Replicas (N)':^12} | {'Arrival/Rep (lam/N)':^19} | {'CPU/Rep (mCPU)':^14} | "
        f"{'RAM/Rep (MB)':^12} | {'Combined Obj':^12}"
    )
    print("-" * 72)
    for n in range(1, 6):
        m = build_container_replicas_model(
            lambda_req=lambda_req,
            w_max=w_max,
            replica_bounds=(1, 5),
            fixed_replicas=n,
            w_resources=w_resources,
            w_replicas=w_replicas,
        )
        solve_replicas_model(m)
        lam_rep = pyo.value(m.lambda_per_replica)
        cpu_v = pyo.value(m.cpu)
        ram_v = pyo.value(m.ram)
        obj_v = pyo.value(m.total_multi_objective_cost)
        print(
            f"{n:^12d} | {f'{lambda_req:.0f}/{n} = {lam_rep:.2f} req/s':^19} | "
            f"{cpu_v:^14.2f} | {ram_v:^12.2f} | {obj_v:^12.4f}"
        )
    print("-" * 72)


def plot_and_save_replicas_model(
    lambda_req: float = 120.0,
    w_max: float = 0.05,
    assets_dir: str | None = None,
) -> None:
    """
    Generates and saves the discrete Pareto front and 3D multi-objective surface
    for Part 2 into the assets directory.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if assets_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = (
            os.path.join(base_dir, "..", "assets", "part2")
            if os.path.basename(base_dir) == "part2"
            else os.path.join(base_dir, "assets", "part2")
        )
    os.makedirs(assets_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    pareto_rows = []
    for n_fixed in range(1, 6):
        m = build_container_replicas_model(
            lambda_req=lambda_req,
            w_max=w_max,
            replica_bounds=(1, 5),
            fixed_replicas=n_fixed,
        )
        solve_replicas_model(m)
        pareto_rows.append({
            "Replicas (N)": n_fixed,
            "CPU/Container (mCPU)": pyo.value(m.cpu),
            "RAM/Container (MB)": pyo.value(m.ram),
            "Combined Obj": pyo.value(m.total_multi_objective_cost),
        })
    df_pareto = pd.DataFrame(pareto_rows)

    # 1. Pareto Front
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1 = axes[0]
    ax1.plot(df_pareto["Replicas (N)"], df_pareto["CPU/Container (mCPU)"], marker="o", markersize=9, lw=2.5, color="#1f77b4", label="Pareto-Optimal Solutions")
    for _, row in df_pareto.iterrows():
        ax1.annotate(f"N={int(row['Replicas (N)'])}: {row['CPU/Container (mCPU)']:.0f} mCPU",
                     xy=(row["Replicas (N)"], row["CPU/Container (mCPU)"]),
                     xytext=(row["Replicas (N)"] + 0.12, row["CPU/Container (mCPU)"] + 40),
                     fontsize=8.5, color="#084594")
    ax1.set_title("Discrete Pareto Frontier: Replicas vs. Per-Container CPU", fontweight="bold")
    ax1.set_xlabel("Objective 2: Active Replicas (N)", fontweight="bold")
    ax1.set_ylabel("Per-Container CPU Allocation (mCPU)", color="#1f77b4", fontweight="bold")
    ax1.set_xticks(range(1, 6))
    ax1.set_xlim(0.7, 5.8)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = axes[1]
    ax2.plot(df_pareto["Replicas (N)"], df_pareto["Combined Obj"], marker="s", markersize=9, lw=2.5, color="#d62728", label="Total Cost Function $F(CPU, RAM, N)$")
    opt_row = df_pareto.loc[df_pareto["Combined Obj"].idxmin()]
    ax2.scatter([opt_row["Replicas (N)"]], [opt_row["Combined Obj"]], color="#2ca02c", s=180, zorder=6, edgecolors="black",
                label=f"Global Optimum: N={int(opt_row['Replicas (N)'])} (Score: {opt_row['Combined Obj']:.4f})")
    ax2.set_title("Weighted-Sum Objective Value across Replicas", fontweight="bold")
    ax2.set_xlabel("Active Replicas (N)", fontweight="bold")
    ax2.set_ylabel("Combined Objective Score", color="#d62728", fontweight="bold")
    ax2.set_xticks(range(1, 6))
    ax2.set_xlim(0.7, 5.8)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    f1_path = os.path.join(assets_dir, "01_pareto_replicas_tradeoff.png")
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. 3D Surface
    cpu_vec = np.linspace(500, 2000, 300)
    ram_req_vec = 128.0 + 4.0 * (0.08 * cpu_vec)
    slice_colors = {1: "#d62728", 2: "#2ca02c", 3: "#1f77b4", 4: "#9467bd", 5: "#ff7f0e"}

    fig = plt.figure(figsize=(15.5, 6.4))
    ax1 = fig.add_subplot(1, 2, 1)
    for n in range(1, 6):
        lam_rep_n = lambda_req / n
        cpu_sla_n = (lam_rep_n + 1.0 / w_max) / 0.08
        ax1.plot(cpu_vec, ram_req_vec, color=slice_colors[n], lw=1.6, ls="--", alpha=0.55)
        ax1.fill_between(cpu_vec, ram_req_vec, 1200, where=(cpu_vec >= cpu_sla_n), color=slice_colors[n], alpha=0.10)
        ax1.axvline(cpu_sla_n, color=slice_colors[n], ls=":", lw=1.6)
        row_n = df_pareto.loc[df_pareto["Replicas (N)"] == n].iloc[0]
        marker = "*" if n == 2 else "o"
        size = 180 if n == 2 else 75
        ax1.scatter([row_n["CPU/Container (mCPU)"]], [row_n["RAM/Container (MB)"]],
                    color=slice_colors[n], s=size, marker=marker, zorder=6, edgecolors="black",
                    label=f"N={n}: {row_n['CPU/Container (mCPU)']:.0f} mCPU")
    ax1.set_xlim(500, 2000)
    ax1.set_ylim(128, 900)
    ax1.set_title("Discrete Slices: Feasible Resource Regions per Replica Count", fontweight="bold")
    ax1.set_xlabel("Per-Container CPU (millicores)")
    ax1.set_ylabel("Per-Container RAM (MB)")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=True)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for n in range(1, 6):
        lam_rep_n = lambda_req / n
        cpu_sla_n = (lam_rep_n + 1.0 / w_max) / 0.08
        c_line = np.linspace(cpu_sla_n, 2000, 80)
        r_line = 128.0 + 0.32 * c_line
        n_line = np.full_like(c_line, n)
        obj_line = (c_line / 2000.0) + (r_line / 4096.0) + (n / 5.0)
        ax2.plot(c_line, n_line, obj_line, color=slice_colors[n], lw=2.4)
        row_n = df_pareto.loc[df_pareto["Replicas (N)"] == n].iloc[0]
        opt_obj = row_n["Combined Obj"]
        marker = "*" if n == 2 else "o"
        size = 180 if n == 2 else 70
        ax2.scatter([row_n["CPU/Container (mCPU)"]], [n], [opt_obj], color=slice_colors[n], s=size, marker=marker, edgecolors="black", depthshade=False)
    ax2.set_title("3D Mixed-Integer Optimization Space: CPU x Replicas x Objective", fontweight="bold")
    ax2.set_xlabel("CPU (mCPU)")
    ax2.set_ylabel("Replicas (N)")
    ax2.set_zlabel("Combined Cost F")
    ax2.set_yticks(range(1, 6))
    ax2.view_init(elev=24, azim=-120)
    plt.tight_layout()
    f2_path = os.path.join(assets_dir, "04_3d_multiobjective_surface.png")
    plt.savefig(f2_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated and saved Part 2 plots to: {assets_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part 2: Multi-Objective Container & Replica Optimization (MILP)")
    parser.add_argument("--lambda_req", type=float, default=120.0, help="Total arrival rate in req/s (default: 120.0)")
    parser.add_argument("--w_max", type=float, default=0.05, help="Max mean response time SLA in seconds (default: 0.05)")
    parser.add_argument("--plot", action="store_true", help="Generate and save Pareto and 3D surface plots to assets/part2/")
    args = parser.parse_args()

    # Solve Case 2 for specified workload
    replicas_model = build_container_replicas_model(
        lambda_req=args.lambda_req,
        w_max=args.w_max,
        alpha_cpu=0.08,
        ram_base=128.0,
        beta_ram=4.0,
        cpu_bounds=(500.0, 2000.0),
        replica_bounds=(1, 5),
        w_resources=1.0,
        w_replicas=1.0,
    )

    solve_replicas_model(replicas_model)
    print_replicas_solution(replicas_model)
    print_workload_distribution_table(lambda_req=args.lambda_req, w_max=args.w_max)

    if args.plot:
        plot_and_save_replicas_model(lambda_req=args.lambda_req, w_max=args.w_max)
