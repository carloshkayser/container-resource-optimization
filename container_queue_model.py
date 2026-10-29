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
    print("\n" * 3)

    # 3. Solution Status Banner
    is_optimal = (
        res is not None
        and res.solver.status == pyo.SolverStatus.ok
        and res.solver.termination_condition == pyo.TerminationCondition.optimal
    )
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


if __name__ == "__main__":
    # Build and solve the model for a default workload scenario
    container_model = build_container_queue_model(
        lambda_req=50.0,  # 50 requests/second arriving at the container
        w_max=0.05,       # SLA: average response time <= 50 ms (0.05 s)
        alpha_cpu=0.08,   # Each millicore processes 0.08 req/s (1000 mCPU -> 80 req/s)
        ram_base=128.0,   # 128 MB base memory
        beta_ram=4.0,     # 4 MB per req/s of service capacity
        cpu_bounds=(500.0, 2000.0),
    )

    solver_results = solve_model(container_model)
    print_solution(container_model, solver_results)
