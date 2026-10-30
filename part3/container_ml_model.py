"""
Case 3: Data-Driven Container Resource Optimization using Machine Learning Surrogate Models.

In contrast to analytical Queueing Theory (M/M/1 in Case 1 & Case 2), Case 3 uses
empirical telemetry data from real container benchmarks (UPB Microservices / Nginx benchmark)
to learn a non-linear surrogate model of container response time:

    W_hat = f_ML(CPU, RAM, lambda_req, payload_type)

We train, cross-validate, and compare multiple ML models:
  1. Ridge Regression (Linear Baseline)
  2. Random Forest Regressor (Nonlinear Ensemble)
  3. Multi-Layer Perceptron / MLP (Deep/Shallow Neural Network with ReLU activations)

The trained surrogate model (e.g., MLP) is then embedded into the mathematical
optimization formulation to find the minimal CPU and RAM allocation that satisfies
the latency SLA under any target workload:

    min   w_cpu * (CPU / CPU_max) + w_ram * (RAM / RAM_max)
    s.t.  W_hat_ML(CPU, RAM, lambda_req, payload_type) <= W_max
          CPU_min <= CPU <= CPU_max
          RAM_min <= RAM <= RAM_max
"""

import argparse
import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "container_performance_web1.csv"
)
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FEATURE_COLS = ["cpu_mcpu", "ram_mb", "rps", "is_heavy_payload"]


# =============================================================================
# Artifact Persistence (Joblib)
# =============================================================================

def save_artifacts(
    models: Dict[str, Any],
    scaler: StandardScaler,
    metrics: Dict[str, Dict[str, float]],
    directory: str = DEFAULT_MODELS_DIR,
) -> None:
    """Saves trained models, scaler, and metrics dictionary to disk."""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(models, os.path.join(directory, "models.joblib"))
    joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))
    joblib.dump(metrics, os.path.join(directory, "metrics.joblib"))


def load_artifacts(
    directory: str = DEFAULT_MODELS_DIR,
) -> Tuple[Dict[str, Any], StandardScaler, Dict[str, Dict[str, float]]] | None:
    """Loads models, scaler, and metrics if they exist."""
    m_path = os.path.join(directory, "models.joblib")
    s_path = os.path.join(directory, "scaler.joblib")
    met_path = os.path.join(directory, "metrics.joblib")
    if os.path.exists(m_path) and os.path.exists(s_path):
        models = joblib.load(m_path)
        scaler = joblib.load(s_path)
        metrics = joblib.load(met_path) if os.path.exists(met_path) else {}
        return models, scaler, metrics
    return None


# =============================================================================
# 1. Dataset Loading and Preprocessing
# =============================================================================

def load_and_preprocess_dataset(
    csv_path: str = DEFAULT_DATASET_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Loads container performance telemetry data.

    Returns
    -------
    data : pd.DataFrame
        Cleaned dataframe with features and latency.
    X : pd.DataFrame
        Features matrix: ['cpu_mcpu', 'ram_mb', 'rps', 'is_heavy_payload'].
    y : pd.Series
        Target response times in milliseconds (ms).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Please run dataset download first."
        )

    data = pd.read_csv(csv_path)
    for col in FEATURE_COLS + ["latency_ms"]:
        if col not in data.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    # Filter out invalid entries
    clean_data = data[
        (data["rps"] > 0)
        & (data["latency_ms"] > 0)
        & (data["cpu_mcpu"] > 0)
        & (data["ram_mb"] > 0)
    ].copy()

    X = clean_data[FEATURE_COLS]
    y = clean_data["latency_ms"]
    return clean_data, X, y


# =============================================================================
# 2. Model Training and Benchmarking
# =============================================================================

def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    log_target: bool = True,
) -> Tuple[Dict[str, Any], StandardScaler, Dict[str, Dict[str, float]], Tuple]:
    """
    Trains Ridge, Random Forest, and MLP models on container telemetry.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix [cpu_mcpu, ram_mb, rps, is_heavy_payload].
    y : pd.Series
        Target latency in milliseconds (ms).
    test_size : float
        Fraction of data for test split.
    random_state : int
        Seed for reproducibility.
    log_target : bool
        Whether to train on log1p(latency) to handle heavy-tailed queue latency.

    Returns
    -------
    models : Dict[str, Any]
        Dictionary containing fitted models: 'ridge', 'random_forest', 'mlp'.
    scaler : StandardScaler
        Fitted scaler for input features.
    metrics : Dict[str, Dict[str, float]]
        Validation metrics (R2, MAE, RMSE) per model.
    split_data : Tuple
        (X_train, X_test, y_train, y_test, X_train_s, X_test_s).
    """
    y_target = np.log1p(y) if log_target else y

    X_train, X_test, y_train_t, y_test_t = train_test_split(
        X, y_target, test_size=test_size, random_state=random_state
    )
    y_test_orig = y.loc[y_test_t.index].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=random_state,
        ),
    }

    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        model.fit(X_train_s, y_train_t)
        preds_t = model.predict(X_test_s)
        preds_orig = np.expm1(preds_t) if log_target else preds_t

        preds_orig = np.clip(preds_orig, a_min=0.01, a_max=None)

        r2 = r2_score(y_test_orig, preds_orig)
        mae = mean_absolute_error(y_test_orig, preds_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, preds_orig))

        metrics[name] = {
            "r2": float(r2),
            "mae_ms": float(mae),
            "rmse_ms": float(rmse),
        }

    split_data = (X_train, X_test, y_train_t, y_test_t, X_train_s, X_test_s)
    return models, scaler, metrics, split_data


# =============================================================================
# 3. Surrogate Prediction Function
# =============================================================================

def predict_latency(
    model: Any,
    scaler: StandardScaler,
    cpu_mcpu: float | np.ndarray,
    ram_mb: float | np.ndarray,
    rps: float | np.ndarray,
    is_heavy_payload: int | np.ndarray = 1,
    log_target: bool = True,
) -> float | np.ndarray:
    """
    Uses a trained ML surrogate to predict container response time (ms).
    """
    is_scalar = (
        np.isscalar(cpu_mcpu)
        and np.isscalar(ram_mb)
        and np.isscalar(rps)
        and np.isscalar(is_heavy_payload)
    )

    cpu_arr = np.atleast_1d(cpu_mcpu).astype(float)
    ram_arr = np.atleast_1d(ram_mb).astype(float)
    rps_arr = np.atleast_1d(rps).astype(float)
    hvy_arr = np.atleast_1d(is_heavy_payload).astype(float)

    if not (len(cpu_arr) == len(ram_arr) == len(rps_arr) == len(hvy_arr)):
        cpu_arr, ram_arr, rps_arr, hvy_arr = np.broadcast_arrays(
            cpu_arr, ram_arr, rps_arr, hvy_arr
        )

    feats_df = pd.DataFrame(
        {
            "cpu_mcpu": cpu_arr,
            "ram_mb": ram_arr,
            "rps": rps_arr,
            "is_heavy_payload": hvy_arr,
        }
    )
    feats_s = scaler.transform(feats_df)
    feats_s_df = pd.DataFrame(feats_s, columns=FEATURE_COLS)

    preds = model.predict(feats_s_df)
    if log_target:
        preds = np.expm1(preds)

    preds = np.clip(preds, a_min=0.01, a_max=None)
    return float(preds[0]) if is_scalar else preds


# =============================================================================
# 4. Data-Driven Optimization (ML Surrogate)
# =============================================================================

def optimize_container_resources_ml(
    lambda_req: float,
    w_max_ms: float,
    model: Any,
    scaler: StandardScaler,
    is_heavy_payload: int = 1,
    cpu_bounds: Tuple[float, float] = (100.0, 1000.0),
    ram_bounds: Tuple[float, float] = (64.0, 512.0),
    w_cpu: float = 1.0,
    w_ram: float = 1.0,
    log_target: bool = True,
    method: str = "adaptive_grid",
) -> Dict[str, Any]:
    """
    Solves the data-driven container sizing optimization problem:

        min   w_cpu * (CPU / CPU_max) + w_ram * (RAM / RAM_max)
        s.t.  W_hat_ML(CPU, RAM, lambda_req, payload) <= w_max_ms
              CPU in [CPU_min, CPU_max]
              RAM in [RAM_min, RAM_max]

    Parameters
    ----------
    lambda_req : float
        Arrival rate in requests/second (RPS).
    w_max_ms : float
        Maximum allowed latency SLA in milliseconds (ms).
    model : Any
        Trained ML model (e.g. MLPRegressor or RandomForestRegressor).
    scaler : StandardScaler
        Fitted feature scaler.
    is_heavy_payload : int
        0 for lightweight web requests, 1 for CPU-intensive payload.
    cpu_bounds : Tuple[float, float]
        Allowed range for CPU in millicores (100 to 1000 mCPU).
    ram_bounds : Tuple[float, float]
        Allowed range for RAM in MB (64 to 512 MB).
    w_cpu, w_ram : float
        Objective function weights.
    log_target : bool
        Whether model was trained with log1p target.
    method : str
        'adaptive_grid' (guarantees global optimum) or 'slsqp'.
    """
    cpu_min, cpu_max = cpu_bounds
    ram_min, ram_max = ram_bounds

    def objective(cpu: float, ram: float) -> float:
        return w_cpu * (cpu / cpu_max) + w_ram * (ram / ram_max)

    if method == "adaptive_grid":
        cpu_grid = np.linspace(cpu_min, cpu_max, 91)   # step 10 mCPU
        ram_grid = np.linspace(ram_min, ram_max, 57)   # step 8 MB
        cpu_mesh, ram_mesh = np.meshgrid(cpu_grid, ram_grid)

        c_flat = cpu_mesh.ravel()
        r_flat = ram_mesh.ravel()
        rps_flat = np.full_like(c_flat, fill_value=lambda_req)
        hvy_flat = np.full_like(c_flat, fill_value=is_heavy_payload)

        pred_latencies = predict_latency(
            model=model,
            scaler=scaler,
            cpu_mcpu=c_flat,
            ram_mb=r_flat,
            rps=rps_flat,
            is_heavy_payload=hvy_flat,
            log_target=log_target,
        )

        feasible_mask = pred_latencies <= w_max_ms

        if not np.any(feasible_mask):
            min_idx = np.argmin(pred_latencies)
            return {
                "feasible": False,
                "status": "Infeasible (SLA cannot be met within container hardware limits)",
                "cpu_optimal": float(c_flat[min_idx]),
                "ram_optimal": float(r_flat[min_idx]),
                "predicted_latency_ms": float(pred_latencies[min_idx]),
                "w_max_ms": w_max_ms,
                "lambda_req": lambda_req,
                "is_heavy_payload": is_heavy_payload,
                "cpu_bounds": cpu_bounds,
                "ram_bounds": ram_bounds,
                "w_cpu": w_cpu,
                "w_ram": w_ram,
                "method": method,
                "objective_value": None,
            }

        feasible_cpu = c_flat[feasible_mask]
        feasible_ram = r_flat[feasible_mask]
        feasible_lat = pred_latencies[feasible_mask]
        costs = objective(feasible_cpu, feasible_ram)

        best_idx = np.argmin(costs)
        opt_cpu = float(feasible_cpu[best_idx])
        opt_ram = float(feasible_ram[best_idx])
        opt_lat = float(feasible_lat[best_idx])
        opt_cost = float(costs[best_idx])

        return {
            "feasible": True,
            "status": "Optimal",
            "cpu_optimal": round(opt_cpu, 1),
            "ram_optimal": round(opt_ram, 1),
            "predicted_latency_ms": round(opt_lat, 3),
            "w_max_ms": w_max_ms,
            "lambda_req": lambda_req,
            "is_heavy_payload": is_heavy_payload,
            "cpu_bounds": cpu_bounds,
            "ram_bounds": ram_bounds,
            "w_cpu": w_cpu,
            "w_ram": w_ram,
            "method": method,
            "objective_value": round(opt_cost, 4),
        }

    else:
        def opt_func(x):
            return objective(x[0], x[1])

        def sla_constraint(x):
            lat = predict_latency(
                model, scaler, x[0], x[1], lambda_req, is_heavy_payload, log_target
            )
            return w_max_ms - lat

        starts = [
            [(cpu_min + cpu_max) / 2, (ram_min + ram_max) / 2],
            [cpu_max, ram_max],
            [cpu_min, ram_min],
        ]

        best_res = None
        for x0 in starts:
            res = minimize(
                opt_func,
                x0=x0,
                bounds=[(cpu_min, cpu_max), (ram_min, ram_max)],
                constraints={"type": "ineq", "fun": sla_constraint},
                method="SLSQP",
            )
            if res.success and (best_res is None or res.fun < best_res.fun):
                best_res = res

        if best_res is not None and best_res.success:
            opt_cpu, opt_ram = best_res.x
            opt_lat = predict_latency(
                model, scaler, opt_cpu, opt_ram, lambda_req, is_heavy_payload, log_target
            )
            return {
                "feasible": True,
                "status": "Optimal",
                "cpu_optimal": round(float(opt_cpu), 1),
                "ram_optimal": round(float(opt_ram), 1),
                "predicted_latency_ms": round(float(opt_lat), 3),
                "w_max_ms": w_max_ms,
                "lambda_req": lambda_req,
                "is_heavy_payload": is_heavy_payload,
                "cpu_bounds": cpu_bounds,
                "ram_bounds": ram_bounds,
                "w_cpu": w_cpu,
                "w_ram": w_ram,
                "method": method,
                "objective_value": round(float(best_res.fun), 4),
            }
        else:
            return {
                "feasible": False,
                "status": "Infeasible",
                "cpu_optimal": None,
                "ram_optimal": None,
                "predicted_latency_ms": None,
                "w_max_ms": w_max_ms,
                "lambda_req": lambda_req,
                "is_heavy_payload": is_heavy_payload,
                "cpu_bounds": cpu_bounds,
                "ram_bounds": ram_bounds,
                "w_cpu": w_cpu,
                "w_ram": w_ram,
                "method": method,
                "objective_value": None,
            }


# =============================================================================
# 5. Queueing Theory (M/M/1) vs ML Comparison
# =============================================================================

def compare_ml_vs_queue_model(
    lambda_req: float,
    w_max_ms: float,
    ml_model: Any,
    scaler: StandardScaler,
    is_heavy_payload: int = 1,
    alpha_cpu: float = 1.0,   # 1 req/s per mCPU for heavy payload
    ram_base: float = 64.0,
    beta_ram: float = 0.05,
    cpu_bounds: Tuple[float, float] = (100.0, 1000.0),
    ram_bounds: Tuple[float, float] = (64.0, 512.0),
) -> Dict[str, Any]:
    """
    Compares the resource allocation recommendations between:
      1. Case 1: Analytical M/M/1 Queueing Theory
      2. Case 3: Data-Driven Machine Learning Surrogate (e.g. MLP)
    """
    w_max_s = w_max_ms / 1000.0

    # 1. Analytical M/M/1 sizing
    mu_min = lambda_req + (1.0 / w_max_s)
    cpu_mm1 = mu_min / alpha_cpu
    ram_mm1 = ram_base + beta_ram * mu_min

    cpu_min, cpu_max = cpu_bounds
    ram_min, ram_max = ram_bounds

    mm1_feasible = (cpu_mm1 <= cpu_max) and (ram_mm1 <= ram_max)
    cpu_mm1_clamped = min(max(cpu_mm1, cpu_min), cpu_max)
    ram_mm1_clamped = min(max(ram_mm1, ram_min), ram_max)

    # 2. ML Surrogate sizing
    ml_result = optimize_container_resources_ml(
        lambda_req=lambda_req,
        w_max_ms=w_max_ms,
        model=ml_model,
        scaler=scaler,
        is_heavy_payload=is_heavy_payload,
        cpu_bounds=cpu_bounds,
        ram_bounds=ram_bounds,
    )

    # Cross-evaluate: What latency does the ML surrogate predict for the M/M/1 allocation?
    mm1_pred_latency_ms = predict_latency(
        model=ml_model,
        scaler=scaler,
        cpu_mcpu=cpu_mm1_clamped,
        ram_mb=ram_mm1_clamped,
        rps=lambda_req,
        is_heavy_payload=is_heavy_payload,
    )

    return {
        "lambda_req": lambda_req,
        "w_max_ms": w_max_ms,
        "is_heavy_payload": is_heavy_payload,
        "mm1_solution": {
            "feasible": mm1_feasible,
            "cpu_mcpu": round(float(cpu_mm1_clamped), 1),
            "ram_mb": round(float(ram_mm1_clamped), 1),
            "theoretical_latency_ms": round(w_max_ms, 2),
            "actual_ml_predicted_latency_ms": round(float(mm1_pred_latency_ms), 2),
            "sla_satisfied_in_reality": bool(mm1_pred_latency_ms <= w_max_ms),
        },
        "ml_solution": ml_result,
    }


# =============================================================================
# 6. CLI Runner
# =============================================================================

def print_model_report(metrics: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 72)
    print("  CASE 3: CONTAINER LATENCY ML BENCHMARK EVALUATION")
    print("=" * 72)
    print(f"{'Model':<22} | {'R² Score':<12} | {'MAE (ms)':<14} | {'RMSE (ms)':<12}")
    print("-" * 72)
    for name, m in metrics.items():
        print(
            f"{name:<22} | {m['r2']:<12.4f} | {m['mae_ms']:<14.2f} | {m['rmse_ms']:<12.2f}"
        )
    print("=" * 72)


def print_optimization_report(result: Dict[str, Any], model_name: str = "MLP") -> None:
    """
    Prints:
    1. Problem Title
    2. Description of the Parameters Used
    3. Solver Status / Whether an Optimal Solution was Found
    4. Details of the Solution Found
    """
    cpu_lb, cpu_ub = result.get("cpu_bounds", (100.0, 1000.0))
    ram_lb, ram_ub = result.get("ram_bounds", (64.0, 512.0))
    w_cpu = result.get("w_cpu", 1.0)
    w_ram = result.get("w_ram", 1.0)
    method = result.get("method", "adaptive_grid")
    payload_desc = (
        "Heavy (CPU/IO bound, payload=1)"
        if result["is_heavy_payload"]
        else "Lightweight Web (payload=0)"
    )

    # 1. Problem Title
    print("\n" + "=" * 72)
    print(f"  CASE 3: DATA-DRIVEN CONTAINER RESOURCE OPTIMIZATION (Surrogate: {model_name})")
    print("=" * 72)

    # 2. Description of Parameters Used
    print("Workload, SLA & Surrogate Optimization Parameters:")
    print(f"  Workload (lambda)             : {result['lambda_req']:.2f} requests/sec")
    print(f"  Payload Type                  : {payload_desc}")
    print(f"  Target Latency SLA (W_max)    : {result['w_max_ms']:.2f} ms ({result['w_max_ms'] / 1000.0:.4f} s)")
    print(f"  ML Surrogate Model            : {model_name}")
    print(f"  Optimization Search Method    : {method}")
    print(f"  CPU Allocation Bounds         : [{cpu_lb:.0f}, {cpu_ub:.0f}] millicores")
    print(f"  RAM Allocation Bounds         : [{ram_lb:.0f}, {ram_ub:.0f}] MB")
    print(f"  Objective Weights (CPU, RAM)  : w_cpu={w_cpu:.2f}, w_ram={w_ram:.2f}")
    print(f"  Feasibility Status            : {result['status']}")
    print("=" * 72)

    # 3. Solution Status Banner
    print("\n")
    print("*" * 72)
    if result["feasible"]:
        print("  OPTIMAL SOLUTION FOUND")
    else:
        print(f"  SOLUTION STATUS: {str(result['status']).upper()}")
    print("*" * 72)

    # 4. Details of the Solution Found
    if result["feasible"]:
        cpu_opt = result["cpu_optimal"]
        ram_opt = result["ram_optimal"]
        pred_lat = result["predicted_latency_ms"]
        w_max = result["w_max_ms"]
        obj_cpu = w_cpu * (cpu_opt / cpu_ub)
        obj_ram = w_ram * (ram_opt / ram_ub)

        print("Optimal Resource Allocation:")
        print(f"  Optimal CPU Allocation        : {cpu_opt:.1f} mCPU (bounds: [{cpu_lb:.0f}, {cpu_ub:.0f}])")
        print(f"  Optimal RAM Allocation        : {ram_opt:.1f} MB (bounds: [{ram_lb:.0f}, {ram_ub:.0f}])")
        print("-" * 72)
        print("ML Surrogate Performance & Objective Breakdown:")
        print(f"  Predicted Latency (W_hat_ML)  : {pred_lat:.3f} ms (SLA <= {w_max:.1f} ms)")
        print(f"  Latency SLA Headroom          : {w_max - pred_lat:.3f} ms")
        print(f"  CPU Cost Component            : {obj_cpu:.4f}")
        print(f"  RAM Cost Component            : {obj_ram:.4f}")
        print(f"  Normalized Cost Obj           : {result['objective_value']:.4f}")
    else:
        print("SLA could not be met within the single container hardware bounds.")
        if result.get("cpu_optimal") is not None:
            print(f"  Best Achievable Latency       : {result['predicted_latency_ms']:.3f} ms "
                  f"at ({result['cpu_optimal']:.1f} mCPU, {result['ram_optimal']:.1f} MB)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Case 3: Data-Driven Container Resource Optimization using Machine Learning"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Path to container benchmark CSV",
    )
    parser.add_argument(
        "--lambda_req",
        type=float,
        default=1.0,
        help="Target request arrival rate (RPS)",
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=500.0,
        help="Latency SLA threshold in milliseconds (ms)",
    )
    parser.add_argument(
        "--payload",
        type=int,
        default=1,
        choices=[0, 1],
        help="Payload type: 0 = lightweight web, 1 = heavy payload",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "random_forest", "ridge"],
        help="Surrogate model to use for optimization",
    )
    parser.add_argument(
        "--compare_mm1",
        action="store_true",
        help="Compare ML sizing against M/M/1 Queueing Theory",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining of models instead of loading saved artifacts",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save ML response surface and comparison plots to assets/part3/",
    )
    args = parser.parse_args()

    artifacts = None if args.retrain else load_artifacts()
    if artifacts is not None:
        models, scaler, metrics = artifacts
        print(f"Loaded pre-trained models from: {DEFAULT_MODELS_DIR}")
        print_model_report(metrics)
    else:
        print(f"Loading benchmark dataset from: {args.dataset}")
        _, X, y = load_and_preprocess_dataset(args.dataset)
        print(f"Loaded {len(X)} samples. Training candidate ML models...")
        models, scaler, metrics, _ = train_and_evaluate_models(X, y)
        save_artifacts(models, scaler, metrics)
        print(f"Saved trained models to: {DEFAULT_MODELS_DIR}")
        print_model_report(metrics)

    selected_model = models[args.model]
    print(
        f"\nSolving resource optimization for lambda = {args.lambda_req} RPS, "
        f"payload = {'heavy' if args.payload else 'light'}, SLA = {args.w_max} ms..."
    )
    opt_result = optimize_container_resources_ml(
        lambda_req=args.lambda_req,
        w_max_ms=args.w_max,
        model=selected_model,
        scaler=scaler,
        is_heavy_payload=args.payload,
    )
    print_optimization_report(opt_result, model_name=args.model.upper())

    if args.compare_mm1:
        comp = compare_ml_vs_queue_model(
            lambda_req=args.lambda_req,
            w_max_ms=args.w_max,
            ml_model=selected_model,
            scaler=scaler,
            is_heavy_payload=args.payload,
        )
        print("\n" + "=" * 65)
        print("   COMPARISON: CASE 1 (M/M/1 Queue) vs CASE 3 (ML Surrogate)")
        print("=" * 65)
        print(f"{'Metric':<30} | {'M/M/1 Queueing':<15} | {'ML Surrogate (MLP)':<15}")
        print("-" * 65)
        print(
            f"{'Allocated CPU (mCPU)':<30} | {comp['mm1_solution']['cpu_mcpu']:<15.1f} | {comp['ml_solution']['cpu_optimal']:<15.1f}"
        )
        print(
            f"{'Allocated RAM (MB)':<30} | {comp['mm1_solution']['ram_mb']:<15.1f} | {comp['ml_solution']['ram_optimal']:<15.1f}"
        )
        print(
            f"{'Predicted Latency (ms)':<30} | {comp['mm1_solution']['actual_ml_predicted_latency_ms']:<15.2f} | {comp['ml_solution']['predicted_latency_ms']:<15.2f}"
        )
        print(
            f"{'SLA Met in Practice?':<30} | {str(comp['mm1_solution']['sla_satisfied_in_reality']):<15} | {str(comp['ml_solution']['feasible']):<15}"
        )
        print("=" * 65)

    if args.plot:
        plot_and_save_ml_models(
            model=selected_model,
            scaler=scaler,
            opt_result=opt_result,
            comp_result=comp if args.compare_mm1 else None,
        )


def plot_and_save_ml_models(
    model: Any,
    scaler: StandardScaler,
    opt_result: Dict[str, Any],
    comp_result: Dict[str, Any] | None = None,
    assets_dir: str | None = None,
) -> None:
    """
    Generates and saves the 2D latency contour surface, optimal sizing point,
    and empirical comparison bar chart for Part 3 into the assets directory.
    """
    import matplotlib.pyplot as plt

    if assets_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = (
            os.path.join(base_dir, "..", "assets", "part3")
            if os.path.basename(base_dir) == "part3"
            else os.path.join(base_dir, "assets", "part3")
        )
    os.makedirs(assets_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    cpu_vals = np.linspace(100, 1000, 60)
    ram_vals = np.linspace(64, 512, 60)
    CPU_grid, RAM_grid = np.meshgrid(cpu_vals, ram_vals)
    pred_latencies = predict_latency(
        model=model,
        scaler=scaler,
        cpu_mcpu=CPU_grid.ravel(),
        ram_mb=RAM_grid.ravel(),
        rps=np.full(CPU_grid.size, opt_result.get("lambda_req", 1.0)),
        is_heavy_payload=np.full(CPU_grid.size, opt_result.get("is_heavy_payload", 1)),
    ).reshape(CPU_grid.shape)

    # 1. 2D Response Surface
    plt.figure(figsize=(10, 7))
    contour = plt.contourf(CPU_grid, RAM_grid, pred_latencies, levels=35, cmap="viridis_r")
    cbar = plt.colorbar(contour)
    cbar.set_label("Predicted Latency (ms)", fontweight="bold")
    sla_lines = plt.contour(CPU_grid, RAM_grid, pred_latencies, levels=[300.0, 500.0, 1000.0], colors=["white", "red", "yellow"], linewidths=2)
    plt.clabel(sla_lines, inline=True, fmt={300.0: "SLA: 300ms", 500.0: "SLA: 500ms", 1000.0: "SLA: 1000ms"}, fontsize=11)
    plt.title("MLP Latency Response Surface", fontweight="bold")
    plt.xlabel("Allocated CPU (mCPU)", fontweight="bold")
    plt.ylabel("Allocated RAM (MB)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "07_mlp_latency_contour_surface.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Optimal Sizing Contour
    plt.figure(figsize=(10, 6))
    contour = plt.contourf(CPU_grid, RAM_grid, pred_latencies, levels=35, cmap="viridis_r")
    cbar = plt.colorbar(contour)
    cbar.set_label("Predicted Latency (ms)", fontweight="bold")
    w_max = opt_result.get("w_max_ms", 500.0)
    plt.contour(CPU_grid, RAM_grid, pred_latencies, levels=[w_max], colors="red", linewidths=3, linestyles="--")
    plt.scatter([opt_result["cpu_optimal"]], [opt_result["ram_optimal"]], color="magenta", s=220, marker="*", edgecolors="black", zorder=10,
                label=f"Optimal Allocation: {opt_result['cpu_optimal']:.0f} mCPU, {opt_result['ram_optimal']:.0f} MB")
    plt.title(f"Optimal Container Sizing on MLP Response Surface (SLA <= {w_max:.0f}ms)", fontweight="bold")
    plt.xlabel("Allocated CPU (mCPU)", fontweight="bold")
    plt.ylabel("Allocated RAM (MB)", fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, "08_optimal_sizing_contour.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Empirical Showdown Bar Chart (if comparison was made)
    if comp_result is not None:
        comparison_table = []
        for w in [300.0, 400.0, 500.0, 700.0, 1000.0, 1200.0]:
            comp = compare_ml_vs_queue_model(
                lambda_req=opt_result.get("lambda_req", 1.0),
                w_max_ms=w,
                ml_model=model,
                scaler=scaler,
                is_heavy_payload=opt_result.get("is_heavy_payload", 1),
            )
            comparison_table.append({
                "SLA (ms)": w,
                "MM1 CPU (mCPU)": comp["mm1_solution"]["cpu_mcpu"],
                "MLP CPU (mCPU)": comp["ml_solution"]["cpu_optimal"],
                "MM1 Meets SLA?": comp["mm1_solution"]["sla_satisfied_in_reality"],
            })
        comp_df = pd.DataFrame(comparison_table)

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(comp_df))
        width = 0.35
        ax.bar(x - width/2, comp_df["MM1 CPU (mCPU)"], width, label="Case 1: M/M/1 Theory", color="coral", edgecolor="black")
        ax.bar(x + width/2, comp_df["MLP CPU (mCPU)"], width, label="Case 3: MLP Surrogate", color="royalblue", edgecolor="black")
        ax.set_xlabel("Latency SLA Threshold W_max (ms)", fontweight="bold")
        ax.set_ylabel("Allocated CPU (mCPU)", fontweight="bold")
        ax.set_title("Container CPU Sizing: M/M/1 Theory Under-Provisions vs. ML Surrogate", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w:.0f} ms" for w in comp_df["SLA (ms)"]])
        ax.legend(loc="upper right")
        for i in range(len(comp_df)):
            violated = not comp_df["MM1 Meets SLA?"].iloc[i]
            status = "SLA Violated" if violated else "OK"
            color = "red" if violated else "green"
            ax.text(x[i] - width/2, comp_df["MM1 CPU (mCPU)"].iloc[i] + 15, status, ha="center", fontsize=9, color=color, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(assets_dir, "10_mm1_vs_mlp_empirical_showdown.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Generated and saved Part 3 plots to: {assets_dir}")


if __name__ == "__main__":
    main()
