"""
Main entry point for the Container Resource Optimization Project.

Provides a unified CLI interface to run Case 1, Case 2, and Case 3.
"""

import sys


def print_help() -> None:
    print("""
========================================================================
Container Resource & Replica Optimization (Queueing Theory & ML)
========================================================================
Usage:
    uv run python3 main.py <case> [options]

Available Cases:
    case1     Run Case 1: Single Container Optimization (M/M/1 Queue)
    case2     Run Case 2: Multi-Objective Replicas Optimization (Parallel M/M/1)
    case3     Run Case 3: Data-Driven ML Surrogate Optimization (MLP)
    train     Retrain ML models from scratch on benchmark telemetry
    help      Show this help menu

Examples:
    uv run python3 main.py case1
    uv run python3 main.py case2
    uv run python3 main.py case3 --w_max 500 --compare_mm1
    uv run python3 main.py train
========================================================================
""")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        return

    cmd = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if cmd in ("case1", "1"):
        import container_queue_model
        sys.argv = ["container_queue_model.py"] + extra_args
        container_queue_model.main()
    elif cmd in ("case2", "2"):
        import container_replicas_queue_model
        sys.argv = ["container_replicas_queue_model.py"] + extra_args
        container_replicas_queue_model.main()
    elif cmd in ("case3", "3"):
        import container_ml_model
        sys.argv = ["container_ml_model.py"] + extra_args
        container_ml_model.main()
    elif cmd == "train":
        import container_ml_model
        sys.argv = ["container_ml_model.py", "--retrain"] + extra_args
        container_ml_model.main()
    else:
        print(f"Unknown command: '{cmd}'. Run 'uv run python3 main.py help' for usage.")


if __name__ == "__main__":
    main()
