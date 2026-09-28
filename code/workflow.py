"""Shared solver settings and reproducible JSON output for the entry scripts."""

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def configure_solver(model, time_limit, epsilon, seed=1, threads=1, quiet=False):
    """Use a feasibility tolerance strictly below the branch separation margin."""
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive.")
    if not 1e-8 <= epsilon <= 1:
        raise ValueError("epsilon must be between 1e-8 and 1.")
    if threads < 1:
        raise ValueError("threads must be positive.")
    if not 0 <= seed <= 2_000_000_000:
        raise ValueError("seed must be between 0 and 2000000000.")
    model.Params.OutputFlag = int(not quiet)
    model.Params.TimeLimit = time_limit
    model.Params.FeasibilityTol = min(1e-6, epsilon / 10)
    model.Params.IntFeasTol = 1e-9
    model.Params.Seed = seed
    model.Params.Threads = threads


def solver_summary(model):
    """Report an incumbent separately from proof of optimality."""
    result = {
        "status": int(model.Status),
        "solution_count": int(model.SolCount),
        "runtime_seconds": float(model.Runtime),
        "optimal": model.Status == 2,
    }
    if model.SolCount:
        result.update(
            objective=float(model.ObjVal),
            objective_bound=float(model.ObjBound),
            mip_gap=float(model.MIPGap),
        )
    return result


def environment_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "sktime", "gurobipy", "matplotlib"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def write_json(path, data):
    """Write portable JSON (nonfinite solver values become null)."""

    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
