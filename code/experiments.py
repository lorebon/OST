"""Shared synthetic-data experiments for the three original entry scripts.

The exemplar is known to the learner in these oracle experiments. It is drawn
from the full simulated population, unlike the training-only UCR workflow.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from classification import predictModel
from preprocessing import SeriesMinMaxScaler, positive_integer
from training import generateModel, retrieveSolution
from workflow import configure_solver, environment_versions, solver_summary, write_json

DEFAULT_GRIDS = {
    "depth": [1, 2, 3, 4],
    "length": [3, 6, 9, 12, 15, 18],
    "learning": [50, 100, 200, 400],
}


def brownian(x0, n, dt, delta, out=None, rng=None):
    """Generate Brownian paths; an explicit Generator makes workers reproducible."""
    rng = np.random.default_rng() if rng is None else rng
    x0 = np.asarray(x0)
    increments = rng.normal(scale=delta * np.sqrt(dt), size=x0.shape + (n,))
    if out is None:
        out = np.empty_like(increments)
    np.cumsum(increments, axis=-1, out=out)
    out += np.expand_dims(x0, axis=-1)
    return out


def assignLabel(distance, b, depth, index, node=0, sign=1, epsilon=0.0002):
    """Original binary parity labels: every right turn reverses the class sign.

    Preserve the original generator's distance + epsilon <= threshold rule.
    Zero-window branches route right, as in the historical scripts.
    """
    branches = 2**depth - 1
    while node < branches:
        left = distance[node][index] + epsilon <= b[node]
        if not left:
            sign = -sign
        node = 2 * node + (1 if left else 2)
    return int((sign + 1) // 2)


def make_windows(J, H, depth, rng):
    """Generate pairs of contiguous windows, with the original 5/6 split rate.

    Inactive branches have start -1. An inactive ancestor cannot have active
    descendants; the root is always active. Each active node adds one leaf.
    """
    branches = 2**depth - 1
    active = np.zeros(branches, dtype=bool)
    active[0] = True
    input_starts = np.full(branches, -1)
    exemplar_starts = np.full(branches, -1)
    for node in range(branches):
        if node:
            active[node] = active[(node - 1) // 2] and rng.random() < 5 / 6
        if active[node]:
            exemplar_starts[node], input_starts[node] = rng.integers(J - H + 1, size=2)
    return input_starts, exemplar_starts


def simulate(J, H, depth, population, windows, rng, epsilon):
    """Return normalized Brownian series, binary labels, and the true exemplar."""
    X = np.empty((population, J))
    X[:, 0] = rng.uniform(0, 1, population)
    brownian(X[:, 0], J - 1, 1, 1, out=X[:, 1:], rng=rng)
    X = SeriesMinMaxScaler().fit_transform(np.round(X, 3))
    exemplar = X[int(rng.integers(population))].copy()
    distances = np.zeros((2**depth - 1, population))
    for node, (start, ex_start) in enumerate(zip(*windows)):
        if start >= 0:
            distances[node] = np.round(
                np.abs(X[:, start : start + H] - exemplar[ex_start : ex_start + H]).sum(axis=1), 3
            )
    thresholds = np.round(np.median(distances, axis=1), 3)
    y = np.array(
        [assignLabel(distances, thresholds, depth, i, epsilon=epsilon) for i in range(population)]
    )
    return X, y, exemplar


def fit_tree(X, y, exemplar, depth, H, alpha, epsilon, LT, args, seed, cv=False):
    """Solve one model, retaining failure status and releasing solver resources."""
    model, branches, leaves, n, exemplar = generateModel(
        X, y, alpha, depth, H, epsilon, LT=LT, K=2, true_exemplar=exemplar
    )
    try:
        limit = args.cv_time_limit if cv else args.time_limit
        configure_solver(model, limit, epsilon, seed, args.threads, args.quiet)
        model.Params.Presolve = 2
        if args.kind == "depth":
            model.Params.MIPGapAbs = 1 / n
        elif args.kind == "length":
            model.Params.MIPFocus = 1 if cv else 3
        model.optimize()
        info = solver_summary(model)
        info["prediction_tolerance"] = model.Params.FeasibilityTol
        tree = (
            retrieveSolution(model, branches, leaves, X.shape[1], H, n, 2)
            if model.SolCount
            else None
        )
        return tree, info
    finally:
        model.dispose()


def _score(tree, X, y, exemplar, tolerance):
    return predictModel(*tree, X, y, exemplar, tolerance=tolerance)


def run_parameter(job):
    """Run one grid value; seeds depend on the value, not process scheduling."""
    parameter, args = job
    kind = args.kind
    if kind == "learning":
        J, H_true, depth_true = parameter, max(1, parameter // 5), 1
        candidates = [(1, H_true, 1.0)]
    elif kind == "depth":
        J, H_true, depth_true = args.series_length, args.true_shapelet_length, parameter
        candidates = [
            (depth, H_true, alpha)
            for depth, alpha in product(args.depth_grid, args.alpha_grid)
            if depth != 1 or alpha == args.alpha_grid[0]
        ]
    else:
        J, H_true, depth_true = args.series_length, parameter, args.true_depth
        candidates = [(depth_true, H, 1.0) for H in args.shapelet_grid]
    if H_true > J:
        raise ValueError("True shapelet length exceeds the series length.")
    epsilon = 0.0001 if kind == "learning" else 0.0002
    if any(H > J for _, H, _ in candidates):
        raise ValueError("A candidate shapelet length exceeds the series length.")
    records = []
    mode_seed = {"depth": 1, "length": 2, "learning": 3}[kind]
    for tree_repeat in range(args.tree_repeats):
        window_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, mode_seed, parameter, tree_repeat])
        )
        windows = make_windows(J, H_true, depth_true, window_rng)
        for data_repeat in range(args.data_repeats):
            seed = int(
                np.random.SeedSequence(
                    [args.seed, mode_seed, parameter, tree_repeat, data_repeat, 1]
                ).generate_state(1)[0]
                % 2_000_000_001
            )
            rng = np.random.default_rng(seed)
            X, y, exemplar = simulate(J, H_true, depth_true, args.population, windows, rng, epsilon)
            counts = np.bincount(y, minlength=2)
            if counts.min() < 2:
                raise ValueError(
                    "Simulation produced fewer than two samples in a class; increase population or change seed."
                )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=args.train_size, stratify=y, random_state=seed
            )
            LT = int(np.bincount(y_train).max()) if kind == "depth" else 1
            cv_results = []
            if kind != "learning":
                if np.bincount(y_train, minlength=2).min() < args.folds:
                    raise ValueError(
                        "Each training class needs at least --folds samples; increase --train-size."
                    )
                folds = list(
                    StratifiedKFold(args.folds, shuffle=True, random_state=seed).split(
                        X_train, y_train
                    )
                )
                for depth, H, alpha in candidates:
                    scores, solvers = [], []
                    for train_index, val_index in folds:
                        fold_LT = (
                            int(np.bincount(y_train[train_index]).max()) if kind == "depth" else 1
                        )
                        tree, info = fit_tree(
                            X_train[train_index],
                            y_train[train_index],
                            exemplar,
                            depth,
                            H,
                            alpha,
                            epsilon,
                            fold_LT,
                            args,
                            seed,
                            cv=True,
                        )
                        # Preserve the original zero-score policy for failed CV solves,
                        # but retain their status so they cannot disappear in a mean.
                        score = (
                            0.0
                            if tree is None
                            else _score(
                                tree,
                                X_train[val_index],
                                y_train[val_index],
                                exemplar,
                                info["prediction_tolerance"],
                            )
                        )
                        scores.append(score)
                        solvers.append(info)
                    cv_results.append(
                        {
                            "depth": depth,
                            "H": H,
                            "alpha": alpha,
                            "mean_accuracy": float(np.mean(scores)),
                            "fold_scores": scores,
                            "solvers": solvers,
                        }
                    )
                best = int(np.argmax([result["mean_accuracy"] for result in cv_results]))
            else:
                best = 0
            depth, H, alpha = candidates[best]
            tree, info = fit_tree(
                X_train, y_train, exemplar, depth, H, alpha, epsilon, LT, args, seed
            )
            record = {
                "parameter": parameter,
                "tree_repeat": tree_repeat,
                "data_repeat": data_repeat,
                "seed": seed,
                "series_length": J,
                "true_H": H_true,
                "true_depth": depth_true,
                "true_leaves": int((windows[0] >= 0).sum() + 1),
                "selected_depth": depth,
                "selected_H": H,
                "selected_alpha": alpha,
                **info,
                "train_accuracy": np.nan,
                "test_accuracy": np.nan,
                "active_leaves": np.nan,
                "cv_failed_solves": sum(
                    s["solution_count"] == 0 for r in cv_results for s in r["solvers"]
                ),
                "cv_results": cv_results,
            }
            if tree is not None:
                record.update(
                    train_accuracy=_score(
                        tree, X_train, y_train, exemplar, info["prediction_tolerance"]
                    ),
                    test_accuracy=_score(
                        tree, X_test, y_test, exemplar, info["prediction_tolerance"]
                    ),
                    active_leaves=int(sum(tree[3])),
                )
            records.append(record)
            print(
                f"{kind}={parameter}, repetition {tree_repeat + 1}/{data_repeat + 1}: "
                f"status={info['status']}, test={record['test_accuracy']:.4f}",
                flush=True,
            )
    return records


def summarize(records):
    """Aggregate successful fits; report attempted/successful counts explicitly.

    Standard deviations use ddof=0, matching the original NumPy summaries.
    Empty groups produce NaN rather than a fabricated zero accuracy.
    """
    frame = pd.DataFrame(records)
    rows = []
    metrics = (
        "train_accuracy",
        "test_accuracy",
        "active_leaves",
        "selected_depth",
        "selected_H",
        "true_leaves",
    )
    for parameter, group in frame.groupby("parameter", sort=False):
        successful = group[group.solution_count > 0]
        row = {
            "parameter": parameter,
            "attempted_runs": len(group),
            "successful_runs": len(successful),
            "failed_runs": len(group) - len(successful),
            "cv_failed_solves": int(group.cv_failed_solves.sum()),
        }
        for metric in metrics:
            values = group[metric] if metric == "true_leaves" else successful[metric]
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser(kind):
    parser = argparse.ArgumentParser(
        description=f"Synthetic OST {kind} experiment (known true exemplar)."
    )
    parser.set_defaults(kind=kind)
    parser.add_argument(
        "--values",
        type=int,
        nargs="+",
        default=DEFAULT_GRIDS[kind],
        help="Values of the varied parameter",
    )
    parser.add_argument("--tree-repeats", type=int, default=5)
    parser.add_argument("--data-repeats", type=int, default=5)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--series-length", type=int, default=100)
    parser.add_argument("--true-depth", type=int, default=3, help="For the shapelet-length study")
    parser.add_argument("--true-shapelet-length", type=int, default=9, help="For the depth study")
    parser.add_argument("--depth-grid", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--alpha-grid", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1, 1, 10]
    )
    parser.add_argument("--shapelet-grid", type=int, nargs="+", default=list(range(2, 19, 2)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=3600 if kind == "learning" else 600)
    parser.add_argument("--cv-time-limit", type=float, default=60)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel parameter values; each uses --threads solver threads",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="Suppress Gurobi logs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"results/{kind}.csv"),
        help="Summary CSV; also writes .runs.csv and .json",
    )
    return parser


def validate_args(args):
    for name in (
        "tree_repeats",
        "data_repeats",
        "population",
        "train_size",
        "series_length",
        "true_depth",
        "true_shapelet_length",
        "folds",
        "workers",
        "threads",
    ):
        positive_integer(getattr(args, name), name)
    for value in args.values + args.depth_grid + args.shapelet_grid:
        positive_integer(value, "grid value")
    if args.train_size >= args.population or args.folds < 2:
        raise ValueError("Require train-size < population and folds >= 2.")
    if not 0 <= args.seed <= 2_000_000_000:
        raise ValueError("seed must be in 0..2000000000.")
    if any(not np.isfinite(value) or value < 0 for value in args.alpha_grid):
        raise ValueError("Alpha values must be finite and nonnegative.")
    if any(not np.isfinite(value) or value <= 0 for value in [args.time_limit, args.cv_time_limit]):
        raise ValueError("Time limits must be finite and positive.")
    if len(set(args.values)) != len(args.values):
        raise ValueError("Parameter values must be unique.")
    if args.output.suffix.lower() != ".csv":
        raise ValueError("output must have the .csv extension.")


def compute_all(kind, parameter):
    """Compatibility entry for the original computeAll(parameter) functions."""
    args = build_parser(kind).parse_args([])
    return summarize(run_parameter((parameter, args)))


def main(kind, argv=None):
    parser = build_parser(kind)
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        jobs = [(value, args) for value in args.values]
        if args.workers == 1:
            batches = [run_parameter(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                batches = list(pool.map(run_parameter, jobs))
        records = [record for batch in batches for record in batch]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summarize(records).to_csv(args.output, index=False)
        pd.DataFrame(
            [{k: v for k, v in row.items() if k != "cv_results"} for row in records]
        ).to_csv(args.output.with_suffix(".runs.csv"), index=False)
        write_json(
            args.output.with_suffix(".json"),
            {"settings": vars(args), "versions": environment_versions(), "runs": records},
        )
        print(
            f"Saved {args.output}, {args.output.with_suffix('.runs.csv')}, and {args.output.with_suffix('.json')}"
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
