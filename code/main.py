"""Train and evaluate OST on an official UCR train/test split.

Run `python code/main.py --help` from the repository root.
"""

import argparse
from math import ceil, log2
from pathlib import Path

import numpy as np

from classification import plotShapelets, predict
from preprocessing import preprocessTest, preprocessTrain
from training import KMedoids, generateModel, retrieveSolution
from workflow import configure_solver, environment_versions, solver_summary, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ItalyPowerDemand", help="UCR dataset name")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Dataset download/cache folder"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Tree depth (default: minimum needed for the classes)",
    )
    parser.add_argument("--shapelet-length", type=int, default=4, metavar="H")
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Active-leaf penalty when 2**depth > K"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-5, help="Right-branch separation margin"
    )
    parser.add_argument("--max-size", type=int, help="Stratified training subset size")
    parser.add_argument(
        "--max-length", type=int, help="Keep this many initial points after scaling"
    )
    parser.add_argument(
        "--exemplar-class",
        type=int,
        default=None,
        help="Encoded class whose medoid is used (default: class 1, or 0 for one class)",
    )
    parser.add_argument("--time-limit", type=float, default=600, help="Solver seconds")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="Gurobi threads")
    parser.add_argument(
        "--output", type=Path, help="Save metrics, settings, and extracted tree as JSON"
    )
    parser.add_argument("--plot", action="store_true", help="Show learned exemplar windows")
    parser.add_argument("--quiet", action="store_true", help="Suppress the Gurobi optimization log")
    return parser


def run(args):
    from sktime.datasets import load_UCR_UEA_dataset

    args.data_dir.mkdir(parents=True, exist_ok=True)
    # numpy3D retains the variable axis, so multivariate input is rejected
    # rather than silently flattened into a single series.
    X_train, y_train = load_UCR_UEA_dataset(
        args.dataset,
        split="train",
        return_X_y=True,
        return_type="numpy3D",
        extract_path=str(args.data_dir.resolve()),
    )
    X_train, y_train, classes, _, scaler = preprocessTrain(
        X_train, y_train, args.max_size, args.max_length, args.seed
    )
    K = len(classes)
    depth = args.depth if args.depth is not None else max(1, ceil(log2(K)))
    exemplar_class = min(1, K - 1) if args.exemplar_class is None else args.exemplar_class
    if not 0 <= exemplar_class < K:
        raise ValueError(f"exemplar-class must be in 0..{K - 1}.")
    exemplar = KMedoids(X_train[y_train == exemplar_class])
    baseline_count = int(np.bincount(y_train).max())
    print(f"Training: {len(X_train)} series, {X_train.shape[1]} points, {K} classes")
    print(f"Class encoding: {dict(enumerate(classes.tolist()))}")
    print(f"Majority baseline (training): {baseline_count / len(y_train):.4f}")
    model, branches, leaves, n, exemplar = generateModel(
        X_train,
        y_train,
        args.alpha,
        depth,
        args.shapelet_length,
        args.epsilon,
        LT=baseline_count,
        true_exemplar=exemplar,
    )
    result = {
        "dataset": args.dataset,
        "settings": vars(args).copy(),
        "depth": depth,
        "exemplar_class": exemplar_class,
        "classes": classes,
        "training_shape": X_train.shape,
        "original_series_length": scaler.n_features_in_,
        "versions": environment_versions(),
        "baseline_train_accuracy": baseline_count / n,
    }
    try:
        configure_solver(model, args.time_limit, args.epsilon, args.seed, args.threads, args.quiet)
        model.optimize()
        result["solver"] = solver_summary(model)
        if model.SolCount < 1:
            if args.output:
                write_json(args.output, result)
            raise RuntimeError(
                f"No feasible tree found (status {model.Status}); try a longer time limit or inspect feasibility."
            )
        A, A_hat, b, active, labels = retrieveSolution(
            model, branches, leaves, X_train.shape[1], args.shapelet_length, n, K
        )
        tolerance = model.Params.FeasibilityTol
    finally:
        model.dispose()

    X_test, y_test = load_UCR_UEA_dataset(
        args.dataset,
        split="test",
        return_X_y=True,
        return_type="numpy3D",
        extract_path=str(args.data_dir.resolve()),
    )
    X_test, y_test = preprocessTest(X_test, y_test, classes, scaler, args.max_length)
    train_pred = predict(A, A_hat, b, active, labels, X_train, exemplar, tolerance)
    test_pred = predict(A, A_hat, b, active, labels, X_test, exemplar, tolerance)
    result.update(
        train_accuracy=float(np.mean(train_pred == y_train)),
        test_accuracy=float(np.mean(test_pred == y_test)),
        test_shape=X_test.shape,
        tree={
            "A": A,
            "A_hat": A_hat,
            "thresholds": b,
            "active_leaves": active,
            "leaf_labels": labels,
            "exemplar": exemplar,
            "prediction_tolerance": tolerance,
        },
    )
    print(f"Training accuracy: {result['train_accuracy']:.4f}")
    print(f"Test accuracy: {result['test_accuracy']:.4f}; active leaves: {active.sum()}")
    print(
        f"Solver status: {result['solver']['status']}; optimal: {result['solver']['optimal']}; gap: {result['solver']['mip_gap']:.6g}"
    )
    if args.output:
        write_json(args.output, result)
        print(f"Saved {args.output}")
    if args.plot:
        plotShapelets(A_hat, active, exemplar)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
