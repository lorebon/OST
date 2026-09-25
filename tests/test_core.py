"""Regression tests for scientific behavior, independent of solver licensing."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from classification import plotShapelets, postProcess, predict, predictModel
from experiments import make_windows, simulate, summarize
from preprocessing import as_series_array, preprocessTest, preprocessTrain
from training import KMedoids, compute_big_M, compute_lb_ub, generateModel, retrieveSolution
from workflow import write_json


def test_label_encoding_does_not_merge_overlapping_labels():
    X = np.tile([0, 2, 4], (4, 1))
    _, y, classes, _, scaler = preprocessTrain(X, [1, 0, 1, 0])
    np.testing.assert_array_equal(y, [1, 0, 1, 0])
    _, test_y = preprocessTest(X, [0, 1, 0, 1], classes, scaler)
    np.testing.assert_array_equal(test_y, [0, 1, 0, 1])


@pytest.mark.parametrize("labels", [[-3, 8], ["dog", "cat"], ["1.5", "2.5"]])
def test_nonconsecutive_and_string_labels(labels):
    X = [[0, 3, 6], [2, 2, 2]]
    scaled, y, classes, _, scaler = preprocessTrain(X, labels)
    assert len(np.unique(y)) == 2
    np.testing.assert_allclose(scaled, [[0, 0.5, 1], [0, 0, 0]])
    _, mapped = preprocessTest(X, labels, classes, scaler)
    np.testing.assert_array_equal(mapped, y)


def test_truncation_keeps_requested_prefix_and_full_size_is_noop():
    X = [[0, 1, 2, 3], [3, 2, 1, 0]]
    scaled, _, classes, _, scaler = preprocessTrain(X, [0, 1], maxSize=2, maxLength=2)
    np.testing.assert_allclose(scaled, [[0, 1 / 3], [1, 2 / 3]])
    test, _ = preprocessTest(X, [0, 1], classes, scaler, maxLength=2)
    np.testing.assert_array_equal(test, scaled)


def test_unknown_label_and_length_mismatch_rejected():
    _, _, classes, _, scaler = preprocessTrain([[0, 1]], ["a"])
    with pytest.raises(ValueError, match="absent"):
        preprocessTest([[0, 1]], ["b"], classes, scaler)
    with pytest.raises(ValueError, match="same original length"):
        preprocessTest([[0, 1, 2]], ["a"], classes, scaler)


def test_nested_and_numpy3d_conversion():
    nested = pd.DataFrame({"series": [pd.Series([1, 2]), pd.Series([3, 4])]})
    np.testing.assert_array_equal(as_series_array(nested), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(as_series_array(np.ones((2, 1, 3))), np.ones((2, 3)))


@pytest.mark.parametrize("X", [[], [[np.nan]], [[np.inf]], np.ones((2, 2, 3)), [[1], [1, 2]]])
def test_unsupported_series_rejected(X):
    with pytest.raises(ValueError):
        as_series_array(X)


@pytest.mark.parametrize("max_length", [0, -1, 1.5, 4])
def test_invalid_truncation_rejected(max_length):
    with pytest.raises(ValueError):
        preprocessTrain([[0, 1, 2]], [0], maxLength=max_length)


def test_subsampling_is_stratified_and_seeded():
    X = np.arange(60).reshape(20, 3)
    labels = np.repeat([0, 1], 10)
    first = preprocessTrain(X, labels, maxSize=8, random_state=13)
    second = preprocessTrain(X, labels, maxSize=8, random_state=13)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(np.bincount(first[1]), [4, 4])


def test_medoid_minimizes_total_l1_distance():
    np.testing.assert_array_equal(KMedoids([[0, 0], [1, 1], [10, 10]]), [1, 1])
    np.testing.assert_array_equal(KMedoids([[0], [2]]), [0])
    with pytest.raises(ValueError, match="k=1"):
        KMedoids([[0]], k=2)


def test_window_bounds_include_last_start_and_full_length():
    lb, ub = compute_lb_ub([[0, 1, 9]], 2)
    np.testing.assert_array_equal(lb, [[0, 1]])
    np.testing.assert_array_equal(ub, [[1, 9]])
    lb, ub = compute_lb_ub([[0, 1, 9]], 3)
    np.testing.assert_array_equal(lb, ub)
    np.testing.assert_array_equal(lb, [[0, 1, 9]])


def test_distance_bounds_cover_every_window_pair():
    X = np.array([[0, 0.5, 1], [1, 0.2, 0]])
    ex = np.array([[0.2, 0.7, 0.9]])
    M1, M2 = compute_big_M(*compute_lb_ub(X, 2), *compute_lb_ub(ex, 2))
    for i, row in enumerate(X):
        for start in range(2):
            for ex_start in range(2):
                delta = abs(row[start : start + 2] - ex[0, ex_start : ex_start + 2])
                assert np.all(delta <= M1[i] + 1e-12)
                assert delta.sum() <= M2[i] + 1e-12


def test_prediction_uses_last_window_and_unrounded_threshold():
    A = A_hat = np.array([[0, 0, 1]])
    X = [[1, 1, 0.12346], [0, 0, 0.12344]]
    output = predict(A, A_hat, [0.12345], [1, 1], [0, 1], X, [0, 0, 0])
    np.testing.assert_array_equal(output, [1, 0])
    assert predictModel(A, A_hat, [0.12345], [1, 1], [0, 1], X, [1, 0], [0, 0, 0]) == 1


def test_pruning_works_at_depth_four():
    active = np.zeros(16)
    active[10] = 1
    labels = np.full(16, -1)
    labels[10] = 7
    # No selector is needed when all decisions are bypassed.
    output = predict(
        np.zeros((15, 2)), np.zeros((15, 2)), np.zeros(15), active, labels, [[0, 1]], [0, 1]
    )
    np.testing.assert_array_equal(output, [7])
    with pytest.raises(ValueError, match="no active leaves"):
        postProcess([0], [0, 0])


def test_plot_uses_leaf_mask_and_correct_shapelet_length():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = plotShapelets([[0, 1]], [1, 1], [0, 0.5, 1], show=False)
    assert len(figures) == 1
    np.testing.assert_array_equal(figures[0].axes[0].lines[1].get_ydata(), [0.5, 1])
    plt.close("all")


def test_missing_incumbent_has_clear_error():
    with pytest.raises(RuntimeError, match="No feasible incumbent"):
        retrieveSolution(SimpleNamespace(SolCount=0, Status=9), 1, 2, 3, 1, 2, 2)


@pytest.mark.parametrize(
    "override", [{"H": 4}, {"depth": 0}, {"LT": 0}, {"epsilon": 0}, {"alpha": -1}]
)
def test_model_validation_before_solver_import(override):
    args = dict(alpha=1, depth=1, H=1, epsilon=1e-5)
    args.update(override)
    with pytest.raises(ValueError):
        generateModel([[0, 1, 0], [1, 0, 1]], np.array([0, 1]), **args)


def test_simulation_reproducible_and_inactive_ancestors_stay_inactive():
    windows = make_windows(8, 2, 4, np.random.default_rng(1))
    for node in range(1, len(windows[0])):
        if windows[0][(node - 1) // 2] == -1:
            assert windows[0][node] == -1
    first = simulate(8, 2, 4, 30, windows, np.random.default_rng(2), 0.0002)
    second = simulate(8, 2, 4, 30, windows, np.random.default_rng(2), 0.0002)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_failed_runs_are_counted_and_excluded_from_accuracy():
    base = dict(parameter=1, cv_failed_solves=2, selected_depth=1, selected_H=2, true_leaves=2)
    records = [
        dict(base, solution_count=1, train_accuracy=0.8, test_accuracy=0.6, active_leaves=2),
        dict(
            base,
            solution_count=0,
            train_accuracy=np.nan,
            test_accuracy=np.nan,
            active_leaves=np.nan,
        ),
    ]
    row = summarize(records).iloc[0]
    assert row.failed_runs == 1
    assert row.successful_runs == 1
    assert row.test_accuracy_mean == 0.6
    assert row.cv_failed_solves == 4


def test_json_nonfinite_values_are_portable(tmp_path):
    import json

    path = tmp_path / "nested" / "result.json"
    write_json(path, {"gap": float("inf"), "array": np.array([1, np.nan])})
    assert json.loads(path.read_text()) == {"gap": None, "array": [1.0, None]}
