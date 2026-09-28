"""Small real-solver regressions, compatible with a size-limited license."""

import numpy as np
import pytest

from classification import predict
from training import generateModel, retrieveSolution
from workflow import configure_solver


@pytest.fixture(scope="module", autouse=True)
def licensed_solver():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as exc:
        if exc.errno == 10009:
            pytest.skip(f"Gurobi license unavailable: {exc}")
        raise


@pytest.mark.gurobi
@pytest.mark.parametrize("H", [1, 3])
def test_solved_tree_matches_assignments_including_constant_series(H):
    X = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model, branches, leaves, n, ex = generateModel(X, y, 1, 1, H, 1e-4, exemplar=0)
    try:
        configure_solver(model, 10, 1e-4, quiet=True)
        model.optimize()
        assert model.Status == 2
        assert model.ObjVal == pytest.approx(0)
        tree = retrieveSolution(model, branches, leaves, X.shape[1], H, n, 2)
        output = predict(*tree, X, ex, tolerance=model.Params.FeasibilityTol)
        np.testing.assert_array_equal(output, y)
        for i in range(n):
            leaf = next(t for t in range(leaves) if model.getVarByName(f"z[{i},{t}]").X > 0.5)
            assert output[i] == tree[4][leaf]
    finally:
        model.dispose()


@pytest.mark.gurobi
def test_last_window_can_be_selected():
    X = np.array([[0, 0, 0], [0, 0, 1]], dtype=float)
    model, branches, leaves, n, ex = generateModel(X, np.array([0, 1]), 1, 1, 1, 1e-4, exemplar=0)
    try:
        configure_solver(model, 10, 1e-4, quiet=True)
        model.optimize()
        assert model.Status == 2
        tree = retrieveSolution(model, branches, leaves, 3, 1, n, 2)
        np.testing.assert_array_equal(tree[0], [[0, 0, 1]])
        np.testing.assert_array_equal(predict(*tree, X, ex), [0, 1])
    finally:
        model.dispose()


@pytest.mark.gurobi
def test_loss_matches_selected_class_even_when_not_leaf_majority():
    # Representing every class can require a non-majority leaf label. The
    # unselected majority label must not incorrectly cap that leaf's loss.
    X = np.array([[0], [0], [0], [0], [1]], dtype=float)
    y = np.array([0, 0, 0, 1, 0])
    model, branches, leaves, n, ex = generateModel(X, y, 1, 1, 1, 1e-4, exemplar=0)
    try:
        model.update()
        model.addConstr(model.getVarByName("c[1,0]") == 1)
        configure_solver(model, 10, 1e-4, quiet=True)
        model.optimize()
        assert model.Status == 2
        assert model.ObjVal == pytest.approx(3)
        tree = retrieveSolution(model, branches, leaves, 1, 1, n, 2)
        assert np.count_nonzero(predict(*tree, X, ex) != y) == 3
    finally:
        model.dispose()


@pytest.mark.gurobi
def test_multiclass_tree_uses_three_active_leaves():
    X = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    y = np.array([0, 1, 2])
    model, branches, leaves, n, ex = generateModel(X, y, 0.01, 2, 2, 1e-4, exemplar=0)
    try:
        configure_solver(model, 10, 1e-4, quiet=True)
        model.optimize()
        assert model.Status == 2
        tree = retrieveSolution(model, branches, leaves, 2, 2, n, 3)
        assert sum(tree[3]) == 3
        assert model.ObjVal == pytest.approx(0.03)
        np.testing.assert_array_equal(predict(*tree, X, ex), y)
    finally:
        model.dispose()


@pytest.mark.gurobi
def test_stump_matches_exhaustive_window_and_threshold_search():
    X = np.array(
        [[0, 0.2, 0.7], [0, 0.1, 1], [1, 0.8, 0], [0.1, 0.8, 0.2], [1, 0.5, 0.2], [0.8, 0.2, 0.1]]
    )
    y = np.array([0, 1, 1, 0, 1, 0])
    epsilon = 1e-4
    best = len(y)
    for start in range(2):
        for ex_start in range(2):
            distances = abs(X[:, start : start + 2] - X[0, ex_start : ex_start + 2]).sum(axis=1)
            for threshold in np.unique(distances):
                left = distances <= threshold
                if left.all() or not left.any():
                    continue
                if distances[~left].min() < threshold + epsilon:
                    continue
                for left_label in (0, 1):
                    labels = np.where(left, left_label, 1 - left_label)
                    best = min(best, np.count_nonzero(labels != y))
    model, branches, leaves, n, ex = generateModel(X, y, 1, 1, 2, epsilon, exemplar=0)
    try:
        configure_solver(model, 10, epsilon, quiet=True)
        model.optimize()
        assert model.Status == 2
        assert model.ObjVal == pytest.approx(best)
        tree = retrieveSolution(model, branches, leaves, 3, 2, n, 2)
        assert (
            np.count_nonzero(predict(*tree, X, ex, tolerance=model.Params.FeasibilityTol) != y)
            == best
        )
    finally:
        model.dispose()
