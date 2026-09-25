"""Predict with an extracted tree and visualize its exemplar windows."""

import numpy as np

from preprocessing import as_series_array


def postProcess(b, active_leaves):
    """Find empty subtrees bottom-up at any depth.

    Codes: 0 = normal split, -1 = left only, 1 = right only,
    2 = no active descendants. Leaves are in breadth-first order.
    """
    branches = len(b)
    if branches < 1 or ((branches + 1) & branches) or len(active_leaves) != branches + 1:
        raise ValueError("Expected a complete binary tree with 2**depth leaves.")
    active = np.zeros(2 * branches + 1, dtype=bool)
    active[branches:] = np.asarray(active_leaves) > 0.5
    process = np.zeros(branches, dtype=int)
    for node in range(branches - 1, -1, -1):
        left, right = active[2 * node + 1 : 2 * node + 3]
        active[node] = left or right
        process[node] = 0 if left and right else -1 if left else 1 if right else 2
    if not active[0]:
        raise ValueError("The tree has no active leaves.")
    return process


def _selected_index(row):
    indices = np.flatnonzero(np.asarray(row) > 0.5)
    if len(indices) != 1:
        raise ValueError("Each branch must select exactly one window.")
    return int(indices[0])


def computeLabel(A, A_hat, b, branches, data, exemplar, node=0, process=None, tolerance=1e-8):
    """Return a leaf index using unrounded thresholds and L1 distances.

    tolerance absorbs floating-point noise at the left (<=) boundary. It must
    be smaller than the training epsilon; it is not threshold rounding.
    """
    if process is None:
        process = np.zeros(branches, dtype=int)
    H = len(exemplar) - len(A[0]) + 1
    while node < branches:
        if process[node] == 2:
            raise ValueError("Prediction reached an empty subtree.")
        if process[node] == -1:
            go_left = True
        elif process[node] == 1:
            go_left = False
        else:
            start = _selected_index(A[node])
            ex_start = _selected_index(A_hat[node])
            distance = np.abs(exemplar[ex_start : ex_start + H] - data[start : start + H]).sum()
            go_left = distance <= b[node] + tolerance
        node = 2 * node + (1 if go_left else 2)
    return node - branches


def predict(A, A_hat, b, active_leaves, labels, X, exemplar, tolerance=1e-8):
    """Return encoded predictions; X must use the training preprocessing."""
    X = as_series_array(X)
    exemplar = np.asarray(exemplar, dtype=float)
    A, A_hat, b = np.asarray(A), np.asarray(A_hat), np.asarray(b)
    labels = np.asarray(labels)
    if exemplar.shape != (X.shape[1],) or not np.isfinite(exemplar).all():
        raise ValueError("The exemplar and input series must have matching finite lengths.")
    if (
        A.ndim != 2
        or A.shape != A_hat.shape
        or A.shape[0] != len(b)
        or not 1 <= A.shape[1] <= X.shape[1]
    ):
        raise ValueError("Window selectors have incompatible dimensions.")
    if b.ndim != 1 or not np.isfinite(b).all() or labels.shape != (len(active_leaves),):
        raise ValueError("Invalid thresholds or leaf labels.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    process = postProcess(b, active_leaves)
    if np.any(labels[np.asarray(active_leaves) > 0.5] < 0):
        raise ValueError("Every active leaf must have a class label.")
    return np.array(
        [
            labels[
                computeLabel(
                    A, A_hat, b, len(b), row, exemplar, process=process, tolerance=tolerance
                )
            ]
            for row in X
        ]
    )


def predictModel(A, A_hat, b, active_leaves, labels, X_test, y_test, exemplar, tolerance=1e-8):
    """Compatibility wrapper returning accuracy; use predict for class labels."""
    predicted = predict(A, A_hat, b, active_leaves, labels, X_test, exemplar, tolerance)
    y_test = np.asarray(y_test)
    if y_test.shape != predicted.shape:
        raise ValueError("y_test must contain one label per test series.")
    return float(np.mean(predicted == y_test))


def plotShapelets(A_hat, active_leaves, exemplar, show=True):
    """Plot genuine splits (both children active); return Matplotlib figures.

    active_leaves is the active-leaf vector from retrieveSolution, not a branch mask.
    Pass show=False to save figures without opening windows.
    """
    import matplotlib.pyplot as plt

    exemplar = np.asarray(exemplar)
    H = len(exemplar) - len(A_hat[0]) + 1
    process = postProcess(np.zeros(len(A_hat)), active_leaves)
    figures = []
    for node in np.flatnonzero(process == 0):
        start = _selected_index(A_hat[node])
        figure, ax = plt.subplots()
        ax.plot(exemplar, label="Exemplar", color="tab:blue")
        ax.plot(
            np.arange(start, start + H),
            exemplar[start : start + H],
            label=f"Branch {node} shapelet",
            color="tab:red",
            linewidth=3,
        )
        ax.set(xlabel="Time index", ylabel="Normalized value", title=f"Branch {node}")
        ax.legend()
        figure.tight_layout()
        figures.append(figure)
    if show:
        plt.show()
    return figures
