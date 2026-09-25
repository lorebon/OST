# Method and code guide

[Home](../README.md) · [Method](method.md) · [Experiments](experiments.md) · [Development](development.md)

This guide describes the implemented model and its core functions. For the scientific context, see the [paper](https://doi.org/10.1016/j.ejco.2024.100091); for corrections to the original code, see [CHANGES.md](../CHANGES.md).

## Optimization model

`training.py::generateModel` builds the model; it does not optimize it. Its return values are `(model, branch_nodes, leaf_nodes, n, exemplar)`. `retrieveSolution` extracts `(A, A_hat, thresholds, active_leaves, leaf_labels)` after a feasible incumbent exists.

- A depth-D tree has `2**D - 1` branch positions and `2**D` leaf positions, numbered breadth-first. Root 0 has children 1 and 2.
- Each branch chooses exactly one input window position and one exemplar window position. There are `J - H + 1` possible starts, including the final window and the full-series case `H = J`.
- The input position is shared across samples at a branch. Prediction compares those selected windows; it does **not** search for a different best-matching input window for every sample.
- Left routing uses `distance <= threshold`; right routing is separated by epsilon during training. Prediction uses unrounded thresholds and a small numerical tolerance. The entry scripts use the solver feasibility tolerance, set below epsilon.
- Every class must label at least one active leaf. Each active leaf must contain at least `ceil(0.05 * n)` training samples. This can make a model infeasible even when its depth is sufficient.
- If `2**depth > K`, the objective is `misclassified_count / LT + alpha * active_leaf_count`. Otherwise, it is just the misclassification count. `LT` retains the original implementation's convention: the **majority-class count** in `main.py` and the depth experiment, and 1 in the length and learning experiments. It is not an accuracy fraction or a baseline error count.
- Empty subtrees are bypassed during prediction. Active-leaf labels are encoded integers; inactive leaves have label -1.

The name “optimal” describes the formulation. A time-limited run may return a feasible tree without proving optimality. Saved results distinguish the solver status, incumbent objective, bound, MIP gap, and optimality flag. No-incumbent runs produce an error instead of reading nonexistent solution values; `main.py` still writes available diagnostics when `--output` is supplied.

A numerical issue is not a reason to round learned thresholds. Keep the branch margin larger than the feasibility tolerance and inspect the solver status/log. See [Gurobi's tolerance documentation](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html).

## Core files

| File | Responsibility |
| --- | --- |
| [training.py](../code/training.py) | MIP formulation, exemplar medoid, distance bounds, solution extraction |
| [preprocessing.py](../code/preprocessing.py) | Input validation, stable label encoding, per-series normalization |
| [classification.py](../code/classification.py) | Predictions and propagation of subtree activity |
| [workflow.py](../code/workflow.py) | Solver configuration, diagnostics, environment metadata, JSON output |

## Using the functions

The core functions retain their historical names. Put custom scripts alongside them in `code/` so ordinary local imports work; no package installation is needed.

1. Call `preprocessTrain` to obtain normalized data, encoded labels, ordered original classes, encoded classes, and a scaler.
2. Choose an exemplar with `KMedoids`, or supply an already normalized series to `generateModel` through `true_exemplar`.
3. Configure the returned model with `workflow.configure_solver`, then call `model.optimize()`.
4. Check that `model.SolCount > 0` before calling `retrieveSolution`. Dispose of the model after extraction, ideally in a `finally` block; see [main.py](../code/main.py) for the complete workflow.
5. Apply `preprocessTest` using the ordered training classes and scaler. Call `classification.predict` for encoded labels, or `predictModel` for accuracy. Use the same prediction tolerance as the training workflow.
6. Recover original labels by indexing the ordered classes array with the encoded predictions.

`main.py --output` saves the extracted tree, ordered classes, original series length, preprocessing settings, and prediction tolerance. The tree's fields are `A`, `A_hat`, `thresholds`, `active_leaves`, `leaf_labels`, `exemplar`, and `prediction_tolerance`. New inputs must use the same original series length, per-series normalization, and truncation as training.
