"""Mixed-integer Optimal Shapelets Tree formulation (Bonasera and Gualandi).

Nodes use breadth-first indexing: root 0, children 2*t+1 and 2*t+2.
At a branch, a and a_hat select fixed windows in the input and exemplar;
their L1 distance is compared with b. Gurobi is needed only to build/solve.
"""

from math import ceil

import numpy as np

from preprocessing import as_series_array, positive_integer


def manhattan_distance(a, b):
    """L1 distance used throughout the model."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# Compatibility alias: the original function actually computed L1 distance.
euclidean_distance = manhattan_distance


def KMedoids(series, k=1):
    """Return the exact single L1 medoid; ties choose the first input series.

    O(n^2 * J) time and O(n * J) working memory. Only k=1 is supported.
    """
    if k != 1:
        raise ValueError("KMedoids supports only k=1 (one exemplar).")
    series = as_series_array(series)
    totals = [np.abs(series - candidate).sum() for candidate in series]
    return series[int(np.argmin(totals))].copy()


def compute_lb_ub(set_of_time_series, H):
    """Coordinate bounds over all J-H+1 valid contiguous windows."""
    X = as_series_array(set_of_time_series)
    positive_integer(H, "H")
    if H > X.shape[1]:
        raise ValueError("H must not exceed the series length.")
    windows = np.lib.stride_tricks.sliding_window_view(X, H, axis=1)
    return windows.min(axis=1), windows.max(axis=1)


def compute_big_M(lb, ub, lb_ex, ub_ex):
    """Safe upper bounds on coordinate-wise and total L1 distances."""
    coordinate = np.maximum(ub_ex - lb, ub - lb_ex)
    return coordinate, coordinate.sum(axis=1)


def findAncestors(A_L, A_R, child):
    """Append ancestors reached through left/right edges to the supplied lists."""
    while child:
        parent = (child - 1) // 2
        (A_L if child % 2 else A_R).append(parent)
        child = parent


def generateModel(
    X_train,
    y_train,
    alpha,
    depth,
    H,
    epsilon,
    LT=1,
    exemplar=None,
    K=None,
    true_exemplar=None,
    random_state=1,
):
    """Build (but do not solve) the MIP and return model/tree sizes/exemplar.

    X_train must already be normalized to [0, 1]; y_train uses 0..K-1.
    Every class receives at least one active leaf. Each active leaf contains
    at least ceil(0.05*n) samples. If 2**depth > K, the objective is
    errors/LT + alpha*active_leaves; otherwise it is the error count.
    LT retains the original scripts' majority-class *count* convention.
    true_exemplar supplies a normalized series; exemplar selects a row index.
    """
    X_train = as_series_array(X_train)
    n, J = X_train.shape
    positive_integer(depth, "depth")
    positive_integer(H, "H")
    if H > J:
        raise ValueError("H must not exceed the series length.")
    if X_train.min() < 0 or X_train.max() > 1:
        raise ValueError("Normalize training data to [0, 1] before building the model.")
    y_train = np.asarray(y_train)
    if y_train.shape != (n,) or not np.issubdtype(y_train.dtype, np.integer):
        raise ValueError("y_train must be a vector of encoded integer labels.")
    if K is None:
        K = len(np.unique(y_train))
    positive_integer(K, "K")
    if np.any(y_train < 0) or np.any(y_train >= K):
        raise ValueError("Labels must be in 0..K-1; use preprocessTrain first.")
    if K > 2**depth:
        raise ValueError("depth is too small: 2**depth must be at least K.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")
    if not np.isfinite(LT) or LT <= 0:
        raise ValueError("LT must be finite and positive.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    Nmin = ceil(0.05 * n)
    if K * Nmin > n:
        raise ValueError("Too many classes for the minimum leaf size ceil(0.05*n).")
    if true_exemplar is not None and exemplar is not None:
        raise ValueError("Specify either exemplar or true_exemplar, not both.")
    if true_exemplar is None:
        if exemplar is None:
            exemplar = int(np.random.default_rng(random_state).integers(n))
        if not isinstance(exemplar, (int, np.integer)) or not 0 <= exemplar < n:
            raise ValueError("exemplar must index a training series.")
        X_exemplar = X_train[exemplar].copy()
    else:
        X_exemplar = np.asarray(true_exemplar, dtype=float)
    if (
        X_exemplar.shape != (J,)
        or not np.isfinite(X_exemplar).all()
        or X_exemplar.min() < 0
        or X_exemplar.max() > 1
    ):
        raise ValueError("The exemplar must have length J and finite values in [0, 1].")

    import gurobipy as gp
    from gurobipy import GRB

    lb, ub = compute_lb_ub(X_train, H)
    lb_ex, ub_ex = compute_lb_ub(X_exemplar[None, :], H)
    M1, M2 = compute_big_M(lb, ub, lb_ex, ub_ex)
    # The relaxed right split must accommodate the largest threshold AND
    # epsilon, including for a constant sample whose own M2 is zero.
    threshold_bound = float(M2.max())
    right_M = threshold_bound + epsilon
    leaf_nodes = 2**depth
    branch_nodes = leaf_nodes - 1
    starts = J - H + 1
    m = gp.Model("Optimal Shapelets Tree")

    active_leaves = m.addVars(leaf_nodes, vtype=GRB.BINARY, name="l")
    z = m.addVars(n, leaf_nodes, vtype=GRB.BINARY, name="z")
    w = m.addVars(n, branch_nodes, vtype=GRB.BINARY, name="w")
    N_kt = m.addVars(K, leaf_nodes, vtype=GRB.INTEGER, name="N_kt")
    N_t = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="N_t")
    c = m.addVars(K, leaf_nodes, vtype=GRB.BINARY, name="c")
    loss = m.addVars(leaf_nodes, vtype=GRB.INTEGER, name="L")
    a = m.addVars(starts, branch_nodes, vtype=GRB.BINARY, name="a")
    a_hat = m.addVars(starts, branch_nodes, vtype=GRB.BINARY, name="a_hat")
    b = m.addVars(branch_nodes, ub=threshold_bound, name="b")
    gamma = m.addVars(n, H, branch_nodes, vtype=GRB.BINARY, name="gamma")
    beta = m.addVars(n, H, branch_nodes, name="beta")
    beta_ex = m.addVars(H, branch_nodes, name="beta_2")
    s1 = m.addVars(n, H, branch_nodes, name="s1")
    s2 = m.addVars(n, H, branch_nodes, name="s2")

    objective = loss.sum() / LT + alpha * active_leaves.sum() if leaf_nodes > K else loss.sum()
    m.setObjective(objective, GRB.MINIMIZE)
    for t in range(branch_nodes):
        m.addConstr(a.sum("*", t) == 1, name=f"input_window[{t}]")
        m.addConstr(a_hat.sum("*", t) == 1, name=f"exemplar_window[{t}]")
        for h in range(H):
            beta_ex[h, t].LB = lb_ex[0, h]
            beta_ex[h, t].UB = ub_ex[0, h]
            m.addConstr(
                beta_ex[h, t] == gp.quicksum(a_hat[p, t] * X_exemplar[p + h] for p in range(starts))
            )
        for i in range(n):
            for h in range(H):
                beta[i, h, t].LB = lb[i, h]
                beta[i, h, t].UB = ub[i, h]
                m.addConstr(
                    beta[i, h, t] == gp.quicksum(a[p, t] * X_train[i, p + h] for p in range(starts))
                )
                m.addConstr(s2[i, h, t] - s1[i, h, t] == beta_ex[h, t] - beta[i, h, t])
                m.addConstr(s1[i, h, t] <= M1[i, h] * gamma[i, h, t])
                m.addConstr(s2[i, h, t] <= M1[i, h] * (1 - gamma[i, h, t]))
            distance = s1.sum(i, "*", t) + s2.sum(i, "*", t)
            m.addConstr(distance <= b[t] + M2[i] * w[i, t])
            m.addConstr(distance >= b[t] + epsilon - right_M * (1 - w[i, t]))

    for k in range(K):
        m.addConstr(c.sum(k, "*") >= 1, name=f"class_represented[{k}]")
    for t in range(leaf_nodes):
        m.addConstr(N_t[t] == z.sum("*", t))
        m.addConstr(active_leaves[t] == c.sum("*", t))
        m.addConstr(N_t[t] >= Nmin * active_leaves[t])
        for k in range(K):
            m.addConstr(N_kt[k, t] == gp.quicksum(z[i, t] for i in range(n) if y_train[i] == k))
            m.addConstr(loss[t] >= N_t[t] - N_kt[k, t] - n * (1 - c[k, t]))
            m.addConstr(loss[t] <= N_t[t] - N_kt[k, t] + n * (1 - c[k, t]))
        left, right = [], []
        findAncestors(left, right, t + branch_nodes)
        for i in range(n):
            m.addConstr(z[i, t] <= active_leaves[t])
            m.addConstr(
                gp.quicksum(1 - w[i, a] for a in left) + gp.quicksum(w[i, a] for a in right)
                >= depth * z[i, t]
            )
    for i in range(n):
        m.addConstr(z.sum(i, "*") == 1)
    return m, branch_nodes, leaf_nodes, n, X_exemplar


def retrieveSolution(model, branch_nodes, leaf_nodes, J, H, n, K):
    """Extract an incumbent with rounded binaries; n is kept for compatibility."""
    if model.SolCount < 1:
        raise RuntimeError(f"No feasible incumbent is available (Gurobi status {model.Status}).")

    def value(name):
        variable = model.getVarByName(name)
        if variable is None:
            raise ValueError(f"Model is missing {name}; check the supplied dimensions.")
        return variable.X

    starts = J - H + 1
    A = np.array(
        [[value(f"a[{p},{t}]") > 0.5 for p in range(starts)] for t in range(branch_nodes)],
        dtype=int,
    )
    A_hat = np.array(
        [[value(f"a_hat[{p},{t}]") > 0.5 for p in range(starts)] for t in range(branch_nodes)],
        dtype=int,
    )
    b = np.array([value(f"b[{t}]") for t in range(branch_nodes)])
    leaves = np.array([value(f"l[{t}]") > 0.5 for t in range(leaf_nodes)], dtype=int)
    labels = np.full(leaf_nodes, -1, dtype=int)
    for t in range(leaf_nodes):
        for k in range(K):
            if value(f"c[{k},{t}]") > 0.5:
                labels[t] = k
    return A, A_hat, b, leaves, labels
