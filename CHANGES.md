# Changes from the original research scripts

This file separates changes in scientific behavior from repository maintenance. The corrected code is not claimed to regenerate the published tables exactly. The paper is [Bonasera and Gualandi (2024)](https://doi.org/10.1016/j.ejco.2024.100091); its numerical experiments have not been rerun in full as part of this cleanup.

## Corrections that can change results

1. **Class encoding:** original labels were converted to integers and replaced in place while iterating over an unordered set. Remapping could collide with labels still awaiting conversion. Encoding now uses a separate, deterministic array and supports string labels. Test data use the exact same ordered mapping.
2. **Medoid:** `KMedoids` previously chose a random starting series, then selected the series nearest to itself. Its precomputed pairwise-distance matrix was unused. It now chooses the series minimizing total L1 distance, breaking ties by input order. Only the already-intended single-medoid case is supported.
3. **Series truncation:** `maxLength` previously discarded that many initial points (`X[:, maxLength:]`). It now retains exactly that many initial points (`X[:, :maxLength]`). Per-series scaling still precedes truncation.
4. **Window endpoints:** the model, bounds, extraction, prediction, and plotting now use all `J-H+1` legal windows. The original `J-H` convention omitted the last window and failed at `H=J`. Saved selector matrices from the old convention are not directly compatible.
5. **Split relaxation:** right-branch big-M values now cover the largest possible threshold plus epsilon, rather than using each sample's own distance bound without the margin. This avoids restricting an inactive inequality, notably for zero-distance/constant samples. Thresholds have a valid explicit upper bound.
6. **Leaf loss:** both bounds now activate the exact misclassification count for the selected leaf label. The original upper bound used `n*c` instead of `n*(1-c)`, which could accidentally constrain a leaf through its unselected labels (particularly when every class must be represented).
7. **Numerical extraction and prediction:** binary solution values are interpreted using a 0.5 cutoff instead of integer truncation or any nonzero value. Thresholds are no longer rounded to three decimals at prediction time. The entry scripts pass the solver feasibility tolerance to prediction and keep that tolerance below epsilon.
8. **Pruning and plots:** subtree activity is propagated bottom-up at every depth. Shapelet plotting receives the active-leaf vector and displays branches with two active child subtrees; it no longer indexes a branch array using leaf indices.
9. **Synthetic leaf counts:** the undefined `d` variable in both ground-truth studies is replaced by the actual active-leaf count. Reported counts are leaves, without an extra `+1`.
10. **Synthetic tree generation:** the root is always active; all ancestors must be active before a descendant can split. Previously the root's unrelated random flag and checking only one parent flag could disable root children or re-enable descendants beneath an inactive ancestor.
11. **Cross-validation:** folds are stratified. Populations, splits, and solver seeds are explicit and independent of multiprocessing scheduling. This intentionally changes random draws from the original global NumPy/SciPy state. Failed CV solves still score zero and are now disclosed.

## Maintained scientific conventions

The model still uses L1 distances between selected fixed windows, one supplied exemplar, complete binary-tree positions, at least one active leaf per class, and the original 5% minimum leaf size. The objective's conditional leaf penalty and each experiment's LT convention are preserved. The default parameter grids, 5-by-5 repetition counts, population/training sizes, and solve time limits are retained.

The synthetic experiments still supply the known true exemplar sampled before the train/test split. The original rounded synthetic distances, median thresholds, parity labels, and generator routing margin are retained. `experiment_series_length.py` (formerly `learning_test.py`) varies series length, not training size.

## Usability and compatibility

- Directly runnable scripts live together in `code/`; there is no package, setup.py, or build system.
- Command-line options replace editing hard-coded variables. The UCR runner automatically chooses the minimum sufficient depth unless explicitly overridden.
- Per-series normalization is implemented with NumPy; private sktime conversion internals and direct tslearn/SciPy dependencies are removed. scikit-learn and sktime may still depend on SciPy transitively.
- Existing core function names and positional argument order remain available. `euclidean_distance` remains as a compatibility alias for `manhattan_distance`; both compute L1 distance. `KMedoids(k=...)` explicitly rejects values other than 1.
- The third result from `preprocessTrain` is now an ordered class array, not a set. Pass it unchanged to `preprocessTest`. The scaler returns a 2D matrix. Classification leaf-mask arguments use the name `active_leaves`.
- `predict` returns class IDs; `predictModel` remains an accuracy wrapper. Plotting returns figures and supports `show=False`.
- No-incumbent solves are handled before accessing solution attributes. Solver models are disposed after use, and single-worker/single-thread execution is the default.
- Synthetic entry points retain `computeAll(parameter)` for interactive use, backed by shared code. Summaries use descriptive column names, consistent comma-separated CSV, explicit failure counts, and companion raw records/JSON. Older CSV-analysis scripts may need column updates.
- Runtime bounds in `requirements.txt` are convenient installation constraints, not a compatibility test matrix. `requirements-tested.txt` records the actual environment used for local validation.

## Validation scope

Regression tests cover the corrections above with both solver-independent cases and tiny licensed Gurobi solves. Small runs of all three experiment CLIs and the UCR workflow are used as integration checks. These checks verify execution and specific correctness properties; they do not establish statistical performance or reproduce the full paper experiments.

## Repository organization

The implementation moved from `Optimal Shapelets Tree/` to `code/`, eliminating spaces in command paths. Experiment entry points were renamed to explain the parameter they vary:

- `ground_depth.py` → `code/experiment_depth.py`
- `ground_length.py` → `code/experiment_shapelet_length.py`
- `learning_test.py` → `code/experiment_series_length.py`

Update saved commands to the new paths; old-path launcher files are not retained. Core function names, the `computeAll(parameter)` entry functions, experiment seeds/defaults, and output filenames are unchanged by this move. Tests now import from `code/` through `pytest.ini`.

The README is a shorter entry point. Detailed explanations are in [docs/method.md](docs/method.md), study commands and output conventions in [docs/experiments.md](docs/experiments.md), and checks and path migration notes in [docs/development.md](docs/development.md). This organizational pass does not change the mathematical model or scientific behavior.
