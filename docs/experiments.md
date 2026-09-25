# Running the experiments

[Home](../README.md) · [Method](method.md) · [Experiments](experiments.md) · [Development](development.md)

## Scope and entry points

These commands expose the UCR workflow and three synthetic studies supplied with the research code. They are not a complete automated reproduction of every experiment or table in the paper.

| Study | Entry point | Purpose | Default outputs |
| --- | --- | --- | --- |
| UCR classification | [main.py](../code/main.py) | Fit on the official training split and evaluate on the test split | Console metrics; JSON with `--output` |
| Generating depth | [experiment_depth.py](../code/experiment_depth.py) | Vary true depth; select fitted depth and leaf penalty by CV | `results/depth.csv`, `.runs.csv`, `.json` |
| True shapelet length | [experiment_shapelet_length.py](../code/experiment_shapelet_length.py) | Vary true H; select fitted H by CV | `results/length.csv`, `.runs.csv`, `.json` |
| Series length | [experiment_series_length.py](../code/experiment_series_length.py) | Vary series length at a fixed training size | `results/learning.csv`, `.runs.csv`, `.json` |

The output stems `depth`, `length`, and `learning` are retained for continuity with existing runs. Paths such as `data/` and `results/` are relative to the working directory, so run commands from the repository root. Existing files at a chosen output path are overwritten; choose distinct names for runs you want to keep.

## UCR datasets

Run all commands below from the repository root, after activating your environment.

```bash
python code/main.py --dataset ItalyPowerDemand --output results/italy.json
```

The script downloads the official training and test splits into `data/`, encodes the labels, normalizes each series independently to `[0, 1]`, chooses a training-class medoid as the exemplar, solves the model, and reports training/test accuracy. The test split is used only for final evaluation. Downloads require internet access on the first run; cached data can be reused with `--data-dir`.

A smaller first run:

```bash
python code/main.py --dataset ItalyPowerDemand --max-size 20 --time-limit 30 --quiet --output results/quickstart.json
```

Use `--help` for all options:

```bash
python code/main.py --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset` | `ItalyPowerDemand` | UCR dataset name |
| `--depth` | Automatic | Smallest positive depth with `2**depth >= number of classes` |
| `--shapelet-length` | `4` | Window length H; must satisfy `1 <= H <= series length` |
| `--alpha` | `1` | Active-leaf penalty, applied only when the tree has more possible leaves than classes |
| `--epsilon` | `1e-5` | Separation margin for right-branch routing |
| `--max-size` | All training samples | Stratified training subset size |
| `--max-length` | Full series | Number of initial points to retain, after full-series normalization |
| `--exemplar-class` | Encoded class 1, or 0 for a single class | Class whose L1 medoid supplies the exemplar |
| `--time-limit` | `600` | Solver time limit in seconds |
| `--seed` / `--threads` | `1` / `1` | Random seed and Gurobi thread count |
| `--data-dir` | `data/` | Download/cache directory |
| `--output` | None | JSON path for settings, metrics, solver diagnostics, and extracted tree |
| `--plot` | Off | Display the learned shapelet at each genuine split |
| `--quiet` | Off | Suppress the solver optimization log |

Only finite, equal-length, **univariate** time series are supported. Missing values, unequal lengths, and multivariate datasets must be handled explicitly before training. Numeric matrices have shape `(samples, timepoints)`; preprocessing also accepts sktime's `(samples, 1, timepoints)` arrays and single-column nested DataFrames. Constant series normalize to zero. Original class labels can be strings or numbers; they are mapped once in sorted order and the mapping is reused at test time.

If `--max-size` leaves too few samples to form a stratified subset, the script explains the error. A class missing from the training set cannot be introduced at test time.

## Synthetic studies

The three entry points share simulation, cross-validation, solver configuration, and result-writing code in [code/experiments.py](../code/experiments.py).

| Script | Varied parameter | Default values | Fitted parameters |
| --- | --- | --- | --- |
| `experiment_depth.py` | Generating tree depth | 1, 2, 3, 4 | Cross-validates depth in 1–3 and alpha in 0.0001, 0.001, 0.01, 0.1, 1, 10 |
| `experiment_shapelet_length.py` | True shapelet length | 3, 6, 9, 12, 15, 18 | Cross-validates H in 2, 4, …, 18 at depth 3 |
| `experiment_series_length.py` | **Series length**, with training size fixed | 50, 100, 200, 400 | Fits depth 1 and H = floor(series length / 5) |

`experiment_series_length.py` varies series length at a fixed training size. Its results explicitly record the series length.

Default experiments use 1,000 simulated series, 100 training samples, five random generating trees, and five populations per tree (25 final fits per parameter). Cross-validation uses five stratified folds. The final solve time limit is 600 seconds, or 3,600 seconds for `experiment_series_length.py`; each CV solve is limited to 60 seconds. Full sweeps can therefore take a long time. At fitted depth 1 there is no leaf penalty in the objective, so the depth study evaluates only the first alpha.

These are **known-exemplar simulations**: the true exemplar is selected from the full synthetic population before the train/test split and supplied to the learner, following the original experimental design. They should not be interpreted as an ordinary held-out exemplar-selection evaluation. Synthetic labels retain the original parity convention (a right turn flips the class sign), rounded distances, and margin rule; inactive generator branches route right.

A short offline smoke run that needs no dataset download:

```bash
python code/experiment_series_length.py --values 6 --tree-repeats 1 --data-repeats 1 --population 40 --train-size 10 --time-limit 5 --quiet --output results/smoke.csv
```

Full default studies:

```bash
python code/experiment_depth.py --quiet
python code/experiment_shapelet_length.py --quiet
python code/experiment_series_length.py --quiet
```

Every script provides `--help`. Use `--values`, `--tree-repeats`, `--data-repeats`, `--population`, `--train-size`, `--folds`, and the relevant candidate grid options to size a study. `--workers` parallelizes parameter values and defaults to 1; `--threads` is the Gurobi thread count **per worker**. Choose both with memory, CPU, and solver license capacity in mind. Models are disposed after each fit.

Each study writes three files:

- `results/<study>.csv`: means and population standard deviations (`ddof=0`), with attempted, successful, failed, and failed-CV-solve counts.
- `results/<study>.runs.csv`: one record per final fit, including the chosen parameters, seed, solver diagnostics, and accuracies.
- `results/<study>.json`: settings, dependency versions, all run records, and individual CV fold scores/solver diagnostics.

Failed final solves remain in the raw results and are excluded from accuracy means. An entirely unsuccessful group has missing accuracy, not zero. Failed CV solves retain the original zero-score policy and are counted explicitly. Ties select the first candidate in grid order. CSV output now consistently uses commas and descriptive column names.

Seeds are derived from the experiment, parameter value, and repetition indices, so changing worker count does not change generated datasets. Solver versions, hardware, time limits, and thread counts can still affect the returned incumbent. Save the result JSON and dependency snapshot when reporting an experiment.

## Smoke runs and reported experiments

The small commands in the quickstart and above are execution checks, not estimates of the paper's reported performance. The full default synthetic commands use the study sizes and parameter grids described above.

For a result you intend to report, retain the command/configuration, raw run records, solver status and gaps, dataset identity and split, and dependency versions. Do not compare a time-limited incumbent with a proven optimum without reporting that distinction. A successful quickstart does not establish statistical performance.

Browse datasets in the [UCR/UEA archive](https://www.timeseriesclassification.com/dataset.php). Dataset terms are set by their providers. Downloads use sktime's public [UCR/UEA dataset API](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.datasets.load_UCR_UEA_dataset.html).
