# Development and validation

[Home](../README.md) · [Method](method.md) · [Experiments](experiments.md) · [Development](development.md)

## Dependency files

For an installation without exact version pins, `python -m pip install -r requirements.txt` installs the direct runtime dependencies with version lower bounds. Those lower bounds are not a claim that every possible combination has been tested. `requirements-dev.txt` adds pytest and Ruff; the tested snapshot already contains both. The snapshot was validated on Python 3.12 under Windows.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -m "not gurobi" -q
python -m pytest -m gurobi -q
ruff check .
ruff format --check .
```

The solver tests are intentionally tiny and skip when Gurobi or a license is unavailable. They exercise constant data, full-length windows, last-window selection, and consistency between MIP assignments and predictions. Core checks cover label mapping, preprocessing, medoid selection, distance bounds, deep pruning, synthetic reproducibility, and result handling. GitHub Actions runs formatting, lint, and the license-independent tests. It does not run the paper's full experiments.

## Layout and imports

All Python implementation files and experiment entry points live together in `code/`. Run entry points as `python code/<script>.py` from the repository root. There is no `__init__.py`, package metadata, editable installation, or build step. `pytest.ini` adds `code/` to the test import path; the scripts themselves use ordinary local imports.

Keep mathematical changes separate from presentation changes when possible. Describe anything that could alter predictions, optimization feasibility, or experiment results in [CHANGES.md](../CHANGES.md), and add a regression check for meaningful behavior changes. Keep commands and links in the documentation in sync when renaming a script.

`data/`, `results/`, local environments, caches, and solver output are ignored by Git. Keep generated results outside `code/` and do not commit Gurobi license files.

## Previous script names

All scripts moved from `Optimal Shapelets Tree/` to `code/`. The experiment entry points also changed names:

| Previous filename | Current filename |
| --- | --- |
| `ground_depth.py` | `experiment_depth.py` |
| `ground_length.py` | `experiment_shapelet_length.py` |
| `learning_test.py` | `experiment_series_length.py` |

Update saved commands and imports to the new paths. The renamed experiment files retain `computeAll(parameter)` for interactive use. The reorganization changes neither solver logic nor experiment settings, seeds, or output naming.
