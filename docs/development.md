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

Add regression checks for meaningful behavior changes. Keep commands and links in the documentation in sync when renaming a script.

`data/`, `results/`, local environments, caches, and solver output are ignored by Git. Keep generated results outside `code/` and do not commit Gurobi license files.
