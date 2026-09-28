# Optimal Shapelets Tree

Research code for **Optimal shapelets tree for time series interpretable classification**, by Lorenzo Bonasera and Stefano Gualandi, *EURO Journal on Computational Optimization*, 12 (2024), 100091. [Read the paper](https://doi.org/10.1016/j.ejco.2024.100091).

OST learns an interpretable time-series classifier using a mixed-integer optimization model. At each branch, the model selects a contiguous window in an exemplar and a contiguous window position in the input series, then compares their Manhattan (L1) distance with a learned threshold.

This repository consists of directly runnable Python scripts. No package installation or build step is required. The implementation and experiment entry points live together in `code/`.

The original implementation is preserved on the [`old` branch](https://github.com/lorebon/OST/tree/old). The `main` branch contains the maintained version.

## Setup

Use Python 3.12 for the tested environment. From the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the tested dependency versions:

```bash
python -m pip install -r requirements-tested.txt
```

Training requires a working **Gurobi license** appropriate for the model size. See the [Gurobi setup guide](https://docs.gurobi.com/projects/optimizer/en/current/). Preprocessing and prediction do not require Gurobi. See [development notes](docs/development.md) for alternative dependency installs and checks.

## Quickstart

From the repository root, run a small UCR example:

```bash
python code/main.py --dataset ItalyPowerDemand --max-size 20 --time-limit 30 --quiet --output results/quickstart.json
```

The first run downloads the official dataset splits into `data/`. The script trains on a stratified subset, evaluates on the official test split, and saves settings, metrics, solver diagnostics, and the learned tree. Inputs must be finite, equal-length, univariate time series. This small run checks the workflow; it does not reproduce a paper result.

For a synthetic smoke run with no dataset download:

```bash
python code/experiment_series_length.py --values 6 --tree-repeats 1 --data-repeats 1 --population 40 --train-size 10 --time-limit 5 --quiet --output results/smoke.csv
```

Every entry point supports `--help`. Full experiment commands, defaults, and output formats are in the [experiment guide](docs/experiments.md).

## Find your way around

| Task | Start here |
| --- | --- |
| Understand the model and core functions | [Method and code guide](docs/method.md) |
| Train on UCR data or run a synthetic study | [Experiment guide](docs/experiments.md) |
| Inspect the mathematical formulation | [code/training.py](code/training.py) |
| Run tests or modify the scripts | [Development notes](docs/development.md) |

```text
code/                 Model, preprocessing, prediction, and runnable experiments
docs/                 Method, experiment instructions, and development notes
tests/                Core regressions and small Gurobi integration tests
requirements*.txt     Runtime dependencies and tested environment
CITATION.cff          Machine-readable paper citation
```

## Citation and license

```bibtex
@article{bonasera2024optimal,
  title   = {Optimal shapelets tree for time series interpretable classification},
  author  = {Bonasera, Lorenzo and Gualandi, Stefano},
  journal = {EURO Journal on Computational Optimization},
  volume  = {12},
  pages   = {100091},
  year    = {2024},
  doi     = {10.1016/j.ejco.2024.100091}
}
```

Code is distributed under the [MIT license](LICENSE). Dataset terms are set by their respective providers.
