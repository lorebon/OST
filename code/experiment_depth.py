"""Vary the generating tree depth; cross-validate fitted depth and leaf penalty."""

from experiments import compute_all, main


def computeAll(parameter):
    """Run the historical default experiment for one parameter value."""
    return compute_all("depth", parameter)


if __name__ == "__main__":
    main("depth")
