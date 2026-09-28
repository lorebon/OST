"""Vary the true shapelet length; cross-validate fitted shapelet length."""

from experiments import compute_all, main


def computeAll(parameter):
    """Run the historical default experiment for one parameter value."""
    return compute_all("length", parameter)


if __name__ == "__main__":
    main("length")
