"""Vary series length at a fixed training size; fit a depth-one tree."""

from experiments import compute_all, main


def computeAll(parameter):
    """Run the historical default experiment for one parameter value."""
    return compute_all("learning", parameter)


if __name__ == "__main__":
    main("learning")
