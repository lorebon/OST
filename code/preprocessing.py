"""Input validation, stable class encoding, and per-series min-max scaling."""

import numpy as np


def as_series_array(X):
    """Accept a numeric matrix, sktime numpy3D, or univariate nested DataFrame.

    Return finite, equal-length series as (samples, timepoints).
    """
    if hasattr(X, "iloc") and len(X) and X.shape[1] == 1 and np.ndim(X.iloc[0, 0]) == 1:
        try:
            X = np.stack([np.asarray(row, dtype=float) for row in X.iloc[:, 0]])
        except ValueError as exc:
            raise ValueError("Time series must have equal lengths.") from exc
    try:
        X = np.asarray(X, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Expected numeric, equal-length univariate time series.") from exc
    if X.ndim == 3 and X.shape[1] == 1:
        X = X[:, 0, :]
    if X.ndim != 2 or min(X.shape) == 0:
        raise ValueError("X must be a nonempty (samples, timepoints) matrix.")
    if not np.isfinite(X).all():
        raise ValueError("Time series must not contain NaN or infinite values.")
    return X


def positive_integer(value, name):
    """Validate integer dimensions without silently truncating floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


class SeriesMinMaxScaler:
    """Scale each complete series independently to [0, 1]; constants become 0.

    This is per-series normalization, not column-wise scaling across samples.
    Test data never contribute statistics to training data.
    """

    def fit_transform(self, X):
        X = as_series_array(X)
        self.n_features_in_ = X.shape[1]
        return self.transform(X)

    def transform(self, X):
        X = as_series_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Training and test series must have the same original length.")
        minimum = X.min(axis=1, keepdims=True)
        span = X.max(axis=1, keepdims=True) - minimum
        return (X - minimum) / np.where(span == 0, 1, span)


def _truncate(X, max_length):
    if max_length is None:
        return X
    positive_integer(max_length, "maxLength")
    if max_length > X.shape[1]:
        raise ValueError("maxLength exceeds the original series length.")
    return X[:, :max_length]


def _labels(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("y must contain one label per series.")
    if any(label is None or label != label for label in y):
        raise ValueError("Labels must not be missing.")
    return y


def preprocessTrain(X, y, maxSize=None, maxLength=None, random_state=1):
    """Normalize, optionally subsample, and keep the first maxLength points.

    Return X, encoded y, ordered original classes, encoded classes, and scaler.
    Labels may be strings or numbers; sorted order defines their encoding.
    Pass the returned original classes unchanged to preprocessTest.
    Scaling happens before truncation, as in the original workflow.
    """
    X = as_series_array(X)
    y = _labels(y, len(X))
    classes, y = np.unique(y, return_inverse=True)
    scaler = SeriesMinMaxScaler()
    X = scaler.fit_transform(X)
    if maxSize is not None:
        positive_integer(maxSize, "maxSize")
        if maxSize > len(X):
            raise ValueError("maxSize exceeds the number of training samples.")
        if maxSize < len(X):
            from sklearn.model_selection import train_test_split

            try:
                X, _, y, _ = train_test_split(
                    X, y, train_size=maxSize, stratify=y, random_state=random_state
                )
            except ValueError as exc:
                raise ValueError(f"Cannot stratify the requested training subset: {exc}") from exc
            if len(np.unique(y)) != len(classes):
                raise ValueError("The requested subset omits a class; increase maxSize.")
    return _truncate(X, maxLength), y, classes, np.arange(len(classes)), scaler


def preprocessTest(X, y, y_set, scaler, maxLength=None):
    """Apply the training class mapping and identical normalization/truncation."""
    X = as_series_array(X)
    y = _labels(y, len(X))
    if isinstance(y_set, set):
        raise ValueError("Pass ordered classes from preprocessTrain, not an unordered set.")
    mapping = {label: index for index, label in enumerate(y_set)}
    try:
        encoded = np.array([mapping[label] for label in y], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Test label {exc.args[0]!r} was absent from training data.") from exc
    return _truncate(scaler.transform(X), maxLength), encoded
