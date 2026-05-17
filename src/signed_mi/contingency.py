"""Contingency-table construction and numeric feature binning.

Utilities for turning raw observations into the integer-coded count tables
expected by the chi-square, residual, and mutual-information estimators.
"""

from __future__ import annotations

import numpy as np


def contingency_table(
    x_values: np.ndarray,
    class_values: np.ndarray,
    number_of_x_bins: int,
    number_of_classes: int,
) -> np.ndarray:
    """Build the joint count table of (X, C).

    Args:
        x_values: Integer-coded X observations, shape (n_samples,).
        class_values: Integer-coded C observations, shape (n_samples,).
        number_of_x_bins: Number of distinct X values (rows of the table).
        number_of_classes: Number of distinct C values (columns of the table).

    Returns:
        Count table of shape (number_of_x_bins, number_of_classes) with rows
        indexed by X and columns indexed by C.
    """
    flat_indices = (
        x_values.astype(np.int64) * number_of_classes + class_values.astype(np.int64)
    )
    counts = np.bincount(
        flat_indices, minlength=number_of_x_bins * number_of_classes
    )
    return counts.reshape(number_of_x_bins, number_of_classes)


def bin_numeric(
    values: np.ndarray, number_of_bins: int = 10, strategy: str = "quantile"
) -> np.ndarray:
    """Bin a numeric vector into equally-frequent or equally-spaced bins.

    Heavily zero-inflated columns (more than half of the finite values equal
    to zero, with at least 50 strictly positive values) get a dedicated zero
    bin while the strictly positive part is quantile-binned separately.

    Args:
        values: Numeric values, shape (n_samples,). NaNs are allowed.
        number_of_bins: Target number of bins.
        strategy: Either ``"quantile"`` or ``"uniform"``.

    Returns:
        Integer array of bin indices in ``[0, K)`` with the same shape as
        ``values``. NaNs are routed to a separate bin (the largest index).
        Constant columns collapse to a single bin.

    Raises:
        ValueError: If ``strategy`` is not one of the supported values.
    """
    values = np.asarray(values, dtype=np.float64)
    is_finite = np.isfinite(values)
    bin_indices = np.full(values.shape, -1, dtype=np.int64)
    if not is_finite.any():
        bin_indices[:] = 0
        return bin_indices
    finite_values = values[is_finite]
    if strategy == "quantile":
        zero_share = (finite_values == 0).mean()
        has_zero_inflation = zero_share > 0.5 and (finite_values > 0).sum() >= 50
        if has_zero_inflation:
            positive_values = finite_values[finite_values > 0]
            quantile_grid = np.linspace(0.0, 1.0, max(number_of_bins, 2))
            positive_edges = np.unique(np.quantile(positive_values, quantile_grid))
            bin_edges = np.unique(
                np.concatenate([[finite_values.min()], [0.0 + 1e-9], positive_edges])
            )
        else:
            quantile_grid = np.linspace(0.0, 1.0, number_of_bins + 1)
            bin_edges = np.unique(np.quantile(finite_values, quantile_grid))
    elif strategy == "uniform":
        bin_edges = np.unique(
            np.linspace(finite_values.min(), finite_values.max(), number_of_bins + 1)
        )
    else:
        raise ValueError(f"unknown strategy {strategy}")
    if bin_edges.size <= 1:
        bin_indices[:] = 0
        return bin_indices
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    finite_bins = np.digitize(finite_values, bin_edges[1:-1], right=False)
    bin_indices[is_finite] = finite_bins
    bin_indices[~is_finite] = finite_bins.max() + 1 if (~is_finite).any() else 0
    _, consecutive_indices = np.unique(bin_indices, return_inverse=True)
    return consecutive_indices.reshape(values.shape)
