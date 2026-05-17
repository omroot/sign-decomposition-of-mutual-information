"""Signed and standardized residuals for the (X, C) contingency table.

Implements:
  - row-conditional standardized residuals r_v(c) for binary C (Eq. 5)
  - per-cell residuals tilde r_v(c) for multi-class C (Eq. 16)
  - sum-of-squared variants, both marginal and conditional on a stratifier Z
"""

from __future__ import annotations

import numpy as np


def standardized_residuals_binary(
    count_table: np.ndarray,
    positive_class_index: int = 1,
    class_probability_floor: float = 0.0,
) -> np.ndarray:
    """Row-conditional standardized residual r_v(c) for binary C.

    Computes
        r_v(c) = sqrt(N_v) * (p(c|v) - p(c)) / sqrt(p(c) * (1 - p(c))),
    where N_v is the count of X = v.

    Args:
        count_table: Count table of shape (K, 2).
        positive_class_index: Column to treat as the positive class.
        class_probability_floor: Tolerance below which the marginal class
            probability is considered degenerate; residuals are then zero.

    Returns:
        Array of residuals of length K. Bins with zero count contribute zero.
    """
    count_table = np.asarray(count_table, dtype=np.float64)
    if count_table.shape[1] != 2:
        raise ValueError("standardized_residuals_binary expects a K x 2 table")
    row_totals = count_table.sum(axis=1)
    total_count = count_table.sum()
    if total_count == 0:
        return np.zeros(count_table.shape[0])
    positive_class_probability = (
        count_table[:, positive_class_index].sum() / total_count
    )
    is_degenerate = (
        positive_class_probability <= class_probability_floor
        or positive_class_probability >= 1 - class_probability_floor
    )
    if is_degenerate:
        return np.zeros(count_table.shape[0])
    safe_row_totals = np.maximum(row_totals, 1)
    class_probability_given_bin = np.where(
        row_totals > 0,
        count_table[:, positive_class_index] / safe_row_totals,
        0.0,
    )
    residuals = (
        np.sqrt(row_totals)
        * (class_probability_given_bin - positive_class_probability)
        / np.sqrt(positive_class_probability * (1.0 - positive_class_probability))
    )
    residuals[row_totals == 0] = 0.0
    return residuals


def standardized_residuals_multiclass(count_table: np.ndarray) -> np.ndarray:
    """Per-cell standardized residual tilde r_v(c) (Eq. 16).

    Args:
        count_table: Count table of shape (K, M).

    Returns:
        Array of residuals of shape (K, M). Cells whose row total or column
        marginal is zero contribute zero.
    """
    count_table = np.asarray(count_table, dtype=np.float64)
    row_totals = count_table.sum(axis=1, keepdims=True)
    total_count = count_table.sum()
    class_probabilities = count_table.sum(axis=0, keepdims=True) / total_count
    safe_row_totals = np.maximum(row_totals, 1)
    class_probability_given_bin = count_table / safe_row_totals
    safe_sqrt_class_probabilities = np.sqrt(
        np.where(class_probabilities > 0, class_probabilities, 1.0)
    )
    residuals = (
        np.sqrt(row_totals)
        * (class_probability_given_bin - class_probabilities)
        / safe_sqrt_class_probabilities
    )
    residuals = np.where(row_totals > 0, residuals, 0.0)
    residuals = np.where(class_probabilities > 0, residuals, 0.0)
    return residuals


def sum_of_squared_binary_residuals(
    count_table: np.ndarray, positive_class_index: int = 1
) -> float:
    """Return sum_v r_v(c)^2 for a K x 2 count table.

    Args:
        count_table: Count table of shape (K, 2).
        positive_class_index: Column to treat as the positive class.

    Returns:
        The sum of squared row-conditional residuals.
    """
    residuals = standardized_residuals_binary(
        count_table, positive_class_index=positive_class_index
    )
    return float(np.sum(residuals * residuals))


def sum_of_squared_multiclass_residuals(count_table: np.ndarray) -> float:
    """Return sum_{v,c} tilde r_v(c)^2 for a K x M count table.

    Args:
        count_table: Count table of shape (K, M).

    Returns:
        The sum of squared per-cell residuals.
    """
    residuals = standardized_residuals_multiclass(count_table)
    return float(np.sum(residuals * residuals))


def sum_of_squared_conditional_residuals(
    slice_tables: list[np.ndarray],
    positive_class_index: int = 1,
    mode: str = "binary",
) -> float:
    """Sum of squared conditional residuals across slices.

    For ``mode="binary"``, returns sum over (z, v) of r_v(c|z)^2.
    For ``mode="multiclass"``, returns sum over (z, v, c) of tilde r_v(c|z)^2.

    Args:
        slice_tables: A list of within-slice count tables, one per value of Z.
        positive_class_index: Column to treat as the positive class
            (binary mode only).
        mode: Either ``"binary"`` (K x 2 tables) or ``"multiclass"``
            (K x M tables).

    Returns:
        The sum of squared conditional residuals across all slices.

    Raises:
        ValueError: If ``mode`` is not one of the supported values.
    """
    if mode == "binary":
        return float(
            sum(
                sum_of_squared_binary_residuals(
                    table, positive_class_index=positive_class_index
                )
                for table in slice_tables
            )
        )
    if mode == "multiclass":
        return float(
            sum(sum_of_squared_multiclass_residuals(table) for table in slice_tables)
        )
    raise ValueError(f"unknown mode: {mode}")
