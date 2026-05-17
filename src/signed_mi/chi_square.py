"""Pearson chi-square statistics for marginal and stratified count tables."""

from __future__ import annotations

import numpy as np


def pearson_chi_square(count_table: np.ndarray) -> float:
    """Pearson chi-square statistic for an (X, C) count table.

    Args:
        count_table: Count table of shape (K, M).

    Returns:
        The Pearson chi-square value. Returns 0.0 for an empty table.
    """
    count_table = np.asarray(count_table, dtype=np.float64)
    total_count = count_table.sum()
    if total_count == 0:
        return 0.0
    row_marginals = count_table.sum(axis=1, keepdims=True)
    column_marginals = count_table.sum(axis=0, keepdims=True)
    expected_counts = row_marginals @ column_marginals / total_count
    positive_expected = expected_counts > 0
    contribution = np.zeros_like(count_table)
    contribution[positive_expected] = (
        (count_table[positive_expected] - expected_counts[positive_expected]) ** 2
        / expected_counts[positive_expected]
    )
    return float(contribution.sum())


def stratified_chi_square(slice_tables: list[np.ndarray]) -> float:
    """Stratified chi-square statistic across within-slice tables.

    Args:
        slice_tables: A list of within-slice count tables, one per value of Z.

    Returns:
        chi^2_strat = sum_z chi^2_z.
    """
    return float(sum(pearson_chi_square(table) for table in slice_tables))
