"""Plug-in (empirical) mutual information and conditional mutual information."""

from __future__ import annotations

import numpy as np


def plug_in_mutual_information(count_table: np.ndarray) -> float:
    """Plug-in (empirical) mutual information of (X, C) from a count table.

    Args:
        count_table: Count table of shape (K, M).

    Returns:
        Mutual information in nats. Returns 0.0 for an empty table.
    """
    count_table = np.asarray(count_table, dtype=np.float64)
    total_count = count_table.sum()
    if total_count == 0:
        return 0.0
    joint_probabilities = count_table / total_count
    x_marginals = joint_probabilities.sum(axis=1, keepdims=True)
    class_marginals = joint_probabilities.sum(axis=0, keepdims=True)
    independent_probabilities = x_marginals * class_marginals
    nonzero_mask = (joint_probabilities > 0) & (independent_probabilities > 0)
    return float(
        np.sum(
            joint_probabilities[nonzero_mask]
            * (
                np.log(joint_probabilities[nonzero_mask])
                - np.log(independent_probabilities[nonzero_mask])
            )
        )
    )


def plug_in_conditional_mutual_information(joint_counts: np.ndarray) -> float:
    """Plug-in conditional mutual information I(X; C | Z).

    Args:
        joint_counts: Count tensor of shape (K, M, |Z|) with axes (X, C, Z).

    Returns:
        Conditional mutual information in nats.
    """
    joint_counts = np.asarray(joint_counts, dtype=np.float64)
    total_count = joint_counts.sum()
    if total_count == 0:
        return 0.0
    conditional_mutual_information = 0.0
    for z_value in range(joint_counts.shape[2]):
        slice_table = joint_counts[:, :, z_value]
        slice_total = slice_table.sum()
        if slice_total <= 0:
            continue
        slice_probability = slice_total / total_count
        conditional_mutual_information += (
            slice_probability * plug_in_mutual_information(slice_table)
        )
    return conditional_mutual_information
