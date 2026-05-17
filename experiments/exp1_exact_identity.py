"""Experiment 1: verify the exact identity Sigma_v r_v(c)^2 = chi^2.

Generates random K x 2 contingency tables across a (K, N) grid and reports
the absolute and relative discrepancy. Saves a CSV summary and a histogram
PDF to results/ and figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signed_mi import (
    config,
    pearson_chi_square,
    sum_of_squared_binary_residuals,
)


def sample_random_table(
    number_of_x_bins: int, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a random K x 2 table from a randomly drawn joint over (X, C)."""
    x_marginal = rng.dirichlet(np.ones(number_of_x_bins))
    positive_class_marginal = rng.uniform(0.05, 0.95)
    positive_class_given_x = np.clip(
        positive_class_marginal + rng.uniform(-0.4, 0.4, size=number_of_x_bins),
        0.01,
        0.99,
    )
    joint_probabilities = np.empty((number_of_x_bins, 2))
    joint_probabilities[:, 1] = x_marginal * positive_class_given_x
    joint_probabilities[:, 0] = x_marginal * (1.0 - positive_class_given_x)
    counts = rng.multinomial(sample_size, joint_probabilities.flatten()).reshape(
        number_of_x_bins, 2
    )
    return counts.astype(np.int64)


def run(results_directory: Path, figures_directory: Path) -> None:
    rng = np.random.default_rng(config.EXACT_IDENTITY_SEED)
    x_bin_grid = [2, 5, 10, 50]
    sample_size_grid = [100, 1000, 10_000, 100_000]
    tables_per_cell = 625  # 16 cells x 625 = 10_000 tables total

    summary_rows: list[dict] = []
    all_absolute_diffs: list[float] = []
    all_relative_diffs: list[float] = []
    for number_of_x_bins in x_bin_grid:
        for sample_size in sample_size_grid:
            absolute_diffs: list[float] = []
            relative_diffs: list[float] = []
            chi_square_values: list[float] = []
            for _ in range(tables_per_cell):
                count_table = sample_random_table(number_of_x_bins, sample_size, rng)
                chi_square = pearson_chi_square(count_table)
                squared_residual_sum = sum_of_squared_binary_residuals(count_table)
                diff = abs(chi_square - squared_residual_sum)
                absolute_diffs.append(diff)
                relative_diffs.append(diff / max(chi_square, 1e-300))
                chi_square_values.append(chi_square)
            summary_rows.append(
                dict(
                    K=number_of_x_bins,
                    N=sample_size,
                    n_tables=tables_per_cell,
                    max_abs_diff=float(np.max(absolute_diffs)),
                    median_abs_diff=float(np.median(absolute_diffs)),
                    max_rel_diff=float(np.max(relative_diffs)),
                    median_rel_diff=float(np.median(relative_diffs)),
                    median_chi2=float(np.median(chi_square_values)),
                )
            )
            all_absolute_diffs.extend(absolute_diffs)
            all_relative_diffs.extend(relative_diffs)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_directory / "exp1_exact_identity.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nGlobal max relative discrepancy: {max(all_relative_diffs):.3e}")
    print(f"Global max absolute discrepancy: {max(all_absolute_diffs):.3e}")

    relative_diffs_array = np.array(all_relative_diffs)
    positive_relative_diffs = relative_diffs_array[relative_diffs_array > 0]
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    if positive_relative_diffs.size:
        axis.hist(
            np.log10(positive_relative_diffs),
            bins=40,
            edgecolor="black",
            linewidth=0.4,
        )
    axis.set_xlabel(r"$\log_{10}\,|\sum_v r_v(c)^2 - \chi^2|/\chi^2$")
    axis.set_ylabel("count")
    axis.set_title(
        f"Exact identity discrepancy across $10{{,}}000$ random $K\\times 2$ tables\n"
        f"max relative gap = {relative_diffs_array.max():.2e}"
    )
    axis.axvline(
        np.log10(np.finfo(float).eps),
        color="tab:red",
        linestyle="--",
        label=r"machine $\epsilon$",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_directory / "exp1_exact_identity.pdf")
    figure.savefig(figures_directory / "exp1_exact_identity.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    results_directory, figures_directory = config.ensure_output_directories()
    run(results_directory, figures_directory)
