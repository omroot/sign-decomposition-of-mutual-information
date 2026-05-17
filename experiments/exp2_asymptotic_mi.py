"""Experiment 2: characterize Sigma_v r_v(c)^2 - 2N hat I as a function of N and theta.

Parametric family p_theta(X, C):
    X uniform over K bins, p(C=1 | X=v) = 0.5 + theta * a_v, with a_v
    deterministic anchors in [-1, 1] summing to zero. theta=0 => independence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signed_mi import (
    config,
    pearson_chi_square,
    plug_in_mutual_information,
    sum_of_squared_binary_residuals,
)


def build_joint_probabilities(
    number_of_x_bins: int, theta: float
) -> np.ndarray:
    """Build the joint p(X, C) for the parametric family of Experiment 2."""
    anchors = np.linspace(-1.0, 1.0, number_of_x_bins)
    x_marginal = np.full(number_of_x_bins, 1.0 / number_of_x_bins)
    positive_class_given_x = np.clip(
        0.5 + theta * anchors,
        config.PROBABILITY_CLIP_LOWER,
        config.PROBABILITY_CLIP_UPPER,
    )
    joint_probabilities = np.zeros((number_of_x_bins, 2))
    joint_probabilities[:, 1] = x_marginal * positive_class_given_x
    joint_probabilities[:, 0] = x_marginal * (1.0 - positive_class_given_x)
    return joint_probabilities


def sample_table(
    joint_probabilities: np.ndarray, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    counts = rng.multinomial(sample_size, joint_probabilities.flatten())
    return counts.reshape(joint_probabilities.shape).astype(np.int64)


def run(results_directory: Path, figures_directory: Path) -> None:
    number_of_x_bins = 5
    theta_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75]
    sample_size_grid = [100, 1000, 10_000, 100_000, 1_000_000]
    default_repetitions = 1000
    large_sample_repetitions = 200  # cap reps at the largest N to keep runtime sane
    rng = np.random.default_rng(config.ASYMPTOTIC_MI_SEED)

    summary_rows: list[dict] = []
    for theta in theta_grid:
        joint_probabilities = build_joint_probabilities(number_of_x_bins, theta)
        for sample_size in sample_size_grid:
            number_of_repetitions = (
                large_sample_repetitions
                if sample_size >= 1_000_000
                else default_repetitions
            )
            chi_square_array = np.empty(number_of_repetitions)
            two_n_times_mi = np.empty(number_of_repetitions)
            squared_residual_sum = np.empty(number_of_repetitions)
            for repetition_index in range(number_of_repetitions):
                count_table = sample_table(joint_probabilities, sample_size, rng)
                chi_square_array[repetition_index] = pearson_chi_square(count_table)
                squared_residual_sum[repetition_index] = (
                    sum_of_squared_binary_residuals(count_table)
                )
                two_n_times_mi[repetition_index] = (
                    2 * sample_size * plug_in_mutual_information(count_table)
                )
            asymptotic_gap = squared_residual_sum - two_n_times_mi
            identity_difference = np.abs(chi_square_array - squared_residual_sum)
            summary_rows.append(
                dict(
                    theta=theta,
                    N=sample_size,
                    reps=number_of_repetitions,
                    mean_sum_r2=float(squared_residual_sum.mean()),
                    mean_2NI=float(two_n_times_mi.mean()),
                    mean_gap=float(asymptotic_gap.mean()),
                    median_abs_gap=float(np.median(np.abs(asymptotic_gap))),
                    p95_abs_gap=float(np.quantile(np.abs(asymptotic_gap), 0.95)),
                    median_rel_gap=float(
                        np.median(
                            np.abs(asymptotic_gap)
                            / np.maximum(two_n_times_mi, 1e-9)
                        )
                    ),
                    max_ident_diff=float(identity_difference.max()),
                )
            )
            print(
                f"theta={theta:.2f} N={sample_size:>7d}  "
                f"mean sum r^2={squared_residual_sum.mean():.4f}  "
                f"mean 2N*I={two_n_times_mi.mean():.4f}  "
                f"median|gap|={np.median(np.abs(asymptotic_gap)):.4f}"
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_directory / "exp2_asymptotic_mi.csv", index=False)

    # Figure 1: median absolute gap vs N for each theta (log-log).
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    colormap = plt.get_cmap("viridis")
    for theta_index, theta in enumerate(theta_grid):
        subset = summary[summary.theta == theta].sort_values("N")
        axis.loglog(
            subset.N,
            subset.median_abs_gap,
            "o-",
            color=colormap(theta_index / max(1, len(theta_grid) - 1)),
            label=fr"$\theta={theta}$",
        )
    sample_size_array = np.array(sample_size_grid, dtype=float)
    null_curve = summary[summary.theta == 0.0].sort_values("N")
    if len(null_curve) >= 2:
        anchor_sample_size = null_curve.N.iloc[0]
        anchor_gap = null_curve.median_abs_gap.iloc[0]
        reference_curve = (
            anchor_gap * np.sqrt(anchor_sample_size) / np.sqrt(sample_size_array)
        )
        axis.loglog(
            sample_size_array,
            reference_curve,
            "k--",
            alpha=0.6,
            label=r"$N^{-1/2}$ (anchored at null)",
        )
    axis.set_xlabel("sample size $N$")
    axis.set_ylabel(r"median $|\sum_v r_v(c)^2 - 2N\hat I|$")
    axis.set_title("Asymptotic equivalence: gap between residual sum and $2N\\hat I$")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures_directory / "exp2_gap_vs_N.pdf")
    figure.savefig(figures_directory / "exp2_gap_vs_N.png", dpi=160)
    plt.close(figure)

    # Figure 2: relative gap vs N for each theta. theta=0 omitted (2NI -> 0).
    figure, axis = plt.subplots(figsize=(6.0, 4.0))
    for theta_index, theta in enumerate(theta_grid):
        if theta == 0.0:
            continue
        subset = summary[summary.theta == theta].sort_values("N")
        axis.semilogx(
            subset.N,
            subset.median_rel_gap,
            "o-",
            color=colormap(theta_index / max(1, len(theta_grid) - 1)),
            label=fr"$\theta={theta}$",
        )
    axis.set_xlabel("sample size $N$")
    axis.set_ylabel(r"median $|\sum_v r_v(c)^2 - 2N\hat I| / 2N\hat I$")
    axis.set_title("Relative asymptotic gap by effect strength")
    axis.legend(fontsize=8)
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(figures_directory / "exp2_relative_gap.pdf")
    figure.savefig(figures_directory / "exp2_relative_gap.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    results_directory, figures_directory = config.ensure_output_directories()
    run(results_directory, figures_directory)
