"""Experiment 3: verification of the conditional decomposition (Theorem 4).

Family p_{theta, |Z|}(X, Z, C):
    Z uniform over |Z| values.
    Within slice z, X uniform over K=2 bins, p(C=1 | X=v, Z=z) = 0.5 + theta_z * a_v
    with a_v = (-1)^v and theta_z = theta * sin(2 pi z / |Z|).
Closed-form CMI computable from the slab probabilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signed_mi import (
    config,
    plug_in_conditional_mutual_information,
    stratified_chi_square,
    sum_of_squared_conditional_residuals,
)

NUMBER_OF_X_BINS = 2


def build_joint_probabilities(theta: float, number_of_z_values: int) -> np.ndarray:
    """Build the joint p(X, C, Z) tensor of shape (K, 2, |Z|)."""
    joint_probabilities = np.zeros((NUMBER_OF_X_BINS, 2, number_of_z_values))
    anchors = np.array([-1.0, 1.0])
    for z_value in range(number_of_z_values):
        # alternating-sign weight in [-1, 1] across slices, never identically zero
        theta_z = (
            theta
            * (1.0 - 2.0 * (z_value % 2))
            * (0.5 + 0.5 * z_value / max(number_of_z_values - 1, 1))
        )
        positive_class_given_xz = np.clip(
            0.5 + theta_z * anchors,
            config.PROBABILITY_CLIP_LOWER,
            config.PROBABILITY_CLIP_UPPER,
        )
        x_marginal = np.full(NUMBER_OF_X_BINS, 1.0 / NUMBER_OF_X_BINS)
        slab = np.zeros((NUMBER_OF_X_BINS, 2))
        slab[:, 1] = x_marginal * positive_class_given_xz
        slab[:, 0] = x_marginal * (1.0 - positive_class_given_xz)
        joint_probabilities[:, :, z_value] = slab / number_of_z_values
    return joint_probabilities


def closed_form_conditional_mutual_information(
    theta: float, number_of_z_values: int
) -> float:
    """Compute I(X; C | Z) analytically from the slab probabilities (no sampling)."""
    joint_probabilities = build_joint_probabilities(theta, number_of_z_values)
    total = 0.0
    for z_value in range(joint_probabilities.shape[2]):
        slab = joint_probabilities[:, :, z_value]
        slice_probability = slab.sum()
        if slice_probability <= 0:
            continue
        conditional_probabilities = slab / slice_probability
        x_marginal = conditional_probabilities.sum(axis=1, keepdims=True)
        class_marginal = conditional_probabilities.sum(axis=0, keepdims=True)
        nonzero_mask = conditional_probabilities > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            term = conditional_probabilities[nonzero_mask] * (
                np.log(conditional_probabilities[nonzero_mask])
                - np.log(x_marginal @ class_marginal)[nonzero_mask]
            )
        total += slice_probability * float(np.sum(term))
    return total


def sample_joint(
    joint_probabilities: np.ndarray, sample_size: int, rng: np.random.Generator
) -> np.ndarray:
    counts = rng.multinomial(sample_size, joint_probabilities.flatten())
    return counts.reshape(joint_probabilities.shape).astype(np.int64)


def split_into_slice_tables(joint_counts: np.ndarray) -> list[np.ndarray]:
    return [joint_counts[:, :, z] for z in range(joint_counts.shape[2])]


def run(results_directory: Path, figures_directory: Path) -> None:
    z_cardinalities = [2, 4, 8, 16]
    theta_grid = [0.0, 0.1, 0.25, 0.5]
    sample_size_grid = [1000, 10_000, 100_000]
    number_of_repetitions = 500
    rng = np.random.default_rng(config.CONDITIONAL_SEED)

    summary_rows: list[dict] = []
    for number_of_z_values in z_cardinalities:
        for theta in theta_grid:
            population_cmi = closed_form_conditional_mutual_information(
                theta, number_of_z_values
            )
            joint_probabilities = build_joint_probabilities(theta, number_of_z_values)
            for sample_size in sample_size_grid:
                identity_difference = np.empty(number_of_repetitions)
                asymptotic_gap = np.empty(number_of_repetitions)
                empirical_cmi = np.empty(number_of_repetitions)
                squared_residual_sum = np.empty(number_of_repetitions)
                for repetition_index in range(number_of_repetitions):
                    joint_counts = sample_joint(
                        joint_probabilities, sample_size, rng
                    )
                    slice_tables = split_into_slice_tables(joint_counts)
                    stratified_chi = stratified_chi_square(slice_tables)
                    squared_residual_sum[repetition_index] = (
                        sum_of_squared_conditional_residuals(slice_tables)
                    )
                    identity_difference[repetition_index] = abs(
                        stratified_chi - squared_residual_sum[repetition_index]
                    )
                    cmi = plug_in_conditional_mutual_information(joint_counts)
                    empirical_cmi[repetition_index] = cmi
                    asymptotic_gap[repetition_index] = (
                        squared_residual_sum[repetition_index] - 2 * sample_size * cmi
                    )
                summary_rows.append(
                    dict(
                        Z=number_of_z_values,
                        theta=theta,
                        N=sample_size,
                        reps=number_of_repetitions,
                        cmi_pop=population_cmi,
                        mean_cmi_emp=float(empirical_cmi.mean()),
                        max_exact_diff=float(identity_difference.max()),
                        median_exact_diff=float(np.median(identity_difference)),
                        median_abs_gap=float(np.median(np.abs(asymptotic_gap))),
                        mean_gap=float(asymptotic_gap.mean()),
                        median_rel_gap=float(
                            np.median(
                                np.abs(asymptotic_gap)
                                / np.maximum(2 * sample_size * empirical_cmi, 1e-9)
                            )
                        ),
                    )
                )
                print(
                    f"|Z|={number_of_z_values:2d} theta={theta:.2f} "
                    f"N={sample_size:>6d}  "
                    f"max identity diff={identity_difference.max():.2e}  "
                    f"median |gap|={np.median(np.abs(asymptotic_gap)):.4f}  "
                    f"CMI(pop)={population_cmi:.4f}"
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_directory / "exp3_conditional.csv", index=False)

    # Figure 1: exact-identity discrepancy across all (|Z|, theta, N) cells.
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    log_diffs = np.log10(np.maximum(summary.max_exact_diff.values, 1e-300))
    axis.hist(log_diffs, bins=20, edgecolor="black", linewidth=0.4)
    axis.set_xlabel(
        r"$\log_{10}\,\max\,|\sum_{z,v} r_v(c|z)^2 - \chi^2_{\mathrm{strat}}|$"
    )
    axis.set_ylabel(f"count (over {len(summary)} cells)")
    axis.set_title("Exact identity (conditional): floating-point only discrepancy")
    figure.tight_layout()
    figure.savefig(figures_directory / "exp3_exact_identity_cond.pdf")
    figure.savefig(figures_directory / "exp3_exact_identity_cond.png", dpi=160)
    plt.close(figure)

    # Figure 2: median absolute gap vs N, faceted by |Z|, lines = theta.
    figure, axes = plt.subplots(
        1, len(z_cardinalities), figsize=(11, 3.4), sharey=True
    )
    colormap = plt.get_cmap("plasma")
    for axis, number_of_z_values in zip(axes, z_cardinalities):
        for theta_index, theta in enumerate(theta_grid):
            subset = summary[
                (summary.Z == number_of_z_values) & (summary.theta == theta)
            ].sort_values("N")
            axis.loglog(
                subset.N,
                subset.median_abs_gap.clip(lower=1e-6),
                "o-",
                color=colormap(theta_index / max(1, len(theta_grid) - 1)),
                label=fr"$\theta={theta}$",
            )
        axis.set_title(fr"$|\mathcal{{Z}}|={number_of_z_values}$")
        axis.set_xlabel("$N$")
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"median $|\sum r^2 - 2N\hat I(X;C\mid Z)|$")
    axes[-1].legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "Conditional asymptotic gap by sample size, effect, and stratum count"
    )
    figure.tight_layout()
    figure.savefig(figures_directory / "exp3_cond_gap_vs_N.pdf")
    figure.savefig(figures_directory / "exp3_cond_gap_vs_N.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    results_directory, figures_directory = config.ensure_output_directories()
    run(results_directory, figures_directory)
