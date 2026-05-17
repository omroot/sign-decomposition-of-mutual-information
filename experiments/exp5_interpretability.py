"""Experiment 5: per-bin interpretability case study on Adult.

Produces:
  (a) signed unconditional residual profiles r_v(c) for selected features
  (b) conditional residual heatmap r_v(c | z) for one (X, Z) pair
  (c) per-bin attribution comparison vs gradient-boosted-tree feature importance
  (d) per-feature SHAP scatter overlaid with the residual profile (if shap is
      installed)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

from signed_mi import (
    bin_numeric,
    config,
    contingency_table,
    standardized_residuals_binary,
)

try:
    import shap as _shap_module
except ImportError:
    _shap_module = None


INTERPRETABLE_FEATURES = ["education-num", "hours-per-week", "age", "capital-gain"]


def _check_shap() -> None:
    """Raise an informative error if the optional shap dependency is missing."""
    if _shap_module is None:
        raise ImportError(
            "shap is required for this function. Install it with `pip install shap`."
        )


def load_adult() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the Adult dataset and a binary >50K target."""
    dataset = fetch_openml("adult", version=2, as_frame=True, cache=True)
    raw_features = dataset.data.copy()
    high_income_target = (dataset.target == ">50K").astype(int).to_numpy()
    return raw_features, high_income_target


def bin_with_labels(
    values: np.ndarray, number_of_bins: int = config.DEFAULT_NUMBER_OF_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a numeric vector and return labelled bins.

    Heavily zero-inflated features get a dedicated zero bin and the strictly
    positive part is quantile-binned separately.

    Args:
        values: Numeric values, shape (n_samples,).
        number_of_bins: Target number of bins.

    Returns:
        Tuple ``(bin_indices, bin_centers, bin_edges)``.
    """
    is_finite = np.isfinite(values)
    finite_values = values[is_finite]
    zero_share = (finite_values == 0).mean() if finite_values.size else 0.0

    if zero_share > 0.5 and (finite_values > 0).sum() >= 50:
        positive_values = finite_values[finite_values > 0]
        quantile_grid = np.linspace(0.0, 1.0, number_of_bins)
        positive_edges = np.unique(np.quantile(positive_values, quantile_grid))
        bin_edges = np.concatenate(
            [[-np.inf, 0.0 + 1e-9], positive_edges[1:-1], [np.inf]]
        )
    else:
        quantile_grid = np.linspace(0.0, 1.0, number_of_bins + 1)
        bin_edges = np.unique(np.quantile(finite_values, quantile_grid))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    bin_indices = np.digitize(values, bin_edges[1:-1], right=False)
    number_of_actual_bins = int(bin_indices.max()) + 1
    bin_centers = []
    for bin_index in range(number_of_actual_bins):
        mask = (bin_indices == bin_index) & is_finite
        if mask.any():
            bin_centers.append(float(np.median(values[mask])))
        else:
            bin_centers.append(float("nan"))
    return bin_indices.astype(np.int64), np.array(bin_centers), bin_edges


def plot_unconditional_profiles(
    raw_features: pd.DataFrame,
    binary_target: np.ndarray,
    output_path: Path,
    feature_names: list[str] = INTERPRETABLE_FEATURES,
) -> None:
    """Plot signed unconditional residual profiles for selected features."""
    figure, axes = plt.subplots(
        1, len(feature_names), figsize=(13, 3.5), sharey=True
    )
    for axis, feature_name in zip(axes, feature_names):
        values = raw_features[feature_name].to_numpy()
        bin_indices, bin_centers, _ = bin_with_labels(
            values, number_of_bins=config.DEFAULT_NUMBER_OF_BINS
        )
        number_of_x_bins = int(bin_indices.max()) + 1
        count_table = contingency_table(
            bin_indices,
            binary_target,
            number_of_x_bins=number_of_x_bins,
            number_of_classes=2,
        )
        residuals = standardized_residuals_binary(count_table, positive_class_index=1)
        bar_colors = [
            "tab:blue" if residual >= 0 else "tab:red" for residual in residuals
        ]
        axis.bar(
            np.arange(len(residuals)),
            residuals,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.4,
        )
        axis.axhline(0, color="black", linewidth=0.6)
        axis.axhline(1.96, linestyle="--", color="gray", linewidth=0.6)
        axis.axhline(-1.96, linestyle="--", color="gray", linewidth=0.6)
        axis.set_xticks(np.arange(len(residuals)))
        bin_labels = [
            f"{center:.2g}" if np.isfinite(center) else "" for center in bin_centers
        ]
        axis.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=8)
        axis.set_title(
            f"{feature_name}\n$\\sum r^2={np.sum(residuals * residuals):.0f}$"
        )
        axis.set_xlabel("bin (median value)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$r_v(c=>50K)$")
    figure.suptitle(
        r"Signed unconditional residual profiles (Adult, $c$ = income $>50K$)"
    )
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(figure)


def plot_conditional_heatmap(
    raw_features: pd.DataFrame, binary_target: np.ndarray, output_path: Path
) -> None:
    """Plot the conditional residual heatmap r_v(c | z) for one (X, Z) pair."""
    x_values = raw_features["education-num"].to_numpy()
    z_values = raw_features["hours-per-week"].to_numpy()
    x_bin_indices, x_bin_centers, _ = bin_with_labels(
        x_values, number_of_bins=config.DEFAULT_NUMBER_OF_BINS
    )
    z_bin_indices, z_bin_centers, _ = bin_with_labels(z_values, number_of_bins=5)
    number_of_x_bins = int(x_bin_indices.max()) + 1
    number_of_z_bins = int(z_bin_indices.max()) + 1
    residual_matrix = np.full((number_of_x_bins, number_of_z_bins), np.nan)
    for z_bin in range(number_of_z_bins):
        mask = z_bin_indices == z_bin
        if not mask.any():
            continue
        count_table = contingency_table(
            x_bin_indices[mask],
            binary_target[mask],
            number_of_x_bins=number_of_x_bins,
            number_of_classes=2,
        )
        residual_matrix[:, z_bin] = standardized_residuals_binary(
            count_table, positive_class_index=1
        )
    color_limit = np.nanmax(np.abs(residual_matrix))
    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    color_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)
    image = axis.imshow(
        residual_matrix, cmap="RdBu_r", aspect="auto", norm=color_norm, origin="lower"
    )
    axis.set_xticks(np.arange(number_of_z_bins))
    axis.set_xticklabels(
        [f"{center:.0f}" if np.isfinite(center) else "" for center in z_bin_centers]
    )
    axis.set_yticks(np.arange(number_of_x_bins))
    axis.set_yticklabels(
        [f"{center:.0f}" if np.isfinite(center) else "" for center in x_bin_centers]
    )
    axis.set_xlabel("hours-per-week (median in bin)")
    axis.set_ylabel("education-num (median in bin)")
    for row_index in range(number_of_x_bins):
        for column_index in range(number_of_z_bins):
            value = residual_matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    color="white" if abs(value) > color_limit * 0.55 else "black",
                    fontsize=8,
                )
    figure.colorbar(image, ax=axis, label=r"$r_v(c=>50K\,|\,z)$")
    axis.set_title(
        "Conditional residual profile $r_v(c\\,|\\,z)$\n"
        "X = education-num, Z = hours-per-week (Adult)"
    )
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(figure)


def _ordinal_encode_with_median_imputation(
    raw_features: pd.DataFrame,
) -> np.ndarray:
    """Ordinal-encode all columns and impute unknowns with the column median."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoded = encoder.fit_transform(raw_features.astype(str)).astype(np.float32)
    for column_index in range(encoded.shape[1]):
        column = encoded[:, column_index]
        column[column < 0] = np.median(column[column >= 0])
        encoded[:, column_index] = column
    return encoded


def plot_residual_vs_gbm_importance(
    raw_features: pd.DataFrame, binary_target: np.ndarray, output_path: Path
) -> pd.DataFrame:
    """Compare per-feature sum r^2 ranking with GBM feature importance."""
    feature_names = list(raw_features.columns)
    encoded_features = _ordinal_encode_with_median_imputation(raw_features)

    classifier = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=config.INTERPRETABILITY_RANDOM_STATE,
    )
    classifier.fit(encoded_features, binary_target)
    gbm_feature_importance = classifier.feature_importances_

    sum_of_squared_residuals_per_feature: list[float] = []
    for column_index, feature_name in enumerate(feature_names):
        column = raw_features[feature_name]
        if pd.api.types.is_numeric_dtype(column):
            bin_indices = bin_numeric(
                column.to_numpy(),
                number_of_bins=config.DEFAULT_NUMBER_OF_BINS,
            )
        else:
            bin_indices = encoded_features[:, column_index].astype(np.int64)
            bin_indices = bin_indices - bin_indices.min()
        number_of_x_bins = int(bin_indices.max()) + 1
        count_table = contingency_table(
            bin_indices,
            binary_target,
            number_of_x_bins=number_of_x_bins,
            number_of_classes=2,
        )
        residuals = standardized_residuals_binary(count_table, positive_class_index=1)
        sum_of_squared_residuals_per_feature.append(
            float(np.sum(residuals * residuals))
        )

    importance_table = pd.DataFrame(
        {
            "feature": feature_names,
            "sum_r2": sum_of_squared_residuals_per_feature,
            "gbm_importance": gbm_feature_importance,
        }
    )
    importance_table = importance_table.sort_values(
        "sum_r2", ascending=False
    ).reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(8, 4))
    x_positions = np.arange(len(importance_table))
    axis.bar(
        x_positions - 0.2,
        importance_table.sum_r2 / importance_table.sum_r2.max(),
        width=0.4,
        color="tab:blue",
        label=r"$\sum_v r_v^2$ (norm.)",
    )
    axis.bar(
        x_positions + 0.2,
        importance_table.gbm_importance / importance_table.gbm_importance.max(),
        width=0.4,
        color="tab:orange",
        label="GBM importance (norm.)",
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        importance_table.feature, rotation=45, ha="right", fontsize=8
    )
    axis.set_ylabel("normalized importance")
    axis.set_title("Model-free residual sum vs model-based importance (Adult)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(figure)
    return importance_table


def plot_shap_vs_residual(
    raw_features: pd.DataFrame,
    binary_target: np.ndarray,
    output_path: Path,
    feature_names: tuple[str, ...] = (
        "education-num",
        "age",
        "hours-per-week",
        "fnlwgt",
    ),
) -> None:
    """Per-feature SHAP scatter (gray) overlaid with residual profile (red).

    Requires the optional ``shap`` dependency.
    """
    _check_shap()

    encoded_features = _ordinal_encode_with_median_imputation(raw_features)
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        random_state=config.INTERPRETABILITY_RANDOM_STATE,
    )
    classifier.fit(encoded_features, binary_target)
    explainer = _shap_module.TreeExplainer(classifier)
    sample_indices = np.random.default_rng(
        config.INTERPRETABILITY_RANDOM_STATE
    ).choice(
        len(encoded_features),
        size=min(2000, len(encoded_features)),
        replace=False,
    )
    shap_values = explainer.shap_values(encoded_features[sample_indices])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    figure, axes = plt.subplots(1, len(feature_names), figsize=(14, 3.6))
    feature_to_column = {name: index for index, name in enumerate(raw_features.columns)}
    for axis, feature_name in zip(axes, feature_names):
        if feature_name not in feature_to_column:
            axis.set_visible(False)
            continue
        column_index = feature_to_column[feature_name]
        raw_sample_values = raw_features.iloc[sample_indices, column_index].to_numpy()
        try:
            numeric_sample_values = pd.to_numeric(
                pd.Series(raw_sample_values), errors="coerce"
            ).to_numpy()
        except Exception:
            numeric_sample_values = encoded_features[sample_indices, column_index]
        axis.scatter(
            numeric_sample_values,
            shap_values[:, column_index],
            s=6,
            color="gray",
            alpha=0.5,
        )
        axis.axhline(0, color="lightcoral", linewidth=0.6)
        axis.set_xlabel(feature_name)
        axis.set_ylabel("SHAP")
        axis.set_title(feature_name, fontsize=10)

        full_values = raw_features[feature_name].to_numpy()
        try:
            numeric_full_values = pd.to_numeric(
                pd.Series(full_values), errors="coerce"
            ).to_numpy()
        except Exception:
            numeric_full_values = full_values.astype(float)
        has_finite_values = np.isfinite(numeric_full_values).any()
        if has_finite_values and pd.api.types.is_numeric_dtype(
            raw_features[feature_name]
        ):
            bin_indices, bin_centers, _ = bin_with_labels(
                numeric_full_values, number_of_bins=config.DEFAULT_NUMBER_OF_BINS
            )
            number_of_x_bins = int(bin_indices.max()) + 1
            count_table = contingency_table(
                bin_indices,
                binary_target,
                number_of_x_bins=number_of_x_bins,
                number_of_classes=2,
            )
            residuals = standardized_residuals_binary(
                count_table, positive_class_index=1
            )
            order = np.argsort(bin_centers)
            right_axis = axis.twinx()
            right_axis.plot(
                bin_centers[order], residuals[order], "o-", color="tab:red", linewidth=1.8
            )
            right_axis.set_ylabel(r"$r_v(c)$", color="tab:red")
            right_axis.tick_params(axis="y", labelcolor="tab:red")

    figure.suptitle(
        "Exp 5c: SHAP scatter (gray) vs residual profile (red, right axis)"
    )
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".pdf"))
    figure.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(figure)


def run(results_directory: Path, figures_directory: Path) -> None:
    raw_features, binary_target = load_adult()
    print(
        f"Adult: shape={raw_features.shape}, "
        f"class balance (>50K) = {binary_target.mean():.3f}"
    )
    plot_unconditional_profiles(
        raw_features, binary_target, figures_directory / "exp5_uncond_profiles"
    )
    plot_conditional_heatmap(
        raw_features, binary_target, figures_directory / "exp5_cond_heatmap"
    )
    importance_table = plot_residual_vs_gbm_importance(
        raw_features, binary_target, figures_directory / "exp5_residual_vs_gbm"
    )
    importance_table.to_csv(
        results_directory / "exp5_residual_vs_gbm.csv", index=False
    )
    print(importance_table.to_string(index=False))
    if _shap_module is None:
        print("shap not installed; skipping SHAP-vs-residual figure.")
    else:
        plot_shap_vs_residual(
            raw_features, binary_target, figures_directory / "exp5_shap_vs_residual"
        )
        print("SHAP-vs-residual figure produced.")


if __name__ == "__main__":
    results_directory, figures_directory = config.ensure_output_directories()
    run(results_directory, figures_directory)
