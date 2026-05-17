"""Experiment 4: feature ranking on real data (Adult, Bank Marketing, Spambase).

Computes the four scalar scores per feature:
    chi^2,  Sigma_v r_v(c)^2,  plug-in MI,  KSG MI (sklearn)
and reports their Spearman rank correlations.

Then runs forward-selection using the residual-CMI score
    score(X | S) = Sigma_{z, v} r_v(c | z)^2     where z = last-selected feature
and compares against plug-in CMI forward selection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_classif

from signed_mi import (
    bin_numeric,
    config,
    contingency_table,
    pearson_chi_square,
    plug_in_conditional_mutual_information,
    plug_in_mutual_information,
    sum_of_squared_binary_residuals,
)


def encode_categorical_column(series: pd.Series) -> np.ndarray:
    """Encode a categorical column to consecutive int codes.

    Missing values are mapped to their own code.

    Args:
        series: A pandas series of categorical (or string-like) values.

    Returns:
        Integer-coded array of shape (n_samples,).
    """
    series = series.astype("string").fillna("__MISSING__")
    codes = series.astype("category").cat.codes.to_numpy().astype(np.int64)
    assert codes.min() >= 0
    return codes


def load_dataset(
    dataset_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load an OpenML dataset and bin its features for use as integer codes.

    Args:
        dataset_name: One of ``"adult"``, ``"bank"``, ``"spambase"``.

    Returns:
        Tuple ``(raw_features, binned_features, binary_target, feature_names)``.
    """
    if dataset_name == "adult":
        dataset = fetch_openml("adult", version=2, as_frame=True, cache=True)
        raw_features = dataset.data.copy()
        binary_target = (dataset.target == ">50K").astype(int).to_numpy()
    elif dataset_name == "bank":
        dataset = fetch_openml("bank-marketing", version=1, as_frame=True, cache=True)
        raw_features = dataset.data.copy()
        # target is encoded as '1'/'2' classes ('yes'/'no')
        target = dataset.target.astype(str)
        binary_target = (target == "2").astype(int).to_numpy()
    elif dataset_name == "spambase":
        dataset = fetch_openml("spambase", as_frame=True, cache=True)
        raw_features = dataset.data.copy()
        binary_target = dataset.target.astype(int).to_numpy()
    else:
        raise ValueError(dataset_name)

    feature_names = list(raw_features.columns)
    binned_features = np.empty((len(raw_features), len(feature_names)), dtype=np.int64)
    for column_index, feature_name in enumerate(feature_names):
        column = raw_features[feature_name]
        if pd.api.types.is_numeric_dtype(column):
            binned_features[:, column_index] = bin_numeric(
                column.to_numpy(),
                number_of_bins=config.DEFAULT_NUMBER_OF_BINS,
                strategy="quantile",
            )
        else:
            binned_features[:, column_index] = encode_categorical_column(column)
    return raw_features, binned_features, binary_target, feature_names


def per_feature_scores(
    binned_features: np.ndarray,
    binary_target: np.ndarray,
    raw_features: pd.DataFrame,
) -> pd.DataFrame:
    """Score every feature by chi-square, sum r^2, plug-in MI, and KSG MI."""
    score_rows = []
    number_of_classes = int(binary_target.max()) + 1
    for column_index, feature_name in enumerate(raw_features.columns):
        x_values = binned_features[:, column_index]
        number_of_x_bins = int(x_values.max()) + 1
        count_table = contingency_table(
            x_values,
            binary_target,
            number_of_x_bins=number_of_x_bins,
            number_of_classes=number_of_classes,
        )
        chi_square = pearson_chi_square(count_table)
        squared_residual_sum = (
            sum_of_squared_binary_residuals(count_table)
            if number_of_classes == 2
            else float("nan")
        )
        mutual_information = plug_in_mutual_information(count_table)
        score_rows.append(
            dict(
                feature=feature_name,
                chi2=chi_square,
                sum_r2=squared_residual_sum,
                plug_in_mi=mutual_information,
                two_N_mi=2 * len(binary_target) * mutual_information,
            )
        )
    score_table = pd.DataFrame(score_rows)

    # KSG MI: feed continuous columns when numeric, categorical codes otherwise
    ksg_features = np.empty(binned_features.shape, dtype=np.float64)
    discrete_feature_mask = np.zeros(binned_features.shape[1], dtype=bool)
    for column_index, feature_name in enumerate(raw_features.columns):
        column = raw_features[feature_name]
        if pd.api.types.is_numeric_dtype(column):
            ksg_features[:, column_index] = pd.to_numeric(
                column, errors="coerce"
            ).to_numpy()
            missing_mask = ~np.isfinite(ksg_features[:, column_index])
            if missing_mask.any():
                column_median = np.nanmedian(ksg_features[:, column_index])
                ksg_features[missing_mask, column_index] = column_median
            discrete_feature_mask[column_index] = False
        else:
            ksg_features[:, column_index] = encode_categorical_column(column).astype(
                float
            )
            discrete_feature_mask[column_index] = True

    ksg_mutual_information = mutual_info_classif(
        ksg_features,
        binary_target,
        discrete_features=discrete_feature_mask,
        random_state=config.REAL_DATA_RANDOM_STATE,
        n_jobs=1,
    )
    score_table["ksg_mi"] = ksg_mutual_information
    return score_table


def conditional_residual_score(
    x_values: np.ndarray,
    z_values: np.ndarray,
    binary_target: np.ndarray,
    number_of_classes: int = 2,
) -> float:
    """Compute Sum_{z, v} r_v(c | z)^2 with z given by an already-selected feature."""
    number_of_x_bins = int(x_values.max()) + 1
    number_of_z_values = int(z_values.max()) + 1
    score = 0.0
    for z_value in range(number_of_z_values):
        mask = z_values == z_value
        if not mask.any():
            continue
        count_table = contingency_table(
            x_values[mask],
            binary_target[mask],
            number_of_x_bins=number_of_x_bins,
            number_of_classes=number_of_classes,
        )
        score += sum_of_squared_binary_residuals(count_table)
    return score


def conditional_mutual_information(
    x_values: np.ndarray,
    z_values: np.ndarray,
    binary_target: np.ndarray,
    number_of_classes: int = 2,
) -> float:
    """Compute the plug-in conditional MI I(X; C | Z) for a single conditioner Z."""
    number_of_x_bins = int(x_values.max()) + 1
    number_of_z_values = int(z_values.max()) + 1
    joint_counts = np.zeros(
        (number_of_x_bins, number_of_classes, number_of_z_values), dtype=np.int64
    )
    for z_value in range(number_of_z_values):
        mask = z_values == z_value
        if not mask.any():
            continue
        joint_counts[:, :, z_value] = contingency_table(
            x_values[mask],
            binary_target[mask],
            number_of_x_bins=number_of_x_bins,
            number_of_classes=number_of_classes,
        )
    return plug_in_conditional_mutual_information(joint_counts)


def forward_select(
    binned_features: np.ndarray,
    binary_target: np.ndarray,
    feature_names: list[str],
    score_function: str,
    number_of_features_to_select: int = 8,
) -> list[int]:
    """Forward-select up to ``number_of_features_to_select`` features.

    Args:
        binned_features: Integer-coded feature matrix of shape (n_samples, n_features).
        binary_target: Binary target of shape (n_samples,).
        feature_names: Names of the features (currently unused; retained for
            symmetry with the call site).
        score_function: Either ``"sum_r2"`` (residual-based) or ``"plug_mi"``
            (plug-in CMI).
        number_of_features_to_select: Number of features to greedily select.

    Returns:
        Indices into ``binned_features`` columns in selection order.
    """
    del feature_names  # currently unused
    number_of_features = binned_features.shape[1]
    number_of_classes = int(binary_target.max()) + 1
    selected_indices: list[int] = []

    if score_function == "sum_r2":
        marginal_scores = []
        for feature_index in range(number_of_features):
            x_values = binned_features[:, feature_index]
            count_table = contingency_table(
                x_values,
                binary_target,
                number_of_x_bins=int(x_values.max()) + 1,
                number_of_classes=number_of_classes,
            )
            marginal_scores.append(sum_of_squared_binary_residuals(count_table))
    elif score_function == "plug_mi":
        marginal_scores = []
        for feature_index in range(number_of_features):
            x_values = binned_features[:, feature_index]
            count_table = contingency_table(
                x_values,
                binary_target,
                number_of_x_bins=int(x_values.max()) + 1,
                number_of_classes=number_of_classes,
            )
            marginal_scores.append(plug_in_mutual_information(count_table))
    else:
        raise ValueError(score_function)
    selected_indices.append(int(np.argmax(marginal_scores)))

    while len(selected_indices) < number_of_features_to_select:
        previous_selection = selected_indices[-1]
        conditioner = binned_features[:, previous_selection]
        best_feature_index = -1
        best_score = -np.inf
        for feature_index in range(number_of_features):
            if feature_index in selected_indices:
                continue
            x_values = binned_features[:, feature_index]
            if score_function == "sum_r2":
                candidate_score = conditional_residual_score(
                    x_values, conditioner, binary_target, number_of_classes
                )
            else:
                candidate_score = conditional_mutual_information(
                    x_values, conditioner, binary_target, number_of_classes
                )
            if candidate_score > best_score:
                best_score = candidate_score
                best_feature_index = feature_index
        selected_indices.append(best_feature_index)
    return selected_indices


def run_dataset(
    dataset_name: str, results_directory: Path, figures_directory: Path
) -> None:
    print(f"\n=== {dataset_name} ===")
    raw_features, binned_features, binary_target, feature_names = load_dataset(
        dataset_name
    )
    print(
        f"shape={binned_features.shape}, "
        f"class balance={binary_target.mean():.3f}"
    )

    score_table = per_feature_scores(binned_features, binary_target, raw_features)
    score_table = score_table.sort_values("chi2", ascending=False).reset_index(
        drop=True
    )
    score_table.to_csv(
        results_directory / f"exp4_{dataset_name}_scores.csv", index=False
    )

    # Spearman rank correlations among methods (chi2, sum_r2, plug-in MI, KSG)
    method_columns = ["chi2", "sum_r2", "plug_in_mi", "ksg_mi"]
    spearman_correlations = score_table[method_columns].corr(method="spearman")
    spearman_correlations.to_csv(
        results_directory / f"exp4_{dataset_name}_spearman.csv"
    )
    print("Spearman rank correlation:")
    print(spearman_correlations.round(4).to_string())

    # Forward selection on top-K features (capped to keep runtime reasonable).
    number_of_top_features = min(20, binned_features.shape[1])
    top_indices = score_table.index[:number_of_top_features].to_list()
    top_feature_names = [score_table.feature.iloc[i] for i in top_indices]
    name_to_original_column = {
        name: feature_names.index(name) for name in top_feature_names
    }
    top_binned_features = binned_features[
        :, [name_to_original_column[name] for name in top_feature_names]
    ]

    residual_sequence = forward_select(
        top_binned_features,
        binary_target,
        top_feature_names,
        score_function="sum_r2",
        number_of_features_to_select=8,
    )
    plug_in_sequence = forward_select(
        top_binned_features,
        binary_target,
        top_feature_names,
        score_function="plug_mi",
        number_of_features_to_select=8,
    )
    residual_sequence_names = [top_feature_names[i] for i in residual_sequence]
    plug_in_sequence_names = [top_feature_names[i] for i in plug_in_sequence]
    print(f"forward-select (sum r^2): {residual_sequence_names}")
    print(f"forward-select (plug MI): {plug_in_sequence_names}")
    jaccard_similarity = (
        len(set(residual_sequence_names) & set(plug_in_sequence_names))
        / len(set(residual_sequence_names) | set(plug_in_sequence_names))
    )
    print(f"Jaccard top-8: {jaccard_similarity:.3f}")

    pd.DataFrame(
        {
            "step": np.arange(1, len(residual_sequence) + 1),
            "sum_r2": residual_sequence_names,
            "plug_mi": plug_in_sequence_names,
        }
    ).to_csv(
        results_directory / f"exp4_{dataset_name}_forward_selection.csv", index=False
    )

    # Figure: Spearman correlation heatmap
    figure, axis = plt.subplots(figsize=(4.4, 4.0))
    image = axis.imshow(
        spearman_correlations.values, cmap="coolwarm", vmin=0.0, vmax=1.0
    )
    axis.set_xticks(range(len(method_columns)))
    axis.set_yticks(range(len(method_columns)))
    axis.set_xticklabels(method_columns, rotation=30, ha="right")
    axis.set_yticklabels(method_columns)
    for row_index in range(len(method_columns)):
        for column_index in range(len(method_columns)):
            axis.text(
                column_index,
                row_index,
                f"{spearman_correlations.values[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, fraction=0.045)
    axis.set_title(f"Spearman rank correlation\n({dataset_name})")
    figure.tight_layout()
    figure.savefig(figures_directory / f"exp4_{dataset_name}_spearman.pdf")
    figure.savefig(figures_directory / f"exp4_{dataset_name}_spearman.png", dpi=160)
    plt.close(figure)

    # Figure: per-feature score profile (top features) for visual comparison
    top_scores = score_table.head(min(15, len(score_table)))
    figure, axis = plt.subplots(figsize=(7, 4.2))
    x_positions = np.arange(len(top_scores))
    bar_width = 0.22
    columns_to_plot = ["chi2", "plug_in_mi", "ksg_mi"]

    def normalize_to_max(values: np.ndarray) -> np.ndarray:
        return values / max(np.abs(values).max(), 1e-12)

    for column_index, column in enumerate(columns_to_plot):
        axis.bar(
            x_positions + (column_index - 1) * bar_width,
            normalize_to_max(top_scores[column].to_numpy()),
            width=bar_width,
            label=column,
        )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        top_scores["feature"], rotation=45, ha="right", fontsize=8
    )
    axis.set_ylabel("score (normalized to max)")
    axis.set_title(f"Per-feature score (top by $\\chi^2$): {dataset_name}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_directory / f"exp4_{dataset_name}_per_feature.pdf")
    figure.savefig(figures_directory / f"exp4_{dataset_name}_per_feature.png", dpi=160)
    plt.close(figure)


def run(results_directory: Path, figures_directory: Path) -> None:
    for dataset_name in ["adult", "bank", "spambase"]:
        run_dataset(dataset_name, results_directory, figures_directory)


if __name__ == "__main__":
    results_directory, figures_directory = config.ensure_output_directories()
    run(results_directory, figures_directory)
