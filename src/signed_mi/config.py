"""Shared configuration for the signed_mi package and the experiments.

Centralizes filesystem layout, numerical defaults, and reproducibility seeds
so that experiment scripts do not duplicate these constants.
"""

from __future__ import annotations

from pathlib import Path


# ---------- repository layout ----------------------------------------------

PACKAGE_DIRECTORY: Path = Path(__file__).resolve().parent
SOURCE_DIRECTORY: Path = PACKAGE_DIRECTORY.parent
REPOSITORY_ROOT: Path = SOURCE_DIRECTORY.parent

RESULTS_DIRECTORY: Path = REPOSITORY_ROOT / "results"
FIGURES_DIRECTORY: Path = REPOSITORY_ROOT / "paper" / "figures"
EXPERIMENTS_DIRECTORY: Path = REPOSITORY_ROOT / "experiments"


# ---------- numerical defaults ---------------------------------------------

DEFAULT_NUMBER_OF_BINS: int = 10
PROBABILITY_CLIP_LOWER: float = 0.01
PROBABILITY_CLIP_UPPER: float = 0.99


# ---------- reproducibility seeds ------------------------------------------

EXACT_IDENTITY_SEED: int = 20260504
ASYMPTOTIC_MI_SEED: int = 42
CONDITIONAL_SEED: int = 7
REAL_DATA_RANDOM_STATE: int = 0
INTERPRETABILITY_RANDOM_STATE: int = 0


def ensure_output_directories() -> tuple[Path, Path]:
    """Create the results and figures directories if needed.

    Returns:
        A pair (results_directory, figures_directory) of paths guaranteed
        to exist on disk.
    """
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIRECTORY, FIGURES_DIRECTORY
