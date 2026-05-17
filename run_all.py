"""Run all five experiments in order. Outputs go to results/ and figures/."""

from __future__ import annotations

import runpy
import time

from signed_mi import config


EXPERIMENT_SCRIPTS = [
    "exp1_exact_identity.py",
    "exp2_asymptotic_mi.py",
    "exp3_conditional.py",
    "exp4_real_data.py",
    "exp5_interpretability.py",
]


def main() -> None:
    config.ensure_output_directories()
    for script_name in EXPERIMENT_SCRIPTS:
        script_path = config.EXPERIMENTS_DIRECTORY / script_name
        print(f"\n========== {script_name} ==========")
        start_time = time.time()
        runpy.run_path(str(script_path), run_name="__main__")
        print(f"--- {script_name} done in {time.time() - start_time:.1f}s ---")


if __name__ == "__main__":
    main()
