# A Signed Decomposition of Mutual Information via Standardized Residuals

Paper link to be added soon...

Companion code for the paper *A Signed Decomposition of Mutual Information via Standardized Residuals* by Oualid Missaoui.

The package `signed_mi` implements a single object — the per-bin standardized residual profile — that simultaneously functions as:

- a feature-significance test (squared residuals sum **exactly** to Pearson's chi-square),
- a mutual-information estimator (the same sum is asymptotically $2N \cdot \hat I(X; C)$),
- a per-bin attribution diagnostic (the signed profile preserves direction).

The conditional version decomposes $\hat I(X; C \mid Z)$ into a stratified residual profile, linking the framework directly to mRMR- and CMIM-style feature selection.

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

For the SHAP-vs-residual comparison in Experiment 5:

```bash
pip install -e ".[interpretability]"
```

## Quick start

```python
import numpy as np
from signed_mi import (
    contingency_table,
    pearson_chi_square,
    plug_in_mutual_information,
    standardized_residuals_binary,
    sum_of_squared_binary_residuals,
)

x = np.array([0, 0, 1, 1, 2, 2, 2])          # binned feature
c = np.array([0, 1, 0, 0, 1, 1, 0])          # binary class
table = contingency_table(x, c, number_of_x_bins=3, number_of_classes=2)

residuals = standardized_residuals_binary(table)    # signed per-bin profile
chi_square = pearson_chi_square(table)              # significance test
sum_r2 = sum_of_squared_binary_residuals(table)     # equals chi_square exactly
mi = plug_in_mutual_information(table)              # plug-in MI in nats

assert abs(chi_square - sum_r2) < 1e-12             # exact algebraic identity
```

## Public API

| Module | Functions |
|---|---|
| `signed_mi.contingency` | `contingency_table`, `bin_numeric` |
| `signed_mi.residuals` | `standardized_residuals_binary`, `standardized_residuals_multiclass`, `sum_of_squared_binary_residuals`, `sum_of_squared_multiclass_residuals`, `sum_of_squared_conditional_residuals` |
| `signed_mi.chi_square` | `pearson_chi_square`, `stratified_chi_square` |
| `signed_mi.mutual_information` | `plug_in_mutual_information`, `plug_in_conditional_mutual_information` |
| `signed_mi.config` | Shared paths, default bin count, probability clip bounds, reproducibility seeds |

All functions are re-exported at package root so `from signed_mi import …` works.

## Repository layout

```
.
├── pyproject.toml                  # package metadata
├── src/signed_mi/                  # the library
│   ├── chi_square.py
│   ├── config.py
│   ├── contingency.py
│   ├── mutual_information.py
│   └── residuals.py
├── experiments/                    # paper experiments
│   ├── exp1_exact_identity.py      # verify sum r^2 = chi^2 numerically
│   ├── exp2_asymptotic_mi.py       # 2N*I asymptotic equivalence
│   ├── exp3_conditional.py         # conditional decomposition (Theorem 4)
│   ├── exp4_real_data.py           # Adult / Bank / Spambase feature ranking
│   └── exp5_interpretability.py    # per-bin attribution on Adult
├── results/                        # CSV outputs of the experiments
├── paper/
│   ├── figures/                    # generated PDF/PNG figures
│   └── residual_mi_paper.{tex,pdf}
└── run_all.py                      # run the five experiments in order
```

## Reproducing the experiments

```bash
python run_all.py
```

Or run a single experiment:

```bash
python experiments/exp1_exact_identity.py
```

Outputs land in [results/](results/) (CSV tables) and [paper/figures/](paper/figures/) (PDF and PNG). All seeds are centralized in [src/signed_mi/config.py](src/signed_mi/config.py).

| # | Experiment | Verifies |
|---|---|---|
| 1 | Exact identity | $\sum_v r_v(c)^2 = \chi^2$ to floating-point precision, across $10\,000$ random $K \times 2$ tables |
| 2 | Asymptotic MI | $\sum_v r_v(c)^2 - 2N\hat I \to 0$ at rate $N^{-1/2}$ |
| 3 | Conditional | Stratified identity $\sum_{z,v} r_v(c\mid z)^2 = \chi^2_{\mathrm{strat}}$ and its CMI equivalent |
| 4 | Real data | Spearman rank-correlation of chi², sum $r^2$, plug-in MI, KSG MI on three OpenML datasets |
| 5 | Interpretability | Signed residual profiles and SHAP comparison on Adult |
