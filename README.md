# Anomaly Detection on Financial Transactions

A comparative study of 8 fraud-detection methods on a highly imbalanced credit-card dataset
(~0.17% positives), with a focus on what unsupervised and deep-learning approaches can really
deliver against a strong supervised baseline.

---

## Why I built this

I started this project on my own time to go beyond the usual "drop XGBoost on the dataset and
move on" workflow. I wanted to:

- understand **unsupervised** anomaly-detection methods in depth — not just call them, but
  implement and tune them, and see where they actually break down,
- get hands-on with **deep one-class learning** (Autoencoder, VAE, Deep SVDD), implemented
  from scratch in PyTorch rather than using a black-box library,
- run a study that is **statistically defensible** — multiple seeds, paired significance
  tests, false-discovery-rate correction — instead of a single-number comparison.

The Credit Card Fraud Detection dataset (ULB / Kaggle) is the canonical benchmark for this
problem: 284 807 transactions, only 492 frauds (0.172%), with 28 PCA-anonymized features.

---

## Research questions

1. **Do deep methods actually beat Isolation Forest**, the standard strong baseline for
   anomaly detection?
2. **How does each method degrade** as the number of fraud labels available at training time
   shrinks? (robustness to imbalance)
3. **What is the most honest evaluation metric** on a 0.17% positive-rate problem?

---

## Methods

| # | Model | Family | Library |
|---|---|---|---|
| 1 | Logistic Regression | Supervised baseline | scikit-learn |
| 2 | XGBoost             | Supervised ceiling  | xgboost      |
| 3 | Isolation Forest    | Statistical ensemble | scikit-learn |
| 4 | Local Outlier Factor (LOF) | Density-based | scikit-learn |
| 5 | One-Class SVM       | Boundary-based      | scikit-learn |
| 6 | Autoencoder         | Deep reconstruction | PyTorch      |
| 7 | Variational Autoencoder | Deep probabilistic | PyTorch  |
| 8 | Deep SVDD           | Deep one-class      | PyTorch (Ruff et al. 2018) |

All unsupervised models are trained **only on normal transactions** from the train split.
Hyperparameters are tuned with **Optuna** (60 trials per model) optimizing PR-AUC on the
validation set. All experiments use a stratified 80/10/10 train/val/test split, repeated
across 5 random seeds.

---

## Headline results

**Experiment 1 — baseline comparison (PR-AUC, mean ± std over 5 seeds)**

| Model | PR-AUC | F1 | Precision@100 |
|---|---|---|---|
| **XGBoost**            | **0.86 ± 0.04** | 0.86 ± 0.03 | 0.43 ± 0.02 |
| Logistic Regression    | 0.77 ± 0.07     | 0.82 ± 0.04 | 0.43 ± 0.02 |
| LOF                    | 0.52 ± 0.09     | 0.59 ± 0.06 | 0.37 ± 0.04 |
| Deep SVDD              | 0.48 ± 0.22     | 0.54 ± 0.21 | 0.29 ± 0.09 |
| VAE                    | 0.40 ± 0.10     | 0.45 ± 0.10 | 0.29 ± 0.04 |
| Autoencoder            | 0.40 ± 0.13     | 0.44 ± 0.13 | 0.28 ± 0.06 |
| Isolation Forest       | 0.39 ± 0.01     | 0.40 ± 0.04 | 0.29 ± 0.01 |
| One-Class SVM          | 0.30 ± 0.02     | 0.43 ± 0.05 | 0.31 ± 0.02 |

**Key findings:**

- **XGBoost dominates** when fraud labels are available, as expected.
- **LOF is the strongest unsupervised method**, beating all three deep models on PR-AUC. Its
  local nature matches the multi-cluster geometry of fraud transactions visible in the EDA.
- The **deep methods underdeliver** relative to their reputation — they sit between Isolation
  Forest and LOF, with much higher seed variance (Deep SVDD std = 0.22). They are
  tuning-sensitive rather than fundamentally weak.
- **ROC-AUC is misleading**: every method scores ≥ 0.86 on ROC-AUC, which would suggest they
  are all comparable. PR-AUC reveals the real ranking. *PR-AUC is the right primary metric
  on extreme imbalance.*
- **Statistical significance**: 16 of 28 pairwise differences are significant under a paired
  t-test with Benjamini-Hochberg FDR correction. The Wilcoxon counterpart finds none —
  expected with only 5 seeds (its smallest achievable p-value is ~0.0625).

**Experiment 2** — varying the fraction of fraud labels available at training time (0%, 10%,
25%, 50%, 100%) shows supervised models losing 5–10 PR-AUC points and becoming unstable below
10% labels, while unsupervised models stay flat by construction. Even at 10% labels, XGBoost
still beats every unsupervised method — but its variance grows enough that an unsupervised
method becomes a sensible operational choice when labels are noisy or delayed.

Full analysis with figures, ranking heatmaps, and pairwise significance: see
[`notebooks/02_results_analysis.ipynb`](notebooks/02_results_analysis.ipynb).

---

## Reproducing the study

```bash
# 1. install
pip install -e ".[dev]"

# 2. download the dataset (requires ~/.kaggle/kaggle.json)
make download

# 3. (optional) re-tune hyperparameters with Optuna
make tune

# 4. run experiments — Exp 1: 8 models × 5 seeds, ~30–60 min on CPU
make run
make run-exp2

# 5. generate report figures from the saved CSVs
make figures

# 6. browse runs in the MLflow UI
make mlflow-ui   # then open http://localhost:5000

# 7. unit tests
make test
```

All hyperparameters and paths live in `configs/*.yaml`. There are no hardcoded paths or magic
numbers in the code.

---

## Project structure

```
configs/        YAML configs (paths, seeds, hyperparameters)
src/            Library code — flat, no sub-packages
  data.py        load / split / scale / fraud-subsample
  models.py      8 model classes with a shared duck-typed interface
  evaluation.py  metrics, bootstrap CI, McNemar, BH-correction, threshold selection
  plots.py       PR curves, score distributions, training-loss curves
  utils.py       set_seed, load_config, get_logger
scripts/        Entry-point CLIs
  download_data.py     Kaggle download + integrity check
  tune_hyperparams.py  Optuna search → writes back to configs/models.yaml
  run_experiment.py    Exp 1 + Exp 2, MLflow logging, optional MongoDB sink
  make_figures.py      Build report figures from outputs/*.csv
notebooks/      01_eda.ipynb, 02_results_analysis.ipynb
tests/          pytest sanity checks (4 tests)
outputs/        CSV summaries + figures (committed)
mlruns/         MLflow tracking (gitignored, SQLite backend)
```

---

## What this project taught me

Beyond the technical results, this was the first project where I designed the engineering
side as carefully as the modeling side. Specific things I now do differently:

- **Architecture** — splitting code into a real Python package (`src/` with explicit module
  boundaries) instead of doing everything in a single notebook. The code is now testable,
  reusable, and reviewable.
- **PyTorch & OOP** — implementing the deep models from scratch (Autoencoder, VAE, Deep SVDD)
  and exposing them through a common `fit / score_samples / predict` interface taught me to
  think in terms of contracts between objects rather than ad-hoc functions. Deep SVDD in
  particular forced me to read the original paper closely and reproduce details (center
  initialization, zero-clamping) that no library provides.
- **Experiment tracking with MLflow** — moving from "I'll re-run it if I lose the numbers" to
  having every run logged, queryable, and inspectable in a UI. Optional MongoDB sink for
  durable run history.
- **Hyperparameter tuning with Optuna** — running ~60 trials per unsupervised model with PR-AUC
  on validation as the objective, then writing best-params back to a YAML so the main runs are
  reproducible.
- **Statistical rigor** — paired t-tests, Wilcoxon as a robustness check, Benjamini-Hochberg
  FDR correction across 28 model pairs. Acknowledging when the test is underpowered (e.g.
  Wilcoxon at n=5) instead of pretending otherwise.
- **Reproducibility hygiene** — every random call goes through a centralized `set_seed`, no
  hardcoded paths, configs in YAML, deterministic data splits. I learned this the hard way
  when a non-seeded GPU op silently changed results between runs.
- **Engineering for the long tail** — handling Apple-Silicon-specific OpenMP segfaults,
  managing MLflow + SQLite backends, choosing where to log silently and where to fail loudly
  (e.g. MongoDB outage should not kill an experiment).
- **Honesty about results** — writing up a study where the deep models *do not win* was
  uncomfortable but more useful than chasing a result. The whole point of the comparison was
  to find out, not to confirm a hypothesis.

---

## References

- Ruff, L. et al. (2018). *Deep One-Class Classification.* ICML.
- Schölkopf, B. et al. (2001). *Estimating the Support of a High-Dimensional Distribution.*
  Neural Computation.
- Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). *Isolation Forest.* ICDM.
- Breunig, M. et al. (2000). *LOF: Identifying Density-Based Local Outliers.* SIGMOD.
- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced
  Classification.* CIDM. (Source of the dataset.)
