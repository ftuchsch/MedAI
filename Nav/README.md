# ATI Modeling Experiments — Boosting and Linear Ensembles

This track compares selected-feature XGBoost with ridge logistic regression,
shrinkage linear discriminant analysis (LDA), and partial least squares (PLS)
followed by logistic regression. See the [project README](../README.md) for
the task, dataset, recorded results, and validation limitations.

## Setup

From the repository root, use Python 3.12 and an activated virtual environment:

```bash
python -m pip install -r Nav/requirements.txt
```

`requirements-local.txt` remains a compatibility alias to the same dependencies.
Default data and artifact paths resolve relative to the scripts.

## Models

| CLI model name | Protein panel | Approach |
| --- | ---: | --- |
| `Starter XGBoost` | All proteins | Original boosted-tree baseline |
| `Selected XGBoost` | Top 90 | Shallow regularized boosted trees |
| `Selected Ridge LR` | Top 80 | Scaled, L2-regularized logistic regression |
| `Selected Shrinkage LDA` | Top 120 | LDA with covariance shrinkage |
| `Selected PLS LR` | Top 260 | 14 supervised latent components + logistic regression |

All branches retain available clinical covariates. Protein selection occurs
inside each training fold. Parameters and feature counts live in `model.py`.
Repeated stratified CV produces OOF predictions for comparing individual models
and cross-fitted convex blends. OOF log loss selects the final recipe.

## Evaluate, train, predict

Run from the repository root:

```bash
python Nav/evaluate.py --data data/train.csv --out Nav/results/current
python Nav/train.py --data data/train.csv \
  --recipe Nav/results/current/final_recipe.json --out Nav/weights
python Nav/predict.py --data /path/to/new_samples.csv \
  --model-dir Nav/weights --out predictions_nav.csv
```

Evaluation defaults to five folds and three repeats. `--models` accepts quoted
model names; `--blend-models` controls eligibility for blending.
`--xgb-model-seeds` enables seed averaging. For a shorter run:

```bash
python Nav/evaluate.py --data data/train.csv --out Nav/results/quick \
  --folds 3 --repeats 1 --xgb-n-jobs 1 \
  --models "Selected XGBoost" "Selected Ridge LR"
```

Evaluation saves `cv_results.csv`, `base_model_summary.csv`, OOF tables,
`ensemble_summary.csv`, and `final_recipe.json`. Training saves the required
models, feature lists, `ensemble_config.json`, and a training summary.
Selected XGBoost is always retained for starter-interface compatibility.

When a recipe is absent, training uses the historical fixed XGBoost/ridge/PLS
blend (0.45 / 0.40 / 0.15) and logs the fallback. Inference loads
`ensemble_config.json` when present, otherwise the single XGBoost artifact
in `--model-dir`.

```bash
bash Nav/predict.sh /path/to/new_samples.csv predictions_nav.csv
python Nav/evaluate.py --help
python Nav/train.py --help
python Nav/predict.py --help
```

The wrapper uses the caller's active environment; `MEDAI_PYTHON` can override
the interpreter. Output includes `sample_id`, `prob_ati`, and `pred_label`,
component probabilities for ensembles, and `true_label` for labeled inputs.

## Experiment history

- [FINDINGS.md](FINDINGS.md): historical search results and unsuccessful approaches.
- [EXPLAIN.md](EXPLAIN.md): rationale for compact panels and complementary models.
- [research-summary.md](research-summary.md): background from challenge research.
- [results/selected_models/](results/selected_models/): saved five-fold comparison.

These records predate the current recipe-driven evaluator. Re-run evaluation to
assess changes; archived metrics do not guarantee performance on new cohorts.
