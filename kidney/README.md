# ATI Ensemble Pipeline

## Scope

This track implements repeated validation, clinical ablations, and model
ensembling for ATI prediction. See the [project README](../README.md) for the
full project context, preserved modeling tracks, and recorded results.

The implemented approaches are:

- repeated leakage-safe stratified cross-validation
- fold-internal feature selection only
- `XGBoost` anchor models on multiple feature sets
- optional `LightGBM` tree models on the same feature ladders
- elastic-net logistic regression as a diversity model
- clinical and eGFR-only ablations
- second-stage OOF blending / stacking / calibrated stacking
- final saved ensemble bundle for deterministic inference

The implementation is built around the actual local training file:

- `426` rows
- `sample_id`
- binary target `ati`
- `6592` anonymized proteomic features (`feature_XXXX`)
- `age`
- `sex`
- `baseline_egfr_23`

The local dataset is smaller than the planned cohort sizes in the research notes.

## Setup

Use Python 3.12 and an activated virtual environment. From the repository root:

```bash
python -m pip install -r kidney/requirements.txt
# Optional, for lgbm_fs1 and lgbm_fs2:
python -m pip install -r requirements/lightgbm.txt
```

Commands below run from `kidney/`. From the root, prefix script paths with
`kidney/`. Evaluation defaults to five folds and three repeats; use `--repeats 1`
for a shorter run. Early stopping uses validation labels, and ensemble search
is not evaluated with a fully nested outer CV. See the root README for how to
interpret the results.

---

## Feature Ladder

- `FS0 clinical`: `age`, `sex`, `baseline_egfr_23`
- `FS0 clinical no eGFR`: `age`, `sex`
- `FS0 eGFR only`: `baseline_egfr_23`
- `FS1 all`: all proteins plus clinical covariates
- `FS2 top-k`: within-fold univariate filtering, then keep top `1024` or `512`
- `FS3 stable`: within-fold prefiltering plus repeated elastic-net frequency selection

All feature filtering is performed **inside each training fold only**.

---

## Base Models

- `xgb_fs1`: XGBoost on FS1, 3-seed averaging
- `xgb_fs2`: XGBoost on FS2 top-1024, 3-seed averaging
- `xgb_fs3`: XGBoost on FS3 stable features, 3-seed averaging
- `lgbm_fs1`: LightGBM on FS1, 3-seed averaging, optional dependency
- `lgbm_fs2`: LightGBM on FS2 top-1024, 3-seed averaging, optional dependency
- `elastic_fs2`: elastic-net logistic regression on FS2 top-512

Diagnostic-only models:

- `egfr_only_lr`
- `clinical_lr`
- `clinical_no_egfr_lr`

---

## Workflow

```text
data/train.csv
    |
    +--> evaluate.py
    |      - repeated 5-fold StratifiedKFold
    |      - OOF predictions for every base model
    |      - clinical / eGFR stress tests
    |      - OOF blend / stack / calibrated-stack evaluation
    |      - final_recipe.json
    |
    +--> train.py
    |      - fit chosen base models on all rows
    |      - fit saved combiner from OOF base predictions
    |      - save ensemble_bundle.joblib
    |
    +--> predict.py / predict.sh
           - align input columns to saved feature order
           - score each base model
           - combine them with the saved final recipe
```

---

## Step 1 — Evaluate

Run the repeated OOF experiment first:

```bash
python evaluate.py
```

Or with explicit paths:

```bash
python evaluate.py --data ../data/train.csv --out ./results
```

8-core SCC-style example using outer parallelism and LightGBM when installed:

```bash
python evaluate.py \
  --data ../data/train.csv \
  --out ./results_lgbm \
  --repeats 4 \
  --models xgb_fs1 xgb_fs2 lgbm_fs2 \
  --workers 4 \
  --xgb-n-jobs 1 \
  --lgbm-n-jobs 1 \
  --elastic-n-jobs 1
```

Key outputs in `./results/`:

- `base_oof_predictions_long.csv`
- `base_oof_predictions_mean.csv`
- `base_fold_metrics.csv`
- `base_model_summary.csv`
- `ensemble_summary.csv`
- `final_recipe.json`
- `experiment_summary.json`

---

## Step 2 — Train The Final Ensemble

After evaluation:

```bash
python train.py
```

Or:

```bash
python train.py --data ../data/train.csv --results-dir ./results --out ./weights
```

Key outputs in `./weights/`:

- `ensemble_bundle.joblib`
- `ensemble_metadata.json`
- `feature_importance_summary.csv`

---

## Step 3 — Predict

```bash
bash predict.sh /path/to/new_data.csv
```

Or:

```bash
python predict.py --data /path/to/new_data.csv --out predictions.csv
```

Optional:

```bash
python predict.py --data /path/to/new_data.csv --include-base-probs
```

Prediction output columns:

- `sample_id`
- `prob_ati`
- `pred_label`
- one `prob_<base_model>` column per base model when requested
- `true_label` if the input file contains `ati`

---

## Notes

- The pipeline assumes the local `train.csv` is the authoritative dataset.
- `lgbm_fs1` and `lgbm_fs2` require the `lightgbm` Python package, but non-LightGBM runs still work without it.
- `baseline_egfr_23` is treated as a potential proxy trap, so ablations are part of the default evaluation.
- Final ensemble selection uses OOF log loss as the primary metric and AUROC as a secondary check.
- No feature selection is recomputed at inference time.
