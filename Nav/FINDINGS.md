# Findings

- The local dataset in `../data/train.csv` is `426 x 6597`, not the larger `1,500`-row split referenced in the research notes.
- The target is `ati` and the hidden evaluation metric is `log loss`.
- The local training CSV has:
  - `226` negatives and `200` positives
  - `6,592` protein columns
  - `3` clinical columns: `age`, `sex`, `baseline_egfr_23`
  - no missing values
  - no exact duplicate rows
  - no duplicate `sample_id` values
- Near-duplicate screening did not reveal a suspicious leakage cluster:
  - no rounded duplicates
  - nearest-neighbor distances are not unusually tiny
- The starter baseline is materially under-tuned for this `p >> n` setting.

## Baselines Measured

- Starter XGBoost, 5-fold CV:
  - mean AUC: about `0.752`
  - mean log loss: about `0.626`
- Starter L1 logistic baseline, 5-fold CV:
  - mean AUC: about `0.710`
  - mean log loss: about `0.650`

## What Helped

- Fold-safe top-k protein selection helped a lot.
- The useful regime is small:
  - about `80` to `100` selected proteins for the tree and ridge branches
- Best single-model family so far:
  - very shallow regularized XGBoost with top-k selected proteins
- Best repeated-CV blend so far:
  - `45%` selected-feature XGBoost
  - `40%` selected-feature ridge logistic regression
  - `15%` selected-feature PLS logistic regression
- The best XGBoost branch got better after retuning:
  - depth `1` instead of depth `2`
  - more trees (`500`)
  - wider column subsampling (`0.5`)
  - slightly lighter regularization
- The best PLS branch uses:
  - top `260` proteins + 3 clinical features
  - `14` supervised latent components
  - logistic regression with `C=0.5`
- PLS is not good alone on log loss, but it adds useful diversity at low weight.
- Shrinkage LDA helped earlier, but the PLS branch is now better than LDA in the final ensemble.
- Nested calibration did not help:
  - sigmoid calibration worsened log loss
  - isotonic calibration was much worse on this small dataset
- Two-stage tree re-ranking did not help:
  - simple univariate fold-safe filtering beat the more complex reranking variants
- A small LDA branch helped, but only at low weight:
  - aggressive `30%` LDA weighting looked great on one seed and then lost on repeated CV
  - a moderate `20%` LDA branch survived repeated-CV checks after a tighter sweep
- Feature stability is stronger than expected for a high-p setting:
  - for `top_k=90`, `40` proteins were selected in all `25` seed/fold training splits
  - for `top_k=120`, `70` proteins were selected in all `25` seed/fold training splits
  - that suggests the core signal is concentrated in a fairly stable protein subset

## What Did Not Hold Up

- Correlation-penalized `mRMR` style selection improved one seed-42 run, but lost to the simpler selector on repeated 5-seed CV.
- A 3-model nested logistic stack overfit and was worse than the fixed blend.
- Logit-space blending beat the probability average on one seed-42 run, but did not improve repeated-CV log loss.
- Averaging a few XGBoost seeds helped the seed-42 run, but did not clearly beat the current baseline on repeated CV.
- A 4-model blend that kept both LDA and PLS was worse than the cleaner `XGB + Ridge + PLS` ensemble.
- Complementary-feature LDA panels did not beat the simpler shared top-k LDA setup.

## Best Results So Far

- Best single-fold-compatible model:
  - `Selected XGBoost`
  - top `90` proteins + 3 clinical features
  - latest official `evaluate.py` seed-42 5-fold log loss: `0.560`
- Best repeated-CV model overall:
  - `Selected Ensemble (XGB + Ridge + PLS)` with:
    - XGBoost on top `90` proteins
    - ridge LR on top `80` proteins
    - PLS logistic regression on top `260` proteins compressed to `14` latent components
    - blend weights `0.45 / 0.40 / 0.15`
  - repeated 5-seed CV mean log loss: about `0.5547`
  - repeated 5-seed CV mean AUC: about `0.7865`
  - latest official `evaluate.py` seed-42 5-fold log loss: `0.547`
  - latest official `evaluate.py` seed-42 5-fold AUC: `0.795`

## Important Constraint

- If you must stay fully compatible with starter-code inference that only reads `weights/xgboost_model.json` and `weights/feature_cols.json`, the selected-feature XGBoost is the safest final artifact.
- If you use the updated `Nav/predict.py`, the ensemble bundle is better.
