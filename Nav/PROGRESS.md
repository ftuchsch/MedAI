# Progress

## Current Status

- Repo instructions reviewed:
  - `AGENTS.md`
  - `README.md`
  - `research-summary.md`
- Local Mac venv created under `Nav/.venv`
- Starter baseline reproduced on the local dataset
- Model search completed across:
  - clinical-only models
  - LightGBM
  - XGBoost
  - ridge / elastic-net logistic regression
  - shrinkage LDA
  - PLS latent-factor logistic regression
  - fold-safe top-k protein subsets
  - repeated CV seeds
  - simple fixed blends
  - low-weight three-model and four-model blends
  - stability-aware subsample feature selection for the XGBoost branch

## Scoreboard

| Model | CV setup | Mean log loss | Mean AUC |
|---|---:|---:|---:|
| Starter XGBoost | 5-fold, seed 42 | ~0.626 | ~0.752 |
| Selected XGBoost | latest official 5-fold, seed 42 | ~0.557 | ~0.785 |
| Selected Proteomics-Only XGBoost | latest official 5-fold, seed 42 | ~0.559 | ~0.783 |
| Selected Ridge LR | latest official 5-fold, seed 42 | ~0.570 | ~0.766 |
| Selected Shrinkage LDA | latest official 5-fold, seed 42 | ~0.711 | ~0.760 |
| Selected PLS LR | latest official 5-fold, seed 42 | ~1.260 | ~0.733 |
| Selected Ensemble (XGB + ProtXGB + Ridge + PLS) | latest official 5-fold, seed 42 | ~0.544 | ~0.799 |
| Selected Ensemble (XGB + ProtXGB + Ridge + PLS) | repeated 5 seeds x 5 folds | ~0.5535 | ~0.7876 |
| Public leaderboard | hidden public test | 0.413325 | n/a |

## What Changed In Code

- `evaluate.py`
  - now does leak-safe fold-by-fold feature selection
  - compares starter baseline, selected XGBoost, selected ridge LR, selected shrinkage LDA, selected PLS LR, and the fixed weighted ensemble
  - saves OOF predictions for later blending/debugging
- `train.py`
  - now trains the strong single XGBoost fallback plus the stronger `XGB + ProtXGB + Ridge + PLS` ensemble bundle
  - now also saves a proteomics-only XGBoost branch for the final bundle
  - still saves LDA as an analysis artifact even though the final ensemble no longer uses it
  - now trains the XGBoost branch with the same stability-aware selector used in CV
- `predict.py`
  - now supports ensemble inference if the ensemble files exist
  - still falls back to single XGBoost if only starter-style weights are present
  - now uses the updated `0.41 / 0.12 / 0.32 / 0.15` ensemble weights from the saved bundle

## Where We Are For Winning

- We are well past the starter baseline.
- The strongest path right now is:
  - repeated-subsample stable top-k protein filtering for the XGBoost branch
  - a retuned very shallow regularized XGBoost branch
  - a second low-weight proteomics-only XGBoost branch that cannot rely on clinical covariates
  - a smooth ridge logistic branch
  - a low-weight supervised latent-factor PLS branch
  - a `41% / 12% / 32% / 15%` weighted average of:
    - XGBoost on top `90` proteins selected by stable subsample frequency
    - proteomics-only XGBoost on top `90` proteins selected by the same stable screen
    - ridge logistic regression on top `80` proteins
    - PLS logistic regression on top `260` proteins reduced to `14` components
- The selection step also looks reasonably stable:
  - `40` proteins were selected in all `25` repeated-CV training splits for the `top_k=90` branch
  - the winning PLS branch wants a much broader `top_k=260` panel, but compresses it down to `14` latent factors before blending
- I also saved both major contenders side by side:
  - `weights_variants/current_raw_xgb_45_40_15/`
  - `weights_variants/stable_xgb_48_37_15/`
  - `weights_variants/stable_xgb_protxgb_41_32_15_12/`
- The next gains are likely to come from:
  - testing whether the new proteomics-only branch helps on the public leaderboard or only in local CV
  - testing whether the old raw-XGB selector or the new stable-XGB selector wins more reliably on the hidden leaderboard
  - stability-aware selection for the wide PLS branch
  - carefully chosen protein-by-clinical interaction features
  - one last tight blend-weight sweep only if it is evaluated with repeated CV
  - GPU-enabled TabPFN only if compute/auth becomes available, since the official Prior Labs docs recommend GPU-backed usage
