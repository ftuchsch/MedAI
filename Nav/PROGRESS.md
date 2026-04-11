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

## Scoreboard

| Model | CV setup | Mean log loss | Mean AUC |
|---|---:|---:|---:|
| Starter XGBoost | 5-fold, seed 42 | ~0.626 | ~0.752 |
| Selected XGBoost | latest official 5-fold, seed 42 | ~0.560 | ~0.781 |
| Selected Ridge LR | latest official 5-fold, seed 42 | ~0.570 | ~0.766 |
| Selected Shrinkage LDA | latest official 5-fold, seed 42 | ~0.711 | ~0.760 |
| Selected PLS LR | latest official 5-fold, seed 42 | ~1.260 | ~0.733 |
| Selected Ensemble (XGB + Ridge + PLS) | latest official 5-fold, seed 42 | ~0.547 | ~0.796 |
| Selected Ensemble (XGB + Ridge + PLS) | repeated 5 seeds x 5 folds | ~0.5547 | ~0.7865 |

## What Changed In Code

- `evaluate.py`
  - now does leak-safe fold-by-fold feature selection
  - compares starter baseline, selected XGBoost, selected ridge LR, selected shrinkage LDA, selected PLS LR, and the fixed weighted ensemble
  - saves OOF predictions for later blending/debugging
- `train.py`
  - now trains the strong single XGBoost fallback plus the stronger `XGB + Ridge + PLS` ensemble bundle
  - still saves LDA as an analysis artifact even though the final ensemble no longer uses it
- `predict.py`
  - now supports ensemble inference if the ensemble files exist
  - still falls back to single XGBoost if only starter-style weights are present
  - now uses the updated `0.45 / 0.40 / 0.15` ensemble weights from the saved bundle

## Where We Are For Winning

- We are well past the starter baseline.
- The strongest path right now is:
  - selected top-k protein filtering inside CV
  - a retuned very shallow regularized XGBoost branch
  - a smooth ridge logistic branch
  - a low-weight supervised latent-factor PLS branch
  - a `45% / 40% / 15%` weighted average of:
    - XGBoost on top `90` proteins
    - ridge logistic regression on top `80` proteins
    - PLS logistic regression on top `260` proteins reduced to `14` components
- The selection step also looks reasonably stable:
  - `40` proteins were selected in all `25` repeated-CV training splits for the `top_k=90` branch
  - the winning PLS branch wants a much broader `top_k=260` panel, but compresses it down to `14` latent factors before blending
- The next gains are likely to come from:
  - stability-aware selection for the wide PLS branch
  - carefully chosen protein-by-clinical interaction features
  - one last tight blend-weight sweep only if it is evaluated with repeated CV
  - GPU-enabled TabPFN only if compute/auth becomes available, since the official Prior Labs docs recommend GPU-backed usage
