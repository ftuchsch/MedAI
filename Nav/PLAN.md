# Plan

## Already Tried

- Read repo instructions and research summary
- Audited the local CSV schema and class balance
- Reproduced the starter baseline
- Compared:
  - starter XGBoost
  - starter sparse logistic
  - clinical-only models
  - LightGBM with top-k filtering
  - XGBoost with top-k filtering
  - ridge / elastic-net logistic with top-k filtering
  - shrinkage LDA with top-k filtering
  - PLS latent-factor logistic regression
- Tested different top-k feature counts
- Tested multiple CV seeds
- Tested fixed blends of the best models
- Tested correlation-penalized `mRMR` screening
- Tested seed-averaged XGBoost blends
- Tested nested stacking and logit-space blending
- Tested a 4-model `XGB + Ridge + LDA + PLS` blend
- Tested stability-aware repeated-subsample ANOVA selection for the XGBoost branch
- Tested a clinical-cluster GroupKFold stress proxy for cross-cohort robustness
- Tested residualized proteins, rank-normalized proteins, and small interaction panels
- Tested bagged XGBoost and LightGBM diversity branches
- Installed and screened CatBoost as an extra tree-family diversity branch
- Tested low-weight proteomics-only XGBoost branches inside the ensemble

## Current Best

- Best single-model-compatible artifact:
  - stability-selected XGBoost with top `90` proteins + clinical covariates
- Best overall repeated-CV result:
  - `41/12/32/15` blend of:
    - retuned stability-selected XGBoost with top `90` proteins + clinical covariates
    - proteomics-only stability-selected XGBoost with top `90` proteins
    - selected-feature ridge logistic regression with top `80` proteins
    - selected-feature PLS logistic regression with top `260` proteins compressed to `14` latent components

## Completed This Round

- Ran several robustness screens around the current winner:
  - residualized proteins
  - rank-normalized proteins
  - clinical-only cluster GroupKFold proxy
  - bagged tree diversity branches
  - protein-by-clinical interaction panels
- Promoted the stable-subsample XGBoost selector after it beat the raw selector on repeated 5-seed CV
- Updated `preprocess.py`, `evaluate.py`, and `train.py` so the shipped weights now use the same selector that won in CV
- Screened CatBoost and rejected it after it lost to the current winner both alone and in blends
- Added a low-weight proteomics-only XGBoost branch after it improved repeated 5-seed CV and the official seed-42 check
- Ran a fresh official `evaluate.py` check and reached about `0.544` log loss, `0.799` AUC
- Retrained the final `weights/` bundle with the new `0.41 / 0.12 / 0.32 / 0.15` ensemble
- Snapshotted both major leaderboard candidates under `weights_variants/`

## Next Things To Try

- Read the top-level research PDFs again and extract any kidney-specific modeling hints worth encoding directly into features or CV
- Compare the two saved contenders on the public leaderboard:
  - `weights_variants/current_raw_xgb_45_40_15/`
  - `weights_variants/stable_xgb_48_37_15/`
- Compare the new 4-model contender on the public leaderboard:
  - `weights_variants/stable_xgb_protxgb_41_32_15_12/`
- Measure stability of the wider `top_k=260` PLS feature panel:
  - selection frequency
  - overlap with the XGBoost top-`90` panel
- Try a lighter proteomics-only branch weight if the leaderboard prefers robustness over the tiny local CV gain
- Try a slightly more targeted interaction panel:
  - prioritize `baseline_egfr_23`
  - only use proteins that are already highly stable
- Compare a few bootstrap-averaged versions of the current winner:
  - only if the repeated-CV estimate improves, not just the seed-42 run
- If time remains:
  - try a stability-selected elastic-net screen for the PLS branch
  - attempt TabPFN only if GPU/auth becomes available, since the official OSS docs recommend GPU usage and interactive model access
