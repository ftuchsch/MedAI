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

## Current Best

- Best single-model-compatible artifact:
  - selected-feature XGBoost with top `90` proteins + clinical covariates
- Best overall repeated-CV result:
  - `45/40/15` blend of:
    - retuned selected-feature XGBoost with top `90` proteins
    - selected-feature ridge logistic regression with top `80` proteins
    - selected-feature PLS logistic regression with top `260` proteins compressed to `14` latent components

## Completed This Round

- Read the top-level research PDFs again and checked the latest official Prior Labs TabPFN docs
- Retuned the XGBoost branch and promoted a better depth-`1` configuration
- Tested PLS latent-factor branches and found a strong low-weight ensemble gain
- Confirmed the new XGBoost branch beats the earlier XGBoost branch inside the ensemble
- Tested `PLS top_k`, number of latent components, and blend weights
- Tested a cautious 4-model blend with both LDA and PLS and rejected it
- Ran a feature-stability analysis and found a substantial perfectly stable core across repeated CV splits
- Ran a final edge sweep around the new `XGB + Ridge + PLS` winner
- Ran the updated `evaluate.py` end-to-end and saved fresh CV outputs
- Trained fresh final weights with the updated `train.py`
- Verified `predict.py` on the training CSV and confirmed the 3-model ensemble path works

## Next Things To Try

- Measure stability of the wider `top_k=260` PLS feature panel:
  - selection frequency
  - overlap with the XGBoost top-`90` panel
- Try a small number of protein-by-clinical interaction features:
  - especially with `baseline_egfr_23`
  - only if added inside the fold-safe pipeline
- Compare a few bootstrap-averaged versions of the current winner:
  - only if the repeated-CV estimate improves, not just the seed-42 run
- Compare the updated 3-model ensemble against a pure-XGBoost deployment assumption one more time before final handoff
- If time remains:
  - try a stability-selected elastic-net screen for the PLS branch
  - attempt TabPFN only if GPU/auth becomes available, since the official OSS docs recommend GPU usage and interactive model access
