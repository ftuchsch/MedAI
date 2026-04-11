# Explain

## What I’m Doing

I’m treating this as a small-sample, high-dimensional tabular problem. That means raw model size matters less than leakage control, variance control, and probability calibration.

The main changes are:

- selecting only a small subset of proteins inside each CV fold
- keeping the clinical covariates in every model
- using a very shallow regularized tree model instead of deeper trees on all 6,592 proteins
- pairing that tree model with stable linear models and a supervised latent-factor model
- averaging their probabilities rather than doing a high-variance learned stack

## Why Top-k Protein Selection

With `426` rows and `6,592` proteins, training directly on every protein is noisy. The search showed that:

- using all proteins is worse than using a small selected subset
- the best region is around `85–100` proteins, not thousands

The selection is done with an ANOVA F-score on the training fold only. That keeps evaluation leak-safe.

## Why XGBoost

XGBoost handled the selected protein subset best when constrained to an even simpler shape than the earlier version:

- depth-`1` trees
- low learning rate
- moderate row subsampling
- wider column subsampling
- moderate regularization

That reduced variance and improved repeated-CV log loss relative to both the starter code and the earlier depth-`2` branch.

## Why Add Ridge Logistic Regression

The ridge logistic model is weaker alone than the best XGBoost on some splits, but it is complementary:

- it gives smoother probabilities
- it is less flexible and can generalize differently
- averaging it with XGBoost improved repeated-CV log loss

This is exactly the kind of low-complexity blend that is usually worth keeping in a hackathon.

## Why Add PLS LR

PLS is a much better fit for this kind of proteomics problem than a raw wide linear model because it can:

- take a broader protein panel
- compress it into a small number of supervised latent components
- hand those components to a simple logistic regression head

The winning branch uses:

- top `260` proteins + clinical covariates
- `14` latent PLS components
- logistic regression with `C=0.5`

PLS is actually poor alone on log loss here, but at low weight it adds useful diversity that the tree and ridge branches do not capture cleanly.

## Current Final Blend

The current best local blend is:

- `45%` XGBoost using top `90` proteins + clinical covariates
- `40%` ridge logistic regression using top `80` proteins + clinical covariates
- `15%` PLS logistic regression using top `260` proteins + clinical covariates, compressed to `14` components

That beat the earlier LDA-based 3-model blend and survived repeated 5-seed checks.

## Why The PLS Branch Can Help Even Though It Looks Bad Alone

This is the non-obvious part of the current winner. The standalone PLS branch has poor log loss, but the ensemble still improves. That can happen because:

- the tree and ridge branches already cover the well-calibrated core signal
- the PLS branch is allowed to be noisy as long as its errors are different
- a small `15%` weight lets it contribute ranking information without dominating the final probability

In practice, that low-weight PLS branch improved repeated-CV log loss enough to keep.

## What Did Not Help

- nested probability calibration
  - sigmoid was slightly worse
  - isotonic was much worse
- two-stage tree re-ranking after the initial filter
  - the simpler univariate selector generalized better here
- correlation-penalized `mRMR` screening
  - it looked good on one CV seed and then lost on repeated CV
- nested logistic stacking over OOF predictions
  - it overfit and was clearly worse than the fixed blend
- logit-space blending
  - it improved one seed-42 run, but not the repeated-CV average
- keeping both LDA and PLS in the same final ensemble
  - the cleaner `XGB + Ridge + PLS` blend was better

## Why Two Final Outputs

There is a compatibility concern:

- the starter code expects `weights/xgboost_model.json` plus `feature_cols.json`
- the better solution is an ensemble, which needs extra artifacts

So the repo now saves both:

- a strong single XGBoost fallback that stays close to the starter interface
- a better 3-model ensemble bundle that the updated `predict.py` can use automatically

That gives you a safer deployment option without giving up the stronger local result.
