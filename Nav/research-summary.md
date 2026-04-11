# Research Summary

Distilled notes from the PDFs in `research/` for MedAI Hackathon Challenge 3.

## Documents Read

- `research/medaihackathon_spr2026.pdf`
- `research/Initital-research-challenge-3.pdf`
- `research/SOTA-models-challenge-3.pdf`
- `research/Tecchnical-implmentation-challenge-3.pdf`

## Important Note

- `Initital-research-challenge-3.pdf` and `SOTA-models-challenge-3.pdf` have identical extracted text. Treat them as duplicate playbooks with different filenames.

## Modeling Takeaways

- Default strong baseline should be gradient-boosted trees, especially LightGBM and likely XGBoost as a secondary model.
- This challenge is medium-`N`, high-`p`: about 1,500 train samples with 6,592 proteomics features plus clinical covariates.
- The research notes emphasize that signal is likely concentrated in a smaller subset of ATI-associated proteins rather than spread uniformly across all proteins.
- Clinical features may be strong, especially kidney-function proxies like eGFR, but over-reliance is a generalization risk.
- Always run ablations:
  - clinical-only
  - proteomics-only
  - combined
- If combined only barely beats clinical-only, expect brittleness under distribution shift.

## Feature Selection Guidance

- Feature selection is central, not optional.
- All selection must happen inside CV folds.
- Suggested feature subset ladder:
  - FS0: clinical only
  - FS1: clinical + literature-prior ATI proteins
  - FS2: FS1 + top-k fold-wise selected proteins
  - FS3: broader data-driven selected set
- Recommended practical sequence:
  - fold-safe filter/ranking
  - SHAP ranking from LightGBM
  - optional Boruta-style shadow-feature selection on a reduced pool
  - optional elastic-net stability selection
- Do not run Boruta across all ~6,500 proteins at once.
- Penalized linear models are useful, but raw L1 selection can be unstable; repeat across seeds/subsamples if used for screening.

## Validation Rules

- Leakage control is the top technical requirement.
- Imputation, scaling, selection, and any preprocessing must be fit on training folds only.
- Use `StratifiedGroupKFold` if patient IDs exist.
- Otherwise use stratified CV and explicitly check for exact and near duplicates.
- Track mean fold AUC and fold-to-fold standard deviation.
- If fold std is high, prefer repeated CV, seed averaging, and simpler feature sets before chasing more model complexity.
- Generate OOF predictions for every serious model so blending and stacking stay leak-safe.

## Model Strategy

- Anchor the solution on LightGBM.
- Add XGBoost and elastic-net logistic regression as diversity models if they are fast to run.
- Use blending before stacking complexity.
- Ridge/logistic stacking on OOF predictions is reasonable only after base models are stable.
- TabPFN is only relevant after reducing to `<= 500` features.
- Deep tabular models are lower priority than stable tree baselines for a one-day hackathon.

## Execution Order For Tomorrow

- First confirm schema, target encoding, metric, IDs, and submission format.
- Build a leak-safe CV harness before serious tuning.
- Get reproducible FS0 and FS1 baselines first.
- Add a second tree model and a sparse linear model next.
- Then refine top-k feature subsets.
- Use seed averaging and simple blends late in the cycle.
- Treat stacking and optional TabPFN as endgame work, not day-start work.

## SCC / Infra Notes

- The BU SCC slide deck is mostly environment guidance, not modeling guidance.
- Use `eduroam`, not `BU Guest`.
- OnDemand URL: `https://scc-ondemand.bu.edu`
- Team workspace lives under `/projectnb/medaihack/teamN`
- Challenge startup resources live under `/projectnb/medaihack/startup`
- All jobs must use project `medaihack`
- Each team gets one Nvidia `L40S` GPU job at a time, shared across interactive and batch use.
- CPU-only jobs can run concurrently.
- Suggested environment flow from the slides:
  - start a CPU desktop session
  - run `/projectnb/medaihack/startup/setup.sh`
  - create challenge-specific `virtualenv` environments
  - then launch Jupyter or VS Code Server using those venvs
- Slides explicitly say to avoid copying challenge data outside the authorized SCC locations.
- Slides also advise using `virtualenv`, not `conda` or `uv`, for the hackathon setup.
