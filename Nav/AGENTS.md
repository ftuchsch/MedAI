# AGENTS.md

This file is the base operating context for any Codex instance started for the MedAI Hackathon work in this repository.

## Mission

Work on MedAI Hackathon Challenge 3: acute tubular injury (ATI) prediction from kidney proteomics data.

Primary modeling task:
- Predict ATI presence or absence from pre-biopsy blood proteomics and clinical covariates.
- Data includes 6,592 SomaScan protein abundance features plus clinical variables.
- Current public split is 1,500 training samples and 500 validation samples.
- Success is driven by robust validation, leakage control, reproducibility, and fast iteration under hackathon time pressure.

## Repository Context

Current repo structure:
- `Nav/`: primary workspace for Nav-specific planning, notes, experiments, scripts, and agent coordination.
- `research/medaihackathon_spr2026.pdf`: official setup/instructions from organizers.
- `research/Initital-research-challenge-3.pdf`
- `research/SOTA-models-challenge-3.pdf`
- `research/Tecchnical-implmentation-challenge-3.pdf`
- `Felix/`: parallel work by a different teammate. Treat it as out of scope unless the user explicitly asks to inspect it.

## Working Rules

When assisting in this repo:
- Prefer building inside `Nav/` and avoid depending on code from `Felix/`.
- Preserve reproducibility. Any experiment code should make seeds, folds, and feature generation explicit.
- Treat data leakage as a top-tier failure mode. No validation-informed preprocessing should leak across folds.
- Optimize for hackathon speed, but not at the expense of invalid evaluation.
- Keep changes modular and easy to rerun.
- Record assumptions in code comments or short markdown notes when they materially affect model validity.

## Priorities

Default order of work:
1. Understand available data schema, labels, metrics, and submission requirements.
2. Confirm training and validation flow, metrics, and submission format.
3. Strengthen baselines before attempting high-complexity modeling.
4. Improve cross-validation fidelity and feature handling.
5. Explore ensembling only after single-model validation is trustworthy.

## Modeling Guidance

Prefer these habits unless the current repo already has a stronger pattern:
- Start with strong tabular baselines: regularized linear models, gradient boosting, tree ensembles.
- Use feature selection or dimensionality reduction carefully and only inside fold-safe pipelines.
- Compare proteomics-only, clinical-only, and combined feature sets.
- Track class balance and calibrate thresholding separately from rank-based leaderboard optimization if needed.
- Favor repeated, auditable experiments over one-off notebook results.

## Expectations For Agents

Each Codex instance should:
- State what part of the pipeline it is inspecting or changing.
- Avoid broad refactors unless they clearly improve iteration speed or correctness.
- Verify code paths with lightweight checks when possible.
- Leave concise notes if work is incomplete or blocked.
- Work under `Nav/` by default.
- Ignore `Felix/` unless the user explicitly says otherwise.

## Useful References

Use these first when challenge details are unclear:
- `Nav/research-summary.md`
- `research/medaihackathon_spr2026.pdf`
- Files created under `Nav/`

## First Questions A New Agent Should Answer

1. What is the exact target label name and metric?
2. What files contain train, validation, and any test or submission templates?
3. How are clinical covariates encoded and merged with proteomics features?
4. What cross-validation strategy should be implemented in `Nav/`?
5. What baseline score is currently reproducible from the `Nav/` workspace?
