# BKBC ATI Hackathon

Leakage-safe ML pipeline for predicting acute tubular injury from SomaScan proteomics.

## Goals
- Build robust cross-validation pipeline
- Support clinical-only and proteomics-based feature sets
- Generate OOF predictions for ensembling
- Produce final held-out test predictions for submission

## Structure
- `src/`: reusable pipeline code
- `data/`: raw and processed input data
- `outputs/`: predictions, reports, selected features
- `run_pipeline.py`: main entry point