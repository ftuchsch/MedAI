# Boston Kidney Biopsy Cohort — TabPFN Pipeline

## Task

See the [project README](../../README.md) for the full modeling comparison,
recorded results, and validation limitations.

Predict **acute tubular injury (ATI)** from plasma proteomics as a binary
classification problem:

| Label | Meaning |
|-------|---------|
| **0** | No ATI |
| **1** | ATI present |

The final leaderboard metric is **log loss** on an external held-out cohort, so
the Felix pipeline now focuses on a single deployment path:

- TabPFN v2 classifier
- top-k protein filtering before modeling
- the 3 clinical covariates kept alongside the selected proteins

---

## Data

In this repo the default training CSV is:

```bash
../../data/train.csv
```

The table contains **426 patients** with:

| Column group | Description |
|---|---|
| `sample_id` | Anonymised patient identifier |
| `ati` | Binary ATI label |
| `age` | Age (10-year bin midpoint) |
| `sex` | Sex (1 = Male, 2 = Female) |
| `baseline_egfr_23` | Baseline eGFR |
| `feature_XXXX` × 6,592 | ComBat-corrected SomaScan protein abundances |

The current default is **top 100 proteins + 3 clinical features** before
TabPFN, which keeps the model comfortably inside the open-source v2 feature
limits.

---

## Environment Setup

```bash
# From the repository root, using Python 3.12:
python3.12 -m venv .venv-tabpfn
source .venv-tabpfn/bin/activate
pip install -r Felix/kidney/requirements.txt
# Run the commands below from this workspace:
cd Felix/kidney
```

`requirements.txt` now pins `tabpfn==2.0.0`.

The prediction wrapper uses the active environment. `MEDAI_PYTHON` can override
the interpreter; no hardcoded cluster or virtual environment path is needed.

The first `train.py` or `evaluate.py` run may download the TabPFN checkpoint
into the local cache if it is not already present.

---

## Pipeline

```text
data/train.csv
    |
    |-- evaluate.py   -> fold-safe CV, OOF predictions, feature sets
    |-- train.py      -> fitted TabPFN bundle in weights/
    `-- predict.py    -> inference on new samples
```

---

## Step 1 — Cross-Validation

Run fold-safe CV with per-fold top-k feature selection:

```bash
python evaluate.py --data ../../data/train.csv --protein-top-k 100
```

Outputs written to `./results/`:

- `cv_results.csv`
- `oof_predictions_tabpfn.csv`
- `cv_feature_sets_tabpfn.json`
- `cv_confusion_matrix_tabpfn.png`

---

## Step 2 — Train Final Model

Train on all available BKBC samples and save a deployable TabPFN bundle:

```bash
python train.py --data ../../data/train.csv --protein-top-k 100
```

Outputs written to `./weights/`:

- `tabpfn_model.joblib`
- `feature_cols.json`
- `model_metadata.json`
- `feature_selection.csv`

---

## Step 3 — Predict On New Data

```bash
bash predict.sh /path/to/new_data.csv

# optional explicit Python call
python predict.py --data /path/to/new_data.csv --out predictions.csv
```

If the input CSV contains `ati`, `predict.py` also prints AUC and log loss.

Output columns:

| Column | Description |
|--------|-------------|
| `sample_id` | Patient identifier |
| `prob_ati` | Predicted probability of ATI |
| `pred_label` | Thresholded prediction at 0.5 |
| `true_label` | Ground truth when available |

---

## File Structure

```text
Felix/kidney/
├── evaluate.py
├── model.py
├── predict.py
├── predict.sh
├── preprocess.py
├── train.py
├── README.md
├── requirements.txt
└── weights/
    ├── tabpfn_model.joblib
    ├── feature_cols.json
    └── model_metadata.json
```

---

## Help

```bash
python train.py --help
python evaluate.py --help
python predict.py --help
```
