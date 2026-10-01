#!/usr/bin/env python3
"""
predict.py — Run the trained TabPFN kidney model on new data
============================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model import FEATURES_FILENAME, METADATA_FILENAME, MODEL_FILENAME
from preprocess import build_feature_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _SCRIPT_DIR / "weights"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the trained TabPFN ATI model on new proteomics data."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to input CSV (same feature columns as training data)",
    )
    p.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    p.add_argument(
        "--model-dir",
        default=str(_DEFAULT_MODEL_DIR),
        help="Directory containing the trained TabPFN bundle (default: ./weights/)",
    )
    return p.parse_args()


def load_bundle(model_dir: str):
    """Restore the trained TabPFN bundle from disk."""
    bundle_dir = Path(model_dir)
    model_path = bundle_dir / MODEL_FILENAME
    features_path = bundle_dir / FEATURES_FILENAME
    metadata_path = bundle_dir / METADATA_FILENAME

    logging.info(f"Loading model bundle : {bundle_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature list: {features_path}")

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open() as f:
            metadata = json.load(f)

    model = joblib.load(model_path)
    with features_path.open() as f:
        feature_cols = json.load(f)

    logging.info(f"Model expects {len(feature_cols)} features")
    if metadata:
        logging.info(
            f"Loaded {metadata.get('model_name', 'model')} "
            f"(TabPFN v{metadata.get('tabpfn_version', 'unknown')})"
        )

    return model, feature_cols, metadata


def prepare_features(df: pd.DataFrame, feature_cols: list[str]):
    """Align a new DataFrame to the feature columns expected by the model."""
    X = build_feature_frame(df, feature_cols).to_numpy()
    ids = df["sample_id"] if "sample_id" in df.columns else pd.RangeIndex(len(df))
    return X, ids


def run_predict(model, X: np.ndarray, ids, y_true=None) -> pd.DataFrame:
    """Generate ATI predictions for a feature matrix."""
    y_prob = model.predict_proba(X)[:, 1]
    probability_shrinkage = float(getattr(model, "_kidney_probability_shrinkage", 0.0))
    train_prevalence = getattr(model, "_kidney_train_prevalence", None)
    if probability_shrinkage > 0 and train_prevalence is not None:
        y_prob = (1.0 - probability_shrinkage) * y_prob + probability_shrinkage * float(
            train_prevalence
        )
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame(
        {
            "sample_id": ids,
            "prob_ati": y_prob,
            "pred_label": y_pred,
        }
    )
    if y_true is not None:
        out["true_label"] = np.asarray(y_true)
    return out


def evaluate(results: pd.DataFrame):
    """Print classification metrics when ground-truth labels are available."""
    from sklearn.metrics import classification_report, log_loss, roc_auc_score

    y_true = results["true_label"].values
    y_pred = results["pred_label"].values
    y_prob = results["prob_ati"].values

    print(f"\n{'=' * 50}")
    print(
        f"  Samples : {len(results)}  |  "
        f"No ATI: {(y_true == 0).sum()}  |  ATI: {(y_true == 1).sum()}"
    )
    print(f"{'=' * 50}")

    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
        print(f"AUC (ROC) : {roc_auc_score(y_true, y_prob):.3f}")
        print(f"Log loss  : {log_loss(y_true, y_prob):.3f}")
    else:
        print("  (Only one class present in labels — AUC not defined)")


def main():
    args = parse_args()

    model, feature_cols, _ = load_bundle(args.model_dir)

    logging.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples loaded")

    X, ids = prepare_features(df, feature_cols)

    y_true = None
    if "ati" in df.columns:
        y_true = df["ati"].astype(int).values
        logging.info("Found 'ati' column — will compute evaluation metrics")

    results = run_predict(model, X, ids, y_true=y_true)

    if y_true is not None:
        evaluate(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logging.info(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
