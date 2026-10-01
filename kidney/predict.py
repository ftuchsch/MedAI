#!/usr/bin/env python3
"""
predict.py — Run the trained ATI ensemble on new data
=====================================================
Loads the saved ensemble bundle, aligns incoming columns to each base model's
saved feature order, and outputs final ATI probabilities.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss, roc_auc_score

from ensemble_utils import combine_from_base_matrix, predict_model_group
from model import ID_COLUMN, TARGET_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_BUNDLE_PATH = _SCRIPT_DIR / "weights" / "ensemble_bundle.joblib"


def parse_args():
    p = argparse.ArgumentParser(description="Run the trained ATI ensemble on a new CSV.")
    p.add_argument(
        "--data",
        required=True,
        help="Path to input CSV",
    )
    p.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV path (default: predictions.csv)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=_BUNDLE_PATH.parent,
        help="Directory containing ensemble_bundle.joblib (default: ./weights/)",
    )
    p.add_argument(
        "--include-base-probs",
        action="store_true",
        help="Include one probability column per base model in the output.",
    )
    return p.parse_args()


def load_bundle(bundle_path: Path) -> dict:
    """Restore the saved ensemble bundle."""
    logging.info(f"Loading ensemble bundle: {bundle_path}")
    return joblib.load(bundle_path)


def summarize_missing_columns(df: pd.DataFrame, feature_cols: list[str]):
    """Log missing incoming columns relative to one base model's feature order."""
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        logging.warning(
            f"{len(missing)} expected columns absent from input; "
            f"filled with NaN. First few: {missing[:5]}"
        )


def evaluate_predictions(results: pd.DataFrame):
    """Print metrics when ATI labels are available in the input."""
    y_true = results["true_label"].to_numpy()
    y_prob = results["prob_ati"].to_numpy()
    y_pred = results["pred_label"].to_numpy()

    print("\n" + "=" * 60)
    print("ATI ENSEMBLE EVALUATION ON PROVIDED DATA")
    print("=" * 60)
    print(f"Samples: {len(results)} | No ATI: {(y_true == 0).sum()} | ATI: {(y_true == 1).sum()}")
    if len(np.unique(y_true)) > 1:
        print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
        print(f"AUC (ROC): {roc_auc_score(y_true, y_prob):.3f}")
        print(f"Log loss : {log_loss(y_true, y_prob):.3f}")
    else:
        print("Only one class is present; AUC is undefined.")


def main():
    args = parse_args()
    bundle = load_bundle(args.model_dir / _BUNDLE_PATH.name)
    metadata = bundle.get("metadata", {})
    threshold = float(metadata.get("prediction_threshold", 0.5))

    logging.info(f"Loading input data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples loaded")

    base_model_names = bundle["base_model_names"]
    base_prob_columns = {}
    base_matrix_parts = []

    for model_name in base_model_names:
        group = bundle["model_groups"][model_name]
        summarize_missing_columns(df, group["feature_cols"])
        prob = predict_model_group(group, df)
        base_prob_columns[model_name] = prob
        base_matrix_parts.append(prob)

    base_matrix = np.column_stack(base_matrix_parts)
    final_prob = combine_from_base_matrix(base_matrix, bundle["combiner"])
    final_pred = (final_prob >= threshold).astype(int)

    ids = df[ID_COLUMN] if ID_COLUMN in df.columns else pd.RangeIndex(len(df))
    results = pd.DataFrame(
        {
            ID_COLUMN: ids,
            "prob_ati": final_prob,
            "pred_label": final_pred,
        }
    )

    if args.include_base_probs:
        for model_name, prob in base_prob_columns.items():
            results[f"prob_{model_name}"] = prob

    if TARGET_COLUMN in df.columns:
        results["true_label"] = df[TARGET_COLUMN].astype(int).to_numpy()
        evaluate_predictions(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logging.info(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
