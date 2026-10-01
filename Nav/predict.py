#!/usr/bin/env python3
"""
predict.py — Run the trained Nav bundle on new data
===================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from preprocess import build_feature_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _SCRIPT_DIR / "weights"


def parse_args():
    p = argparse.ArgumentParser(description="Run the trained ATI model bundle on new data.")
    p.add_argument("--data", required=True, help="Path to input CSV")
    p.add_argument("--out", default="predictions.csv", help="Output CSV path")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=_WEIGHTS_DIR,
        help="Directory containing saved model artifacts (default: ./weights/)",
    )
    return p.parse_args()


def load_feature_cols(path: Path) -> list[str]:
    with path.open() as f:
        return json.load(f)


def load_artifact(path: Path, artifact_type: str):
    if artifact_type == "xgboost":
        model = XGBClassifier()
        model.load_model(str(path))
        return model
    if artifact_type == "sklearn":
        return joblib.load(path)
    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def load_components(weights_dir: Path) -> list[dict]:
    ensemble_config_path = weights_dir / "ensemble_config.json"
    if ensemble_config_path.exists():
        logging.info("Loading ensemble config: %s", ensemble_config_path)
        with ensemble_config_path.open() as f:
            config = json.load(f)

        components = []
        for component in config["components"]:
            feature_cols = load_feature_cols(weights_dir / component["feature_file"])
            model_files = component.get("model_files") or [component["model_file"]]
            models = [
                load_artifact(weights_dir / model_file, component["artifact_type"])
                for model_file in model_files
            ]
            components.append(
                {
                    "name": component["name"],
                    "weight": float(component["weight"]),
                    "artifact_type": component["artifact_type"],
                    "feature_cols": feature_cols,
                    "models": models,
                }
            )
        logging.info("Loaded %s ensemble components", len(components))
        return components

    logging.info("Ensemble config not found. Falling back to single XGBoost model.")
    model = load_artifact(weights_dir / "xgboost_model.json", "xgboost")
    return [
        {
            "name": "Selected XGBoost",
            "weight": 1.0,
            "artifact_type": "xgboost",
            "feature_cols": load_feature_cols(weights_dir / "feature_cols.json"),
            "models": [model],
        }
    ]


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Reuse the feature alignment used during training and evaluation."""
    return build_feature_frame(df, feature_cols)


def evaluate(results: pd.DataFrame):
    from sklearn.metrics import classification_report, log_loss, roc_auc_score

    y_true = results["true_label"].to_numpy()
    y_pred = results["pred_label"].to_numpy()
    y_prob = results["prob_ati"].to_numpy()

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

    logging.info("Loading data from %s...", args.data)
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info("  %s samples loaded", len(df))

    components = load_components(args.model_dir)
    ids = df["sample_id"] if "sample_id" in df.columns else pd.RangeIndex(len(df))

    final_prob = np.zeros(len(df), dtype=float)
    component_outputs = {}
    for component in components:
        X_component = prepare_features(df, component["feature_cols"])
        seed_probs = [model.predict_proba(X_component)[:, 1] for model in component["models"]]
        y_prob = np.mean(np.vstack(seed_probs), axis=0)
        component_outputs[component["name"]] = y_prob
        final_prob += component["weight"] * y_prob
        logging.info(
            "[%s] weight=%.4f features=%s models=%s",
            component["name"],
            component["weight"],
            len(component["feature_cols"]),
            len(component["models"]),
        )

    final_prob = np.clip(final_prob, 1e-6, 1 - 1e-6)
    results = pd.DataFrame(
        {
            "sample_id": ids,
            "prob_ati": final_prob,
            "pred_label": (final_prob >= 0.5).astype(int),
        }
    )

    if len(components) > 1:
        for component_name, y_prob in component_outputs.items():
            slug = (
                component_name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("+", "plus")
            )
            results[f"prob_{slug}"] = y_prob

    if "ati" in df.columns:
        results["true_label"] = df["ati"].astype(int).to_numpy()
        logging.info("Found 'ati' column — will compute evaluation metrics")
        evaluate(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    logging.info("Predictions saved to: %s", args.out)


if __name__ == "__main__":
    main()
