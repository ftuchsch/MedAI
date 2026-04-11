#!/usr/bin/env python3
"""
predict.py — Run the trained model bundle on new data
=====================================================
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _SCRIPT_DIR / "weights"
_MODEL_PATH = _WEIGHTS_DIR / "xgboost_model.json"
_FEATURES_PATH = _WEIGHTS_DIR / "feature_cols.json"
_ENSEMBLE_CONFIG_PATH = _WEIGHTS_DIR / "ensemble_config.json"


def parse_args():
    p = argparse.ArgumentParser(description="Run the trained ATI model bundle on new data.")
    p.add_argument("--data", required=True, help="Path to input CSV")
    p.add_argument("--out", default="predictions.csv", help="Output CSV path")
    return p.parse_args()


def load_feature_cols(path: Path) -> list[str]:
    with path.open() as f:
        return json.load(f)


def load_components(weights_dir: Path) -> list[dict]:
    if _ENSEMBLE_CONFIG_PATH.exists():
        logging.info(f"Loading ensemble config: {_ENSEMBLE_CONFIG_PATH}")
        with _ENSEMBLE_CONFIG_PATH.open() as f:
            config = json.load(f)
        components = []
        for component in config["components"]:
            feature_cols = load_feature_cols(weights_dir / component["feature_file"])
            model_path = weights_dir / component["model_file"]
            if component["artifact_type"] == "xgboost":
                model = XGBClassifier()
                model.load_model(str(model_path))
            elif component["artifact_type"] == "sklearn":
                model = joblib.load(model_path)
            else:
                raise ValueError(f"Unsupported artifact type: {component['artifact_type']}")
            components.append(
                {
                    "name": component["name"],
                    "weight": component["weight"],
                    "artifact_type": component["artifact_type"],
                    "feature_cols": feature_cols,
                    "model": model,
                }
            )
        logging.info(f"Loaded {len(components)} ensemble components")
        return components

    logging.info("Ensemble config not found. Falling back to single XGBoost model.")
    model = XGBClassifier()
    model.load_model(str(_MODEL_PATH))
    return [
        {
            "name": "Selected XGBoost",
            "weight": 1.0,
            "artifact_type": "xgboost",
            "feature_cols": load_feature_cols(_FEATURES_PATH),
            "model": model,
        }
    ]


def prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logging.warning(
            f"{len(missing)} expected feature columns missing from input — filled with NaN. "
            f"First few: {missing[:5]}"
        )
    return df.reindex(columns=feature_cols).astype(float)


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

    logging.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False, na_values=[".", ""])
    logging.info(f"  {len(df)} samples loaded")

    components = load_components(_WEIGHTS_DIR)
    ids = df["sample_id"] if "sample_id" in df.columns else pd.RangeIndex(len(df))

    final_prob = np.zeros(len(df), dtype=float)
    component_outputs = {}
    for component in components:
        X_component = prepare_features(df, component["feature_cols"])
        y_prob = component["model"].predict_proba(X_component)[:, 1]
        component_outputs[component["name"]] = y_prob
        final_prob += component["weight"] * y_prob
        logging.info(
            f"[{component['name']}] weight={component['weight']:.2f} "
            f"features={len(component['feature_cols'])}"
        )

    results = pd.DataFrame(
        {
            "sample_id": ids,
            "prob_ati": final_prob,
            "pred_label": (final_prob >= 0.5).astype(int),
        }
    )

    if len(components) > 1:
        for component_name, y_prob in component_outputs.items():
            slug = component_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
            results[f"prob_{slug}"] = y_prob

    if "ati" in df.columns:
        results["true_label"] = df["ati"].astype(int).to_numpy()
        logging.info("Found 'ati' column — will compute evaluation metrics")
        evaluate(results)

    results.to_csv(args.out, index=False)
    logging.info(f"Predictions saved to: {args.out}")


if __name__ == "__main__":
    main()
