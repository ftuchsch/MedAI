#!/usr/bin/env python3
"""
train.py — Train the final ATI ensemble on all available data
=============================================================
Uses the evaluation outputs from `evaluate.py` to:

- train the selected base models on all 426 rows
- fit the saved ensemble combiner from OOF base predictions
- persist a single inference bundle for `predict.py`
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ensemble_utils import fit_final_combiner, fit_full_model_group
from model import TARGET_COLUMN, TREE_MODEL_FAMILIES
from preprocess import load_data, validate_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "train.csv"
_DEFAULT_RESULTS = _SCRIPT_DIR / "results"
_DEFAULT_OUT = _SCRIPT_DIR / "weights"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the final ATI ensemble and save inference artifacts."
    )
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../data/train.csv)",
    )
    p.add_argument(
        "--results-dir",
        default=str(_DEFAULT_RESULTS),
        help="Directory containing evaluate.py outputs (default: ./results/)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Directory to save final ensemble artifacts (default: ./weights/)",
    )
    p.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="Override inner XGBoost threads per fit (default: model config).",
    )
    p.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=None,
        help="Override inner LightGBM threads per fit (default: model config).",
    )
    p.add_argument(
        "--elastic-n-jobs",
        type=int,
        default=None,
        help="Override inner LogisticRegressionCV threads per fit (default: model config).",
    )
    return p.parse_args()


def build_feature_importance_table(model_groups: dict[str, dict]) -> pd.DataFrame:
    """Aggregate tree-model feature importances across seeds for reporting."""
    rows = []
    for model_name, group in model_groups.items():
        if group["family"] not in TREE_MODEL_FAMILIES:
            continue
        importances = np.mean(
            np.vstack([est.feature_importances_ for est in group["estimators"]]),
            axis=0,
        )
        for feature, importance in zip(group["feature_cols"], importances):
            rows.append(
                {
                    "model_name": model_name,
                    "feature": feature,
                    "importance": float(importance),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["model_name", "feature", "importance"])
    return pd.DataFrame(rows).sort_values(
        ["model_name", "importance"],
        ascending=[True, False],
    )


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    final_recipe_path = results_dir / "final_recipe.json"
    base_oof_path = results_dir / "base_oof_predictions_mean.csv"

    if not final_recipe_path.exists():
        raise FileNotFoundError(f"Missing final recipe: {final_recipe_path}")
    if not base_oof_path.exists():
        raise FileNotFoundError(f"Missing base OOF matrix: {base_oof_path}")

    with open(final_recipe_path) as f:
        final_recipe = json.load(f)

    base_oof_df = pd.read_csv(base_oof_path)
    base_model_names = list(final_recipe["base_model_names"])
    missing_base_cols = [name for name in base_model_names if name not in base_oof_df.columns]
    if missing_base_cols:
        raise ValueError(f"OOF matrix is missing base model columns: {missing_base_cols}")

    df = load_data(args.data)
    validate_schema(df)
    y = df[TARGET_COLUMN].astype(int).to_numpy()

    recommended_tree_map = final_recipe.get("recommended_n_estimators", {})
    model_groups = {}
    for model_name in base_model_names:
        recommended_trees = recommended_tree_map.get(model_name)
        logging.info(
            f"Training full-data model group {model_name} "
            f"(recommended_n_estimators={recommended_trees})"
        )
        model_groups[model_name] = fit_full_model_group(
            model_name=model_name,
            df_train=df,
            y_train=y,
            recommended_n_estimators=recommended_trees,
            xgb_n_jobs=args.xgb_n_jobs,
            lgbm_n_jobs=args.lgbm_n_jobs,
            elastic_n_jobs=args.elastic_n_jobs,
        )

    meta_X = base_oof_df[base_model_names].to_numpy(dtype=float)
    meta_y = base_oof_df["y_true"].astype(int).to_numpy()
    combiner = fit_final_combiner(
        recipe_type=final_recipe["recipe_type"],
        base_model_names=base_model_names,
        X_meta=meta_X,
        y_true=meta_y,
    )

    bundle = {
        "bundle_version": 1,
        "data_path": str(Path(args.data).resolve()),
        "results_dir": str(results_dir.resolve()),
        "final_recipe": final_recipe,
        "base_model_names": base_model_names,
        "model_groups": model_groups,
        "combiner": combiner,
        "metadata": {
            "n_samples": int(len(df)),
            "n_ati_positive": int((df[TARGET_COLUMN] == 1).sum()),
            "n_ati_negative": int((df[TARGET_COLUMN] == 0).sum()),
            "prediction_threshold": 0.5,
        },
    }

    metadata = {
        "bundle_version": 1,
        "data_path": str(Path(args.data).resolve()),
        "results_dir": str(results_dir.resolve()),
        "recipe_type": final_recipe["recipe_type"],
        "candidate_name": final_recipe["candidate_name"],
        "base_model_names": base_model_names,
        "n_samples": int(len(df)),
        "n_ati_positive": int((df[TARGET_COLUMN] == 1).sum()),
        "n_ati_negative": int((df[TARGET_COLUMN] == 0).sum()),
        "combiner_training_metrics": combiner["training_metrics"],
        "xgb_n_jobs": args.xgb_n_jobs,
        "lgbm_n_jobs": args.lgbm_n_jobs,
        "elastic_n_jobs": args.elastic_n_jobs,
        "model_groups": {
            model_name: {
                "family": group["family"],
                "feature_set": group["feature_set"],
                "feature_count": int(len(group["feature_cols"])),
                "recommended_n_estimators": group["recommended_n_estimators"],
                "selection_meta": group["selection_meta"],
            }
            for model_name, group in model_groups.items()
        },
    }

    bundle_path = out_dir / "ensemble_bundle.joblib"
    metadata_path = out_dir / "ensemble_metadata.json"
    importance_path = out_dir / "feature_importance_summary.csv"

    joblib.dump(bundle, bundle_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    importance_df = build_feature_importance_table(model_groups)
    importance_df.to_csv(importance_path, index=False)

    print("\n" + "=" * 72)
    print("FINAL ATI ENSEMBLE TRAINING")
    print("=" * 72)
    print(f"Recipe type   : {final_recipe['recipe_type']}")
    print(f"Candidate     : {final_recipe['candidate_name']}")
    print(f"Base models   : {', '.join(base_model_names)}")
    print(f"Samples       : {len(df)}")
    print(
        f"ATI positives : {(df[TARGET_COLUMN] == 1).sum()} | "
        f"No ATI: {(df[TARGET_COLUMN] == 0).sum()}"
    )
    print(f"Bundle saved  : {bundle_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Importance    : {importance_path}")


if __name__ == "__main__":
    main()
