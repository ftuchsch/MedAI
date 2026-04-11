#!/usr/bin/env python3
"""
train.py — Train the final compatible XGBoost model and the current best ensemble bundle
=========================================================================================
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from model import (
    ENSEMBLE_COMPONENTS,
    ENSEMBLE_SPECS,
    RANDOM_SEED,
    SELECTED_LDA_TOP_K,
    SELECTED_PLS_PARAMS,
    SELECTED_PLS_TOP_K,
    SELECTED_RIDGE_TOP_K,
    SELECTED_XGBOOST_TOP_K,
    build_model,
)
from preprocess import build_feature_frame, get_target, load_data, select_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "train.csv"
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "weights"


def parse_args():
    p = argparse.ArgumentParser(description="Train final ATI models and save weights.")
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../data/train.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_WEIGHTS),
        help="Directory to save model weights (default: ./weights/)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for model training (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--xgb-top-k",
        type=int,
        default=SELECTED_XGBOOST_TOP_K,
        help=f"Number of protein features to keep for XGBoost (default: {SELECTED_XGBOOST_TOP_K})",
    )
    p.add_argument(
        "--ridge-top-k",
        type=int,
        default=SELECTED_RIDGE_TOP_K,
        help=f"Number of protein features to keep for ridge LR (default: {SELECTED_RIDGE_TOP_K})",
    )
    p.add_argument(
        "--lda-top-k",
        type=int,
        default=SELECTED_LDA_TOP_K,
        help=f"Number of protein features to keep for shrinkage LDA (default: {SELECTED_LDA_TOP_K})",
    )
    p.add_argument(
        "--pls-top-k",
        type=int,
        default=SELECTED_PLS_TOP_K,
        help=f"Number of protein features to keep for PLS LR (default: {SELECTED_PLS_TOP_K})",
    )
    return p.parse_args()


def save_json(path: Path, payload):
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    y = get_target(df)

    xgb_feature_cols = select_feature_columns(df, y, protein_top_k=args.xgb_top_k)
    ridge_feature_cols = select_feature_columns(df, y, protein_top_k=args.ridge_top_k)
    lda_feature_cols = select_feature_columns(df, y, protein_top_k=args.lda_top_k)
    pls_feature_cols = select_feature_columns(df, y, protein_top_k=args.pls_top_k)

    xgb_model = build_model("selected_xgboost", random_state=args.seed)
    ridge_model = build_model("selected_ridge_lr", random_state=args.seed)
    lda_model = build_model("selected_lda", random_state=args.seed)
    pls_model = build_model("selected_pls_lr", random_state=args.seed)

    X_xgb = build_feature_frame(df, xgb_feature_cols)
    X_ridge = build_feature_frame(df, ridge_feature_cols)
    X_lda = build_feature_frame(df, lda_feature_cols)
    X_pls = build_feature_frame(df, pls_feature_cols)

    logging.info(
        f"Training Selected XGBoost on {len(y)} samples with {len(xgb_feature_cols)} features..."
    )
    xgb_model.fit(X_xgb, y)
    logging.info(
        f"Training Selected Ridge LR on {len(y)} samples with {len(ridge_feature_cols)} features..."
    )
    ridge_model.fit(X_ridge, y)
    logging.info(
        f"Training Selected Shrinkage LDA on {len(y)} samples with {len(lda_feature_cols)} features..."
    )
    lda_model.fit(X_lda, y)
    logging.info(
        f"Training Selected PLS LR on {len(y)} samples with {len(pls_feature_cols)} features..."
    )
    pls_model.fit(X_pls, y)

    xgb_prob = xgb_model.predict_proba(X_xgb)[:, 1]
    ridge_prob = ridge_model.predict_proba(X_ridge)[:, 1]
    lda_prob = lda_model.predict_proba(X_lda)[:, 1]
    pls_prob = pls_model.predict_proba(X_pls)[:, 1]

    component_probabilities = {
        "Selected XGBoost": xgb_prob,
        "Selected Ridge LR": ridge_prob,
        "Selected Shrinkage LDA": lda_prob,
        "Selected PLS LR": pls_prob,
    }
    ensemble_name, ensemble_spec = next(iter(ENSEMBLE_SPECS.items()))
    ensemble_prob = np.zeros(len(y), dtype=float)
    for component in ensemble_spec["components"]:
        ensemble_prob += component["weight"] * component_probabilities[component["name"]]

    training_rows = []
    for model_name, y_prob in [
        ("Selected XGBoost", xgb_prob),
        ("Selected Ridge LR", ridge_prob),
        ("Selected Shrinkage LDA", lda_prob),
        ("Selected PLS LR", pls_prob),
        (ensemble_name, ensemble_prob),
    ]:
        training_rows.append(
            {
                "model": model_name,
                "train_auc": roc_auc_score(y, y_prob),
                "train_log_loss": log_loss(y, y_prob),
            }
        )

    print(f"\n{'=' * 60}")
    print(f"Samples   : {len(y)} (ATI={int(y.sum())}, No ATI={int((y == 0).sum())})")
    print(f"XGB feats : {len(xgb_feature_cols)}")
    print(f"Ridge feats: {len(ridge_feature_cols)}")
    print(f"LDA feats : {len(lda_feature_cols)}")
    print(f"PLS feats : {len(pls_feature_cols)}")
    print(f"{'=' * 60}")
    for row in training_rows:
        print(
            f"{row['model']:<24} Train AUC: {row['train_auc']:.3f} | "
            f"Train LogLoss: {row['train_log_loss']:.3f}"
        )

    xgb_model_path = out_dir / "xgboost_model.json"
    xgb_model.save_model(str(xgb_model_path))
    save_json(out_dir / "feature_cols.json", xgb_feature_cols)
    logging.info(f"Saved model: {xgb_model_path}")
    logging.info(f"Saved features: {out_dir / 'feature_cols.json'}")

    if hasattr(xgb_model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": xgb_feature_cols,
                "gain": xgb_model.feature_importances_,
            }
        ).sort_values("gain", ascending=False)
        importance.to_csv(out_dir / "feature_importance.csv", index=False)
        logging.info(f"Saved feature importance: {out_dir / 'feature_importance.csv'}")

    ridge_model_path = out_dir / "ridge_model.joblib"
    joblib.dump(ridge_model, ridge_model_path)
    save_json(out_dir / "ridge_feature_cols.json", ridge_feature_cols)
    logging.info(f"Saved model: {ridge_model_path}")
    logging.info(f"Saved features: {out_dir / 'ridge_feature_cols.json'}")

    lda_model_path = out_dir / "lda_model.joblib"
    joblib.dump(lda_model, lda_model_path)
    save_json(out_dir / "lda_feature_cols.json", lda_feature_cols)
    logging.info(f"Saved model: {lda_model_path}")
    logging.info(f"Saved features: {out_dir / 'lda_feature_cols.json'}")

    pls_model_path = out_dir / "pls_model.joblib"
    joblib.dump(pls_model, pls_model_path)
    save_json(out_dir / "pls_feature_cols.json", pls_feature_cols)
    logging.info(f"Saved model: {pls_model_path}")
    logging.info(f"Saved features: {out_dir / 'pls_feature_cols.json'}")

    ensemble_config = {
        "prediction_mode": "weighted_average",
        "components": ENSEMBLE_COMPONENTS,
    }
    save_json(out_dir / "ensemble_config.json", ensemble_config)
    save_json(
        out_dir / "training_summary.json",
        {
            "data_path": str(Path(args.data).resolve()),
            "seed": args.seed,
            "xgb_top_k": args.xgb_top_k,
            "ridge_top_k": args.ridge_top_k,
            "lda_top_k": args.lda_top_k,
            "pls_top_k": args.pls_top_k,
            "pls_n_components": SELECTED_PLS_PARAMS["n_components"],
            "ensemble_components": ENSEMBLE_COMPONENTS,
            "training_metrics": training_rows,
        },
    )
    logging.info(f"Saved bundle config: {out_dir / 'ensemble_config.json'}")
    logging.info(f"Saved training summary: {out_dir / 'training_summary.json'}")

    print(f"\nWeights saved to {out_dir}/")
    print("Compatibility note: starter-code inference can still use xgboost_model.json + feature_cols.json.")


if __name__ == "__main__":
    main()
