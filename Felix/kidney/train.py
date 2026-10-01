#!/usr/bin/env python3
"""
train.py — Train the TabPFN kidney model on all BKBC samples
============================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROBABILITY_SHRINKAGE,
    DEFAULT_PROTEIN_TOP_K,
    DEFAULT_TABPFN_AVERAGE_BEFORE_SOFTMAX,
    DEFAULT_TABPFN_BALANCE_PROBABILITIES,
    DEFAULT_TABPFN_DEVICE,
    DEFAULT_TABPFN_N_ESTIMATORS,
    DEFAULT_TABPFN_SOFTMAX_TEMPERATURE,
    FEATURES_FILENAME,
    METADATA_FILENAME,
    build_model,
    get_model_spec,
    get_tabpfn_version,
)
from preprocess import (
    build_features_and_labels,
    load_data,
    score_proteins,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent.parent / "data" / "train.csv"
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "weights"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the TabPFN ATI model and save a deployable bundle."
    )
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../../data/train.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_WEIGHTS),
        help="Directory to save model weights (default: ./weights/)",
    )
    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Model to train (default: {DEFAULT_MODEL_NAME})",
    )
    p.add_argument(
        "--protein-top-k",
        type=int,
        default=DEFAULT_PROTEIN_TOP_K,
        help=f"Number of protein features to keep before TabPFN (default: {DEFAULT_PROTEIN_TOP_K})",
    )
    p.add_argument(
        "--device",
        default=DEFAULT_TABPFN_DEVICE,
        help=f"TabPFN device string (default: {DEFAULT_TABPFN_DEVICE})",
    )
    p.add_argument(
        "--tabpfn-n-estimators",
        type=int,
        default=DEFAULT_TABPFN_N_ESTIMATORS,
        help=f"Number of TabPFN ensemble estimators (default: {DEFAULT_TABPFN_N_ESTIMATORS})",
    )
    p.add_argument(
        "--softmax-temperature",
        type=float,
        default=DEFAULT_TABPFN_SOFTMAX_TEMPERATURE,
        help=f"TabPFN softmax temperature (default: {DEFAULT_TABPFN_SOFTMAX_TEMPERATURE})",
    )
    p.add_argument(
        "--average-before-softmax",
        action="store_true",
        default=DEFAULT_TABPFN_AVERAGE_BEFORE_SOFTMAX,
        help="Average logits before softmax for TabPFN ensembling.",
    )
    p.add_argument(
        "--balance-probabilities",
        action="store_true",
        default=DEFAULT_TABPFN_BALANCE_PROBABILITIES,
        help="Reweight probabilities by the training class distribution.",
    )
    p.add_argument(
        "--probability-shrinkage",
        type=float,
        default=DEFAULT_PROBABILITY_SHRINKAGE,
        help="Blend predicted probabilities toward the full-training prevalence. 0 means disabled.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = get_model_spec(args.model_name)

    df = load_data(args.data)
    X, y, feature_cols = build_features_and_labels(
        df,
        protein_top_k=args.protein_top_k,
    )

    model = build_model(
        args.model_name,
        device=args.device,
        n_estimators=args.tabpfn_n_estimators,
        softmax_temperature=args.softmax_temperature,
        average_before_softmax=args.average_before_softmax,
        balance_probabilities=args.balance_probabilities,
    )
    logging.info(f"Training {args.model_name} on {len(y)} samples...")
    model.fit(X, y)

    y_prob = model.predict_proba(X)[:, 1]
    if args.probability_shrinkage > 0:
        train_prevalence = float(y.mean())
        y_prob = (
            1.0 - args.probability_shrinkage
        ) * y_prob + args.probability_shrinkage * train_prevalence
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y, y_prob)
    ll = log_loss(y, y_prob)

    n_proteins = sum(col.startswith("feature_") for col in feature_cols)
    n_clinical = len(feature_cols) - n_proteins

    print(f"\n{'=' * 58}")
    print(f"  Model         : {args.model_name}")
    print(f"  Samples       : {len(y)} (ATI={y.sum()}, No ATI={(y == 0).sum()})")
    print(f"  Features      : {len(feature_cols)} ({n_proteins} protein + {n_clinical} clinical)")
    print(f"  TabPFN ver.   : {get_tabpfn_version()}")
    print(f"  Device        : {args.device}")
    print(f"  Estimators    : {args.tabpfn_n_estimators}")
    print(f"  Softmax T     : {args.softmax_temperature}")
    print(f"  Avg pre-softm : {args.average_before_softmax}")
    print(f"  Prob shrink   : {args.probability_shrinkage}")
    print(f"  Train AUC     : {auc:.3f}")
    print(f"  Train LogLoss : {ll:.3f}")
    print(f"{'=' * 58}")

    model._kidney_probability_shrinkage = float(args.probability_shrinkage)
    model._kidney_train_prevalence = float(y.mean())

    model_path = out_dir / spec.model_file
    joblib.dump(model, model_path, compress=3)
    logging.info(f"Saved model: {model_path}")

    feature_path = out_dir / FEATURES_FILENAME
    with feature_path.open("w") as f:
        json.dump(feature_cols, f, indent=2)
    logging.info(f"Saved feature list: {feature_path}")

    metadata = {
        "model_name": args.model_name,
        "artifact_type": spec.artifact_type,
        "model_file": spec.model_file,
        "feature_file": FEATURES_FILENAME,
        "protein_top_k": int(args.protein_top_k),
        "n_features": int(len(feature_cols)),
        "n_protein_features": int(n_proteins),
        "n_clinical_features": int(n_clinical),
        "device": args.device,
        "tabpfn_n_estimators": int(args.tabpfn_n_estimators),
        "softmax_temperature": float(args.softmax_temperature),
        "average_before_softmax": bool(args.average_before_softmax),
        "balance_probabilities": bool(args.balance_probabilities),
        "probability_shrinkage": float(args.probability_shrinkage),
        "tabpfn_version": get_tabpfn_version(),
        "data_path": str(Path(args.data).resolve()),
        "train_auc": float(auc),
        "train_log_loss": float(ll),
    }
    metadata_path = out_dir / METADATA_FILENAME
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata: {metadata_path}")

    ranking = score_proteins(df, y)
    selected_proteins = {c for c in feature_cols if c.startswith("feature_")}
    if not ranking.empty:
        ranking["selected"] = ranking["feature"].isin(selected_proteins)
        ranking_path = out_dir / "feature_selection.csv"
        ranking.to_csv(ranking_path, index=False)
        logging.info(f"Saved feature ranking: {ranking_path}")

        print("\nTop 10 selected proteins by ANOVA F-score:")
        print(
            ranking[ranking["selected"]]
            .head(10)
            .to_string(index=False, formatters={"score": "{:.4f}".format})
        )

    print(f"\nBundle saved to {out_dir}/")
    print("Next step: bash predict.sh /path/to/new_data.csv")


if __name__ == "__main__":
    main()
