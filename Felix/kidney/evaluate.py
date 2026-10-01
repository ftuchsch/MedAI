#!/usr/bin/env python3
"""
evaluate.py — Fold-safe cross-validation for the TabPFN kidney model
====================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from model import (
    CV_FOLDS,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROBABILITY_SHRINKAGE,
    DEFAULT_PROTEIN_TOP_K,
    DEFAULT_TABPFN_AVERAGE_BEFORE_SOFTMAX,
    DEFAULT_TABPFN_BALANCE_PROBABILITIES,
    DEFAULT_TABPFN_DEVICE,
    DEFAULT_TABPFN_N_ESTIMATORS,
    DEFAULT_TABPFN_SOFTMAX_TEMPERATURE,
    RANDOM_SEED,
    build_model,
    get_model_names,
)
from preprocess import (
    build_feature_frame,
    get_target,
    load_data,
    select_feature_columns,
)

# Use a headless backend for command-line and cluster runs.
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent.parent / "data" / "train.csv"
_DEFAULT_OUT = _SCRIPT_DIR / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Run fold-safe TabPFN cross-validation on BKBC data.")
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../../data/train.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Directory for CV outputs (default: ./results/)",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=CV_FOLDS,
        help=f"Number of CV folds (default: {CV_FOLDS})",
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
        help="Blend predicted probabilities toward the training prevalence. 0 means disabled.",
    )
    return p.parse_args()


def run_cv(
    df: pd.DataFrame,
    *,
    model_name: str,
    n_folds: int,
    protein_top_k: int | None,
    device: str,
    tabpfn_n_estimators: int,
    softmax_temperature: float,
    average_before_softmax: bool,
    balance_probabilities: bool,
    probability_shrinkage: float,
):
    """Run manual stratified CV so feature selection happens inside each fold."""
    y = get_target(df)
    sample_ids = (
        df["sample_id"].astype(str).reset_index(drop=True)
        if "sample_id" in df.columns
        else pd.Series(np.arange(len(df)), name="sample_id")
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    y_prob = np.zeros(len(df), dtype=float)
    fold_results: list[dict] = []
    feature_sets: list[dict] = []

    logging.info(f"[{model_name}] Running {n_folds}-fold CV with protein_top_k={protein_top_k}...")

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df, y), start=1):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_val = y[val_idx]

        feature_cols = select_feature_columns(
            df_train,
            y_train,
            protein_top_k=protein_top_k,
        )
        X_train = build_feature_frame(df_train, feature_cols).to_numpy()
        X_val = build_feature_frame(df_val, feature_cols).to_numpy()

        n_proteins = sum(col.startswith("feature_") for col in feature_cols)
        n_clinical = len(feature_cols) - n_proteins
        logging.info(
            f"[{model_name}] Fold {fold_idx}/{n_folds}: "
            f"{n_proteins} protein + {n_clinical} clinical = {len(feature_cols)} features"
        )

        model = build_model(
            model_name,
            device=device,
            n_estimators=tabpfn_n_estimators,
            softmax_temperature=softmax_temperature,
            average_before_softmax=average_before_softmax,
            balance_probabilities=balance_probabilities,
        )
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        if probability_shrinkage > 0:
            train_prevalence = float(np.mean(y_train))
            p_val = (1.0 - probability_shrinkage) * p_val + probability_shrinkage * train_prevalence
        p_val = np.clip(p_val, 1e-6, 1 - 1e-6)
        y_prob[val_idx] = p_val

        fold_auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else float("nan")
        fold_ll = log_loss(y_val, p_val)
        fold_results.append(
            {
                "model": model_name,
                "fold": fold_idx,
                "n_samples": len(val_idx),
                "n_ati": int(y_val.sum()),
                "n_features": len(feature_cols),
                "n_protein_features": n_proteins,
                "n_clinical_features": n_clinical,
                "tabpfn_n_estimators": tabpfn_n_estimators,
                "softmax_temperature": softmax_temperature,
                "average_before_softmax": average_before_softmax,
                "balance_probabilities": balance_probabilities,
                "probability_shrinkage": probability_shrinkage,
                "auc": fold_auc,
                "log_loss": fold_ll,
            }
        )
        feature_sets.append(
            {
                "model": model_name,
                "fold": fold_idx,
                "protein_top_k_requested": protein_top_k,
                "n_features": len(feature_cols),
                "n_protein_features": n_proteins,
                "n_clinical_features": n_clinical,
                "tabpfn_n_estimators": tabpfn_n_estimators,
                "softmax_temperature": softmax_temperature,
                "average_before_softmax": average_before_softmax,
                "balance_probabilities": balance_probabilities,
                "probability_shrinkage": probability_shrinkage,
                "feature_cols": feature_cols,
            }
        )

    y_pred = (y_prob >= 0.5).astype(int)
    oof_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_label": y,
            "prob_ati": y_prob,
            "pred_label": y_pred,
        }
    )
    return y, y_pred, y_prob, fold_results, feature_sets, oof_df


def print_metrics(y, y_pred, y_prob, title):
    """Print classification metrics."""
    print(f"\n=== {title} ===")
    print(classification_report(y, y_pred, target_names=["No ATI", "ATI"]))
    if len(np.unique(y)) > 1:
        print(f"AUC (ROC):  {roc_auc_score(y, y_prob):.3f}")
    print(f"Log loss:   {log_loss(y, y_prob):.3f}")


def plot_confusion_matrix(y, y_pred, out_path, title, n_folds):
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y, y_pred),
        display_labels=["No ATI", "ATI"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"ATI — {n_folds}-Fold CV — {title}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    logging.info(f"Saved: {out_path}")
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    all_fold_results: list[dict] = []

    for model_name in get_model_names():
        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)

        y, y_pred, y_prob, fold_results, feature_sets, oof_df = run_cv(
            df,
            model_name=model_name,
            n_folds=args.folds,
            protein_top_k=args.protein_top_k,
            device=args.device,
            tabpfn_n_estimators=args.tabpfn_n_estimators,
            softmax_temperature=args.softmax_temperature,
            average_before_softmax=args.average_before_softmax,
            balance_probabilities=args.balance_probabilities,
            probability_shrinkage=args.probability_shrinkage,
        )
        all_fold_results.extend(fold_results)

        print_metrics(y, y_pred, y_prob, f"{model_name} — {args.folds}-Fold CV")

        slug = model_name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y,
            y_pred,
            out_dir / f"cv_confusion_matrix_{slug}.png",
            title=model_name,
            n_folds=args.folds,
        )

        oof_path = out_dir / f"oof_predictions_{slug}.csv"
        oof_df.to_csv(oof_path, index=False)
        logging.info(f"Saved: {oof_path}")

        feature_path = out_dir / f"cv_feature_sets_{slug}.json"
        with feature_path.open("w") as f:
            json.dump(feature_sets, f, indent=2)
        logging.info(f"Saved: {feature_path}")

    results_df = pd.DataFrame(all_fold_results)
    results_path = out_dir / "cv_results.csv"
    results_df.to_csv(results_path, index=False)
    logging.info(f"Saved: {results_path}")

    print("\n" + "=" * 72)
    print(f"{'Model':<12} | {'Mean AUC':>10} | {'Std AUC':>10} | {'Mean LogLoss':>13}")
    print("-" * 72)
    for model_name in get_model_names():
        rows = results_df[results_df["model"] == model_name]
        print(
            f"{model_name:<12} | "
            f"{rows['auc'].mean():>10.3f} | "
            f"{rows['auc'].std():>10.3f} | "
            f"{rows['log_loss'].mean():>13.3f}"
        )
    print("=" * 72)
    print(f"\nDefault deployment target: {DEFAULT_MODEL_NAME}")
    print(f"All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
