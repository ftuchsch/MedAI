#!/usr/bin/env python3
"""
evaluate.py — Leak-safe cross-validation for model development
==============================================================
"""

import argparse
import logging
from pathlib import Path

import matplotlib
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import (
    BASE_MODEL_SPECS,
    CV_FOLDS,
    ENSEMBLE_SPECS,
    RANDOM_SEED,
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
_DEFAULT_OUT = _SCRIPT_DIR / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Run leak-safe CV on BKBC training data.")
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../data/train.csv)",
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
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for CV splitting and model initialization (default: {RANDOM_SEED})",
    )
    return p.parse_args()


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    ll = log_loss(y_true, y_prob)
    return auc, ll


def run_base_model_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    model_name: str,
    spec: dict,
    cv: StratifiedKFold,
    random_state: int,
):
    logging.info(f"[{model_name}] Running {cv.get_n_splits()}-fold CV...")
    oof_prob = np.zeros(len(df), dtype=float)
    fold_rows = []
    fold_indices = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df, y)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_val = y[val_idx]

        feature_cols = select_feature_columns(
            train_df,
            y_train,
            protein_top_k=spec["protein_top_k"],
        )
        X_train = build_feature_frame(train_df, feature_cols)
        X_val = build_feature_frame(val_df, feature_cols)

        model = build_model(spec["builder"], random_state=random_state)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        oof_prob[val_idx] = y_prob

        fold_auc, fold_ll = evaluate_predictions(y_val, y_prob)
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold_idx,
                "n_samples": len(val_idx),
                "n_ati": int(y_val.sum()),
                "n_features": len(feature_cols),
                "auc": fold_auc,
                "log_loss": fold_ll,
            }
        )
        fold_indices.append(np.asarray(val_idx))

    return {
        "oof_prob": oof_prob,
        "fold_rows": fold_rows,
        "fold_indices": fold_indices,
    }


def build_ensemble_results(
    ensemble_name: str,
    ensemble_spec: dict,
    base_results: dict[str, dict],
    y: np.ndarray,
):
    oof_prob = np.zeros(len(y), dtype=float)
    for component in ensemble_spec["components"]:
        member_name = component["name"]
        oof_prob += component["weight"] * base_results[member_name]["oof_prob"]

    fold_rows = []
    reference_indices = base_results[ensemble_spec["components"][0]["name"]]["fold_indices"]
    for fold_idx, val_idx in enumerate(reference_indices):
        y_val = y[val_idx]
        y_prob = oof_prob[val_idx]
        fold_auc, fold_ll = evaluate_predictions(y_val, y_prob)
        fold_rows.append(
            {
                "model": ensemble_name,
                "fold": fold_idx,
                "n_samples": len(val_idx),
                "n_ati": int(y_val.sum()),
                "n_features": np.nan,
                "auc": fold_auc,
                "log_loss": fold_ll,
            }
        )

    return {"oof_prob": oof_prob, "fold_rows": fold_rows}


def print_metrics(y_true: np.ndarray, y_prob: np.ndarray, title: str):
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
    auc, ll = evaluate_predictions(y_true, y_prob)
    if not np.isnan(auc):
        print(f"AUC (ROC):  {auc:.3f}")
    print(f"Log loss:   {ll:.3f}")


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str):
    y_pred = (y_prob >= 0.5).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["No ATI", "ATI"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved: {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    y = get_target(df)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    base_results = {}
    all_rows = []
    oof_df = pd.DataFrame({"sample_id": df.get("sample_id", pd.RangeIndex(len(df))), "ati": y})

    for model_name, spec in BASE_MODEL_SPECS.items():
        results = run_base_model_cv(
            df=df,
            y=y,
            model_name=model_name,
            spec=spec,
            cv=cv,
            random_state=args.seed,
        )
        base_results[model_name] = results
        all_rows.extend(results["fold_rows"])
        slug = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        oof_df[f"prob_{slug}"] = results["oof_prob"]
        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)
        print_metrics(y, results["oof_prob"], f"{model_name} — {args.folds}-Fold CV")
        plot_confusion(
            y,
            results["oof_prob"],
            out_dir / f"cv_confusion_matrix_{slug}.png",
            title=f"ATI — {args.folds}-Fold CV — {model_name}",
        )

    for ensemble_name, ensemble_spec in ENSEMBLE_SPECS.items():
        results = build_ensemble_results(
            ensemble_name=ensemble_name,
            ensemble_spec=ensemble_spec,
            base_results=base_results,
            y=y,
        )
        all_rows.extend(results["fold_rows"])
        slug = ensemble_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        oof_df[f"prob_{slug}"] = results["oof_prob"]
        print("\n" + "=" * 60)
        print(f"MODEL: {ensemble_name}")
        print("=" * 60)
        print_metrics(y, results["oof_prob"], f"{ensemble_name} — {args.folds}-Fold CV")
        plot_confusion(
            y,
            results["oof_prob"],
            out_dir / f"cv_confusion_matrix_{slug}.png",
            title=f"ATI — {args.folds}-Fold CV — {ensemble_name}",
        )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(out_dir / "cv_results.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    logging.info(f"Saved: {out_dir / 'cv_results.csv'}")
    logging.info(f"Saved: {out_dir / 'oof_predictions.csv'}")

    summary = (
        results_df.groupby("model")[["auc", "log_loss"]]
        .agg(["mean", "std"])
        .sort_values(("log_loss", "mean"))
    )

    print("\n" + "=" * 72)
    print(f"{'Model':<28} | {'Mean AUC':>10} | {'Std AUC':>10} | {'Mean LogLoss':>13}")
    print("-" * 72)
    for model_name, row in summary.iterrows():
        print(
            f"{model_name:<28} | "
            f"{row[('auc', 'mean')]:>10.3f} | "
            f"{row[('auc', 'std')]:>10.3f} | "
            f"{row[('log_loss', 'mean')]:>13.3f}"
        )
    print("=" * 72)
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
