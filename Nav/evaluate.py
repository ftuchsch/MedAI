#!/usr/bin/env python3
"""
evaluate.py — Repeated leakage-safe CV for Nav's selected ATI models
====================================================================
"""

from __future__ import annotations

import argparse
import itertools
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
)
from sklearn.model_selection import StratifiedKFold

from ensemble_utils import (
    compute_metrics,
    crossfit_blend_predictions,
    fit_model_on_fold,
    optimize_blend_weights,
)
from model import (
    BASE_MODEL_SPECS,
    CV_FOLDS,
    CV_REPEATS,
    DEFAULT_BLEND_MODEL_NAMES,
    DEFAULT_XGB_MODEL_SEEDS,
    RANDOM_SEED,
    get_seed_list,
)
from preprocess import get_target, load_data

# Use a headless backend for command-line and cluster runs.
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "train.csv"
_DEFAULT_OUT = _SCRIPT_DIR / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Run repeated leakage-safe CV for Nav's ATI model set.")
    p.add_argument(
        "--data",
        default=str(_DEFAULT_DATA),
        help="Path to training CSV (default: ../data/train.csv)",
    )
    p.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="Directory for evaluation outputs (default: ./results/)",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=CV_FOLDS,
        help=f"Number of CV folds (default: {CV_FOLDS})",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=CV_REPEATS,
        help=f"Number of shuffled CV repeats (default: {CV_REPEATS})",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Base random seed for CV splitting (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--xgb-model-seeds",
        type=int,
        default=DEFAULT_XGB_MODEL_SEEDS,
        help=(
            "Number of differently-seeded XGBoost fits to average inside each fold "
            f"(default: {DEFAULT_XGB_MODEL_SEEDS})"
        ),
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(BASE_MODEL_SPECS),
        help="Base model names to evaluate (default: all registered Nav base models).",
    )
    p.add_argument(
        "--blend-models",
        nargs="+",
        default=list(DEFAULT_BLEND_MODEL_NAMES),
        help="Model names eligible for learned convex blends (default: four selected branches).",
    )
    p.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="Override inner XGBoost threads per fit (default: model config).",
    )
    return p.parse_args()


def slugify(value: str) -> str:
    slug = value.lower()
    for old, new in [
        (" ", "_"),
        ("(", ""),
        (")", ""),
        ("+", "_plus_"),
        ("[", "_"),
        ("]", ""),
        (",", "_"),
        ("/", "_"),
    ]:
        slug = slug.replace(old, new)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def run_base_cv(
    df: pd.DataFrame,
    *,
    y: np.ndarray,
    sample_ids: np.ndarray,
    model_names: list[str],
    n_folds: int,
    repeats: int,
    base_seed: int,
    xgb_model_seeds: int,
    xgb_n_jobs: int | None = None,
):
    oof_rows = []
    fold_rows = []

    for repeat_seed in get_seed_list(repeats=repeats, base_seed=base_seed):
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df, y), start=1):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)
            y_train = y[train_idx]
            y_val = y[val_idx]

            logging.info(
                "repeat_seed=%s fold=%s/%s train=%s val=%s",
                repeat_seed,
                fold_idx,
                n_folds,
                len(train_idx),
                len(val_idx),
            )

            for model_pos, model_name in enumerate(model_names):
                spec = BASE_MODEL_SPECS[model_name]
                fit_seed = int(repeat_seed + (fold_idx * 1000) + model_pos)
                fitted = fit_model_on_fold(
                    model_name=model_name,
                    spec=spec,
                    df_train=df_train,
                    y_train=y_train,
                    df_val=df_val,
                    y_val=y_val,
                    seed=fit_seed,
                    xgb_model_seeds=xgb_model_seeds,
                    xgb_n_jobs=xgb_n_jobs,
                )
                logging.info(
                    "  %s: log_loss=%.4f auc=%.4f features=%s seeds=%s",
                    model_name,
                    fitted["metrics"]["log_loss"],
                    fitted["metrics"]["roc_auc"],
                    len(fitted["feature_cols"]),
                    fitted["seed_count"],
                )

                fold_rows.append(
                    {
                        "model": model_name,
                        "repeat_seed": int(repeat_seed),
                        "fold": int(fold_idx),
                        "n_samples": int(len(val_idx)),
                        "n_ati": int(y_val.sum()),
                        "n_features": int(len(fitted["feature_cols"])),
                        "seed_count": int(fitted["seed_count"]),
                        "auc": float(fitted["metrics"]["roc_auc"]),
                        "log_loss": float(fitted["metrics"]["log_loss"]),
                        "trees_used": (
                            None if fitted["trees_used"] is None else int(fitted["trees_used"])
                        ),
                    }
                )

                for row_idx, sample_idx in enumerate(val_idx):
                    oof_rows.append(
                        {
                            "sample_id": sample_ids[sample_idx],
                            "ati": int(y[sample_idx]),
                            "repeat_seed": int(repeat_seed),
                            "fold": int(fold_idx),
                            "model": model_name,
                            "prob_ati": float(fitted["val_prob"][row_idx]),
                        }
                    )

    return pd.DataFrame(oof_rows), pd.DataFrame(fold_rows)


def build_oof_mean_table(
    oof_long_df: pd.DataFrame,
    *,
    sample_ids: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    oof_mean_df = (
        oof_long_df.pivot_table(
            index=["sample_id", "ati"],
            columns="model",
            values="prob_ati",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    order_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "ati": y,
            "_row_order": np.arange(len(sample_ids)),
        }
    )
    merged = order_df.merge(oof_mean_df, on=["sample_id", "ati"], how="left")
    merged = merged.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
    return merged


def summarize_base_models(
    *,
    oof_mean_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    y_true = oof_mean_df["ati"].to_numpy()
    rows = []

    for model_name in model_names:
        fold_subset = fold_metrics_df.loc[fold_metrics_df["model"] == model_name]
        metrics = compute_metrics(y_true, oof_mean_df[model_name].to_numpy())
        trees_used = fold_subset["trees_used"].dropna()
        rows.append(
            {
                "model": model_name,
                "family": BASE_MODEL_SPECS[model_name]["family"],
                "artifact_type": BASE_MODEL_SPECS[model_name]["artifact_type"],
                "oof_log_loss": float(metrics["log_loss"]),
                "oof_roc_auc": float(metrics["roc_auc"]),
                "mean_fold_log_loss": float(fold_subset["log_loss"].mean()),
                "std_fold_log_loss": float(fold_subset["log_loss"].std(ddof=1)),
                "mean_fold_auc": float(fold_subset["auc"].mean()),
                "std_fold_auc": float(fold_subset["auc"].std(ddof=1)),
                "median_feature_count": int(fold_subset["n_features"].median()),
                "mean_seed_count": float(fold_subset["seed_count"].mean()),
                "recommended_n_estimators": (
                    None if trees_used.empty else int(np.median(trees_used.to_numpy()))
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("oof_log_loss").reset_index(drop=True)


def evaluate_learned_blends(
    *,
    oof_mean_df: pd.DataFrame,
    blend_model_names: list[str],
    n_folds: int,
    repeats: int,
    base_seed: int,
):
    valid_model_names = [name for name in blend_model_names if name in oof_mean_df.columns]
    if len(valid_model_names) < 2:
        empty = pd.DataFrame(
            columns=[
                "candidate_name",
                "recipe_type",
                "base_model_names",
                "oof_log_loss",
                "oof_roc_auc",
            ]
        )
        return empty, {}, {"blend_weights_by_candidate": {}}

    y_true = oof_mean_df["ati"].to_numpy()
    meta_seed_list = get_seed_list(repeats=repeats, base_seed=base_seed)
    ensemble_rows = []
    candidate_probs = {}
    detail_payload = {"blend_weights_by_candidate": {}}

    for subset_size in range(2, len(valid_model_names) + 1):
        for subset in itertools.combinations(valid_model_names, subset_size):
            candidate_name = f"blend[{'+'.join(subset)}]"
            meta_X = oof_mean_df[list(subset)].to_numpy(dtype=float)
            blend_oof, weight_rows = crossfit_blend_predictions(
                meta_X,
                y_true,
                seed_list=meta_seed_list,
                n_folds=n_folds,
            )
            metrics = compute_metrics(y_true, blend_oof)
            ensemble_rows.append(
                {
                    "candidate_name": candidate_name,
                    "recipe_type": "blend",
                    "base_model_names": ",".join(subset),
                    "oof_log_loss": float(metrics["log_loss"]),
                    "oof_roc_auc": float(metrics["roc_auc"]),
                }
            )
            candidate_probs[candidate_name] = blend_oof
            detail_payload["blend_weights_by_candidate"][candidate_name] = {
                "base_model_names": list(subset),
                "weights_by_fold": weight_rows,
            }

    ensemble_summary_df = (
        pd.DataFrame(ensemble_rows).sort_values("oof_log_loss").reset_index(drop=True)
    )
    return ensemble_summary_df, candidate_probs, detail_payload


def choose_final_recipe(
    *,
    oof_mean_df: pd.DataFrame,
    base_summary_df: pd.DataFrame,
    ensemble_summary_df: pd.DataFrame,
):
    best_base = base_summary_df.sort_values("oof_log_loss").iloc[0].to_dict()
    base_recipe = {
        "recipe_type": "single_model",
        "candidate_name": str(best_base["model"]),
        "base_model_names": [str(best_base["model"])],
        "candidate_oof_log_loss": float(best_base["oof_log_loss"]),
        "candidate_oof_roc_auc": float(best_base["oof_roc_auc"]),
    }

    if ensemble_summary_df.empty:
        return base_recipe

    best_ensemble = ensemble_summary_df.sort_values("oof_log_loss").iloc[0].to_dict()
    if float(best_base["oof_log_loss"]) <= float(best_ensemble["oof_log_loss"]):
        return base_recipe

    base_model_names = str(best_ensemble["base_model_names"]).split(",")
    meta_X = oof_mean_df[base_model_names].to_numpy(dtype=float)
    y_true = oof_mean_df["ati"].to_numpy()
    weights, optimizer_info = optimize_blend_weights(meta_X, y_true)
    return {
        "recipe_type": "blend",
        "candidate_name": str(best_ensemble["candidate_name"]),
        "base_model_names": base_model_names,
        "weights": [float(value) for value in weights],
        "optimizer": optimizer_info,
        "candidate_oof_log_loss": float(best_ensemble["oof_log_loss"]),
        "candidate_oof_roc_auc": float(best_ensemble["oof_roc_auc"]),
    }


def print_metrics(y_true: np.ndarray, y_prob: np.ndarray, title: str):
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, target_names=["No ATI", "ATI"]))
    metrics = compute_metrics(y_true, y_prob)
    if not np.isnan(metrics["roc_auc"]):
        print(f"AUC (ROC):  {metrics['roc_auc']:.3f}")
    print(f"Log loss:   {metrics['log_loss']:.3f}")


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
    logging.info("Saved: %s", out_path)


def print_console_summary(
    base_summary_df: pd.DataFrame,
    ensemble_summary_df: pd.DataFrame,
    final_recipe: dict,
):
    print("\n" + "=" * 72)
    print("NAV ATI EVALUATION SUMMARY")
    print("=" * 72)
    print("\nBase models ranked by repeated OOF log loss:")
    print(
        base_summary_df[
            [
                "model",
                "oof_log_loss",
                "oof_roc_auc",
                "median_feature_count",
                "mean_seed_count",
            ]
        ].to_string(index=False)
    )

    print("\nLearned blend candidates:")
    if ensemble_summary_df.empty:
        print("No blend candidates evaluated.")
    else:
        print(ensemble_summary_df.to_string(index=False))

    print("\nChosen final recipe:")
    print(json.dumps(final_recipe, indent=2))


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_model_names = list(dict.fromkeys(args.models))
    for model_name in selected_model_names:
        if model_name not in BASE_MODEL_SPECS:
            raise KeyError(f"Unknown model '{model_name}'. Choose from: {sorted(BASE_MODEL_SPECS)}")

    df = load_data(args.data)
    y = get_target(df)
    sample_ids = (
        df["sample_id"].astype(str).to_numpy()
        if "sample_id" in df.columns
        else np.arange(len(df)).astype(str)
    )

    oof_long_df, fold_metrics_df = run_base_cv(
        df,
        y=y,
        sample_ids=sample_ids,
        model_names=selected_model_names,
        n_folds=args.folds,
        repeats=args.repeats,
        base_seed=args.seed,
        xgb_model_seeds=args.xgb_model_seeds,
        xgb_n_jobs=args.xgb_n_jobs,
    )
    oof_mean_df = build_oof_mean_table(oof_long_df, sample_ids=sample_ids, y=y)
    base_summary_df = summarize_base_models(
        oof_mean_df=oof_mean_df,
        fold_metrics_df=fold_metrics_df,
        model_names=selected_model_names,
    )

    blend_model_names = [
        model_name
        for model_name in dict.fromkeys(args.blend_models)
        if model_name in selected_model_names
    ]
    ensemble_summary_df, ensemble_prob_map, ensemble_details = evaluate_learned_blends(
        oof_mean_df=oof_mean_df,
        blend_model_names=blend_model_names,
        n_folds=args.folds,
        repeats=args.repeats,
        base_seed=args.seed,
    )
    final_recipe = choose_final_recipe(
        oof_mean_df=oof_mean_df,
        base_summary_df=base_summary_df,
        ensemble_summary_df=ensemble_summary_df,
    )

    recommended_trees = {
        row.model: (
            None if pd.isna(row.recommended_n_estimators) else int(row.recommended_n_estimators)
        )
        for row in base_summary_df.itertuples(index=False)
        if row.family == "xgboost"
    }
    final_recipe["recommended_n_estimators"] = recommended_trees

    if final_recipe["recipe_type"] == "single_model":
        final_prob = oof_mean_df[final_recipe["base_model_names"][0]].to_numpy(dtype=float)
    else:
        final_prob = ensemble_prob_map[final_recipe["candidate_name"]]
    oof_mean_df["prob_final_recipe"] = final_prob

    for model_name in selected_model_names:
        model_prob = oof_mean_df[model_name].to_numpy(dtype=float)
        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)
        print_metrics(
            y,
            model_prob,
            f"{model_name} — {args.repeats}x repeated {args.folds}-fold CV",
        )
        plot_confusion(
            y,
            model_prob,
            out_dir / f"cv_confusion_matrix_{slugify(model_name)}.png",
            title=f"ATI — repeated {args.folds}-fold CV — {model_name}",
        )

    print("\n" + "=" * 60)
    print(f"FINAL RECIPE: {final_recipe['candidate_name']}")
    print("=" * 60)
    print_metrics(
        y,
        final_prob,
        f"{final_recipe['candidate_name']} — repeated {args.folds}-fold CV",
    )
    plot_confusion(
        y,
        final_prob,
        out_dir / "cv_confusion_matrix_final_recipe.png",
        title=f"ATI — repeated {args.folds}-fold CV — {final_recipe['candidate_name']}",
    )

    experiment_summary = {
        "data_path": str(Path(args.data).resolve()),
        "n_samples": int(len(df)),
        "n_features_total": int(
            len(df.columns) - int("ati" in df.columns) - int("sample_id" in df.columns)
        ),
        "n_folds": int(args.folds),
        "n_repeats": int(args.repeats),
        "seed": int(args.seed),
        "xgb_model_seeds": int(args.xgb_model_seeds),
        "xgb_n_jobs": args.xgb_n_jobs,
        "models_evaluated": selected_model_names,
        "blend_models": blend_model_names,
        "label_counts": {
            "no_ati": int((y == 0).sum()),
            "ati": int((y == 1).sum()),
        },
        "final_recipe": final_recipe,
        "base_models_ranked": base_summary_df.to_dict(orient="records"),
        "ensembles_ranked": ensemble_summary_df.to_dict(orient="records"),
    }

    cv_results_path = out_dir / "cv_results.csv"
    oof_long_path = out_dir / "oof_predictions_long.csv"
    oof_mean_path = out_dir / "oof_predictions.csv"
    base_summary_path = out_dir / "base_model_summary.csv"
    ensemble_summary_path = out_dir / "ensemble_summary.csv"
    final_recipe_path = out_dir / "final_recipe.json"
    experiment_summary_path = out_dir / "experiment_summary.json"
    ensemble_details_path = out_dir / "ensemble_details.json"

    fold_metrics_df.to_csv(cv_results_path, index=False)
    oof_long_df.to_csv(oof_long_path, index=False)
    oof_mean_df.to_csv(oof_mean_path, index=False)
    base_summary_df.to_csv(base_summary_path, index=False)
    ensemble_summary_df.to_csv(ensemble_summary_path, index=False)
    with final_recipe_path.open("w") as f:
        json.dump(final_recipe, f, indent=2)
    with experiment_summary_path.open("w") as f:
        json.dump(experiment_summary, f, indent=2)
    with ensemble_details_path.open("w") as f:
        json.dump(ensemble_details, f, indent=2)

    logging.info("Saved: %s", cv_results_path)
    logging.info("Saved: %s", oof_long_path)
    logging.info("Saved: %s", oof_mean_path)
    logging.info("Saved: %s", base_summary_path)
    logging.info("Saved: %s", ensemble_summary_path)
    logging.info("Saved: %s", final_recipe_path)
    logging.info("Saved: %s", experiment_summary_path)
    logging.info("Saved: %s", ensemble_details_path)

    print_console_summary(base_summary_df, ensemble_summary_df, final_recipe)


if __name__ == "__main__":
    main()
