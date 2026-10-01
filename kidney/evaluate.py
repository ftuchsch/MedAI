#!/usr/bin/env python3
"""
evaluate.py — Repeated leakage-safe CV for the ATI ensemble pipeline
====================================================================
Runs:

- repeated stratified OOF evaluation
- fold-internal feature selection
- XGBoost / LightGBM / elastic-net base models
- clinical/eGFR stress tests
- second-stage blend / stack / calibrated-stack evaluation
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble_utils import (
    compute_metrics,
    crossfit_blend_predictions,
    crossfit_meta_predictions,
    fit_model_on_fold,
)
from model import (
    CV_FOLDS,
    CV_REPEATS,
    DEFAULT_EVALUATION_MODEL_NAMES,
    ID_COLUMN,
    TARGET_COLUMN,
    TREE_MODEL_FAMILIES,
    get_model_config,
    get_seed_list,
    get_splitter,
)
from preprocess import load_data, validate_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "train.csv"
_DEFAULT_OUT = _SCRIPT_DIR / "results"
_WORKER_DF = None
_WORKER_Y = None
_WORKER_SAMPLE_IDS = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Run repeated leakage-safe CV for the ATI ensemble pipeline."
    )
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
        "--models",
        nargs="+",
        default=list(DEFAULT_EVALUATION_MODEL_NAMES),
        help="Specific model names to evaluate (default: all base models plus diagnostics).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of repeat/fold tasks to evaluate in parallel (default: 1).",
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


def _evaluate_single_fold(
    *,
    df: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    model_names: list[str],
    repeat_seed: int,
    fold_idx: int,
    n_folds: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    xgb_n_jobs: int | None = None,
    lgbm_n_jobs: int | None = None,
    elastic_n_jobs: int | None = None,
):
    """Fit all requested models for one repeat/fold pair."""
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_val = y[val_idx]
    feature_cache = {}

    oof_rows = []
    fold_rows = []

    logging.info(
        f"Repeat seed={repeat_seed} fold={fold_idx}/{n_folds} "
        f"train={len(train_idx)} val={len(val_idx)}"
    )

    for model_name in model_names:
        # Python's hash() changes between processes; use a stable model offset.
        fit_seed = repeat_seed + (fold_idx * 100) + zlib.crc32(model_name.encode()) % 97
        logging.info(f"  Fitting {model_name} (seed={fit_seed})")
        fitted = fit_model_on_fold(
            model_name=model_name,
            df_train=df_train,
            y_train=y_train,
            df_val=df_val,
            y_val=y_val,
            seed=fit_seed,
            feature_cache=feature_cache,
            xgb_n_jobs=xgb_n_jobs,
            lgbm_n_jobs=lgbm_n_jobs,
            elastic_n_jobs=elastic_n_jobs,
        )
        logging.info(
            f"    {model_name}: log_loss={fitted['metrics']['log_loss']:.4f} "
            f"auc={fitted['metrics']['roc_auc']:.4f} "
            f"features={len(fitted['feature_cols'])} "
            f"seeds={fitted['seed_count']}"
        )
        config = get_model_config(model_name)
        fold_rows.append(
            {
                "model_name": model_name,
                "model_family": config.family,
                "role": config.role,
                "repeat_seed": int(repeat_seed),
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_val_ati": int(y_val.sum()),
                "seed_count": int(fitted["seed_count"]),
                "feature_count": int(len(fitted["feature_cols"])),
                "log_loss": float(fitted["metrics"]["log_loss"]),
                "roc_auc": float(fitted["metrics"]["roc_auc"]),
                "best_iteration": (
                    None if fitted["best_iteration"] is None else int(fitted["best_iteration"])
                ),
                "trees_used": None if fitted["trees_used"] is None else int(fitted["trees_used"]),
            }
        )

        for row_idx, sample_idx in enumerate(val_idx):
            oof_rows.append(
                {
                    "sample_id": sample_ids[sample_idx],
                    "y_true": int(y[sample_idx]),
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold_idx),
                    "model_name": model_name,
                    "prob_ati": float(fitted["val_prob"][row_idx]),
                }
            )

    return {
        "repeat_seed": int(repeat_seed),
        "fold": int(fold_idx),
        "oof_rows": oof_rows,
        "fold_rows": fold_rows,
    }


def _init_eval_worker(data_path: str):
    """Load the shared training frame once per worker process."""
    global _WORKER_DF, _WORKER_Y, _WORKER_SAMPLE_IDS
    _WORKER_DF = load_data(data_path)
    validate_schema(_WORKER_DF)
    _WORKER_Y = _WORKER_DF[TARGET_COLUMN].astype(int).to_numpy()
    _WORKER_SAMPLE_IDS = _WORKER_DF[ID_COLUMN].astype(str).to_numpy()


def _run_fold_task(task: dict):
    """Process-pool entrypoint for one repeat/fold task."""
    return _evaluate_single_fold(
        df=_WORKER_DF,
        y=_WORKER_Y,
        sample_ids=_WORKER_SAMPLE_IDS,
        **task,
    )


def run_base_cv(
    df: pd.DataFrame,
    *,
    model_names: list[str],
    n_folds: int,
    repeats: int,
    data_path: str | None = None,
    workers: int = 1,
    xgb_n_jobs: int | None = None,
    lgbm_n_jobs: int | None = None,
    elastic_n_jobs: int | None = None,
):
    """Run repeated CV and collect long-form OOF predictions."""
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    sample_ids = df[ID_COLUMN].astype(str).to_numpy()
    seed_list = get_seed_list(repeats=repeats)
    oof_rows = []
    fold_rows = []
    tasks = []

    for repeat_seed in seed_list:
        splitter = get_splitter(repeat_seed, n_splits=n_folds)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(df, y), start=1):
            tasks.append(
                {
                    "model_names": list(model_names),
                    "repeat_seed": int(repeat_seed),
                    "fold_idx": int(fold_idx),
                    "n_folds": int(n_folds),
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "xgb_n_jobs": xgb_n_jobs,
                    "lgbm_n_jobs": lgbm_n_jobs,
                    "elastic_n_jobs": elastic_n_jobs,
                }
            )

    worker_count = max(1, int(workers))
    if worker_count == 1:
        for task in tasks:
            result = _evaluate_single_fold(
                df=df,
                y=y,
                sample_ids=sample_ids,
                **task,
            )
            oof_rows.extend(result["oof_rows"])
            fold_rows.extend(result["fold_rows"])
    else:
        if data_path is None:
            raise ValueError("data_path is required when workers > 1.")
        data_path = str(Path(data_path).resolve())
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(tasks)),
            initializer=_init_eval_worker,
            initargs=(data_path,),
        ) as executor:
            for result in executor.map(_run_fold_task, tasks):
                oof_rows.extend(result["oof_rows"])
                fold_rows.extend(result["fold_rows"])

    oof_long_df = pd.DataFrame(oof_rows)
    fold_metrics_df = pd.DataFrame(fold_rows)
    return oof_long_df, fold_metrics_df


def build_oof_mean_table(oof_long_df: pd.DataFrame) -> pd.DataFrame:
    """Average repeated OOF predictions per sample/model."""
    oof_mean_df = (
        oof_long_df.pivot_table(
            index=["sample_id", "y_true"],
            columns="model_name",
            values="prob_ati",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    oof_mean_df.columns.name = None
    return oof_mean_df


def summarize_base_models(
    *,
    oof_mean_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    model_names: list[str],
) -> pd.DataFrame:
    """Create one summary row per evaluated model."""
    y_true = oof_mean_df["y_true"].to_numpy()
    rows = []

    for model_name in model_names:
        config = get_model_config(model_name)
        fold_subset = fold_metrics_df.loc[fold_metrics_df["model_name"] == model_name]
        metrics = compute_metrics(y_true, oof_mean_df[model_name].to_numpy())
        trees_used = fold_subset["trees_used"].dropna()

        rows.append(
            {
                "model_name": model_name,
                "model_family": config.family,
                "role": config.role,
                "feature_set": config.feature_set,
                "n_seeds": int(config.n_seeds),
                "mean_fold_log_loss": float(fold_subset["log_loss"].mean()),
                "std_fold_log_loss": float(fold_subset["log_loss"].std(ddof=1)),
                "mean_fold_roc_auc": float(fold_subset["roc_auc"].mean()),
                "std_fold_roc_auc": float(fold_subset["roc_auc"].std(ddof=1)),
                "median_feature_count": int(fold_subset["feature_count"].median()),
                "recommended_n_estimators": (
                    None if trees_used.empty else int(np.median(trees_used.to_numpy()))
                ),
                "oof_log_loss": float(metrics["log_loss"]),
                "oof_roc_auc": float(metrics["roc_auc"]),
            }
        )

    return pd.DataFrame(rows).sort_values("oof_log_loss").reset_index(drop=True)


def evaluate_ensembles(
    *,
    oof_mean_df: pd.DataFrame,
    ensemble_model_names: list[str],
    n_folds: int,
    repeats: int,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate blend / stack candidates on the selected-model OOF matrix."""
    if len(ensemble_model_names) < 2:
        empty = pd.DataFrame(
            columns=[
                "candidate_name",
                "recipe_type",
                "base_model_names",
                "oof_log_loss",
                "oof_roc_auc",
            ]
        )
        return empty, {"blend_weights_by_fold": []}

    y_true = oof_mean_df["y_true"].to_numpy()
    meta_seed_list = get_seed_list(repeats=repeats)

    ensemble_rows = []
    blend_weight_rows = []

    for subset_size in range(2, len(ensemble_model_names) + 1):
        for subset in itertools.combinations(ensemble_model_names, subset_size):
            subset_label = "+".join(subset)
            meta_X = oof_mean_df[list(subset)].to_numpy(dtype=float)

            blend_oof, subset_blend_weights = crossfit_blend_predictions(
                meta_X,
                y_true,
                seed_list=meta_seed_list,
                n_folds=n_folds,
            )
            stack_oof = crossfit_meta_predictions(
                meta_X,
                y_true,
                seed_list=meta_seed_list,
                calibrated=False,
                n_folds=n_folds,
            )
            stack_cal_oof = crossfit_meta_predictions(
                meta_X,
                y_true,
                seed_list=meta_seed_list,
                calibrated=True,
                n_folds=n_folds,
            )

            for row in subset_blend_weights:
                blend_weight_rows.append(
                    {
                        "base_model_names": list(subset),
                        **row,
                    }
                )

            for candidate_name, recipe_type, prob in [
                (f"blend[{subset_label}]", "blend", blend_oof),
                (f"stack_ridge[{subset_label}]", "stack_ridge", stack_oof),
                (f"stack_calibrated[{subset_label}]", "stack_calibrated", stack_cal_oof),
            ]:
                metrics = compute_metrics(y_true, prob)
                ensemble_rows.append(
                    {
                        "candidate_name": candidate_name,
                        "recipe_type": recipe_type,
                        "base_model_names": ",".join(subset),
                        "oof_log_loss": float(metrics["log_loss"]),
                        "oof_roc_auc": float(metrics["roc_auc"]),
                    }
                )

    return pd.DataFrame(ensemble_rows).sort_values("oof_log_loss").reset_index(drop=True), {
        "blend_weights_by_fold": blend_weight_rows
    }


def choose_final_recipe(
    *,
    base_summary_df: pd.DataFrame,
    ensemble_summary_df: pd.DataFrame,
) -> dict:
    """Pick the best final recipe by OOF log loss."""
    base_only_df = base_summary_df.loc[base_summary_df["role"] == "base"].sort_values(
        "oof_log_loss"
    )
    if base_only_df.empty:
        raise ValueError("At least one base model is required to choose a final recipe.")
    best_base = base_only_df.iloc[0].to_dict()

    if ensemble_summary_df.empty:
        return {
            "recipe_type": "single_model",
            "candidate_name": best_base["model_name"],
            "base_model_names": [best_base["model_name"]],
            "selection_metric": "oof_log_loss",
            "candidate_oof_log_loss": float(best_base["oof_log_loss"]),
            "candidate_oof_roc_auc": float(best_base["oof_roc_auc"]),
        }

    best_ensemble = ensemble_summary_df.sort_values("oof_log_loss").iloc[0].to_dict()
    if float(best_base["oof_log_loss"]) <= float(best_ensemble["oof_log_loss"]):
        return {
            "recipe_type": "single_model",
            "candidate_name": best_base["model_name"],
            "base_model_names": [best_base["model_name"]],
            "selection_metric": "oof_log_loss",
            "candidate_oof_log_loss": float(best_base["oof_log_loss"]),
            "candidate_oof_roc_auc": float(best_base["oof_roc_auc"]),
        }

    return {
        "recipe_type": best_ensemble["recipe_type"],
        "candidate_name": best_ensemble["candidate_name"],
        "base_model_names": str(best_ensemble["base_model_names"]).split(","),
        "selection_metric": "oof_log_loss",
        "candidate_oof_log_loss": float(best_ensemble["oof_log_loss"]),
        "candidate_oof_roc_auc": float(best_ensemble["oof_roc_auc"]),
    }


def print_console_summary(
    base_summary_df: pd.DataFrame, ensemble_summary_df: pd.DataFrame, final_recipe: dict
):
    """Print the most useful experiment outputs."""
    print("\n" + "=" * 72)
    print("ATI ENSEMBLE EVALUATION SUMMARY")
    print("=" * 72)
    print("\nTop base models by OOF log loss:")
    print(
        base_summary_df[
            [
                "model_name",
                "role",
                "feature_set",
                "oof_log_loss",
                "oof_roc_auc",
                "median_feature_count",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )

    print("\nEnsemble candidates:")
    if ensemble_summary_df.empty:
        print("No ensemble candidates evaluated.")
    else:
        print(ensemble_summary_df.to_string(index=False))

    print("\nChosen final recipe:")
    print(json.dumps(final_recipe, indent=2))

    ablation_rows = base_summary_df.loc[base_summary_df["role"] == "diagnostic"]
    if not ablation_rows.empty:
        print("\nClinical / proxy diagnostics:")
        print(
            ablation_rows[
                ["model_name", "oof_log_loss", "oof_roc_auc", "median_feature_count"]
            ].to_string(index=False)
        )


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_model_names = list(dict.fromkeys(args.models))
    for model_name in selected_model_names:
        get_model_config(model_name)
    selected_base_model_names = [
        model_name
        for model_name in selected_model_names
        if get_model_config(model_name).role == "base"
    ]
    if not selected_base_model_names:
        raise ValueError("At least one base model must be included in --models.")

    df = load_data(args.data)
    validate_schema(df)

    oof_long_df, fold_metrics_df = run_base_cv(
        df,
        model_names=selected_model_names,
        n_folds=args.folds,
        repeats=args.repeats,
        data_path=args.data,
        workers=args.workers,
        xgb_n_jobs=args.xgb_n_jobs,
        lgbm_n_jobs=args.lgbm_n_jobs,
        elastic_n_jobs=args.elastic_n_jobs,
    )
    oof_mean_df = build_oof_mean_table(oof_long_df)
    base_summary_df = summarize_base_models(
        oof_mean_df=oof_mean_df,
        fold_metrics_df=fold_metrics_df,
        model_names=selected_model_names,
    )
    ensemble_summary_df, ensemble_details = evaluate_ensembles(
        oof_mean_df=oof_mean_df,
        ensemble_model_names=selected_model_names,
        n_folds=args.folds,
        repeats=args.repeats,
    )
    final_recipe = choose_final_recipe(
        base_summary_df=base_summary_df,
        ensemble_summary_df=ensemble_summary_df,
    )

    recommended_trees = {
        row.model_name: (
            None if pd.isna(row.recommended_n_estimators) else int(row.recommended_n_estimators)
        )
        for row in base_summary_df.itertuples(index=False)
        if row.model_family in TREE_MODEL_FAMILIES
    }
    final_recipe["recommended_n_estimators"] = recommended_trees

    experiment_summary = {
        "data_path": str(Path(args.data).resolve()),
        "n_samples": int(len(df)),
        "n_features_total": int(len(df.columns) - 2),
        "n_folds": int(args.folds),
        "n_repeats": int(args.repeats),
        "workers": int(max(1, args.workers)),
        "xgb_n_jobs": args.xgb_n_jobs,
        "lgbm_n_jobs": args.lgbm_n_jobs,
        "elastic_n_jobs": args.elastic_n_jobs,
        "models_evaluated": selected_model_names,
        "label_counts": {
            "no_ati": int((df[TARGET_COLUMN] == 0).sum()),
            "ati": int((df[TARGET_COLUMN] == 1).sum()),
        },
        "final_recipe": final_recipe,
        "base_models_ranked": base_summary_df.to_dict(orient="records"),
        "ensembles_ranked": ensemble_summary_df.to_dict(orient="records"),
    }

    oof_long_path = out_dir / "base_oof_predictions_long.csv"
    oof_mean_path = out_dir / "base_oof_predictions_mean.csv"
    fold_metrics_path = out_dir / "base_fold_metrics.csv"
    base_summary_path = out_dir / "base_model_summary.csv"
    ensemble_summary_path = out_dir / "ensemble_summary.csv"
    final_recipe_path = out_dir / "final_recipe.json"
    experiment_summary_path = out_dir / "experiment_summary.json"
    ensemble_details_path = out_dir / "ensemble_details.json"

    oof_long_df.to_csv(oof_long_path, index=False)
    oof_mean_df.to_csv(oof_mean_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    base_summary_df.to_csv(base_summary_path, index=False)
    ensemble_summary_df.to_csv(ensemble_summary_path, index=False)
    with open(final_recipe_path, "w") as f:
        json.dump(final_recipe, f, indent=2)
    with open(experiment_summary_path, "w") as f:
        json.dump(experiment_summary, f, indent=2)
    with open(ensemble_details_path, "w") as f:
        json.dump(ensemble_details, f, indent=2)

    logging.info(f"Saved: {oof_long_path}")
    logging.info(f"Saved: {oof_mean_path}")
    logging.info(f"Saved: {fold_metrics_path}")
    logging.info(f"Saved: {base_summary_path}")
    logging.info(f"Saved: {ensemble_summary_path}")
    logging.info(f"Saved: {final_recipe_path}")
    logging.info(f"Saved: {experiment_summary_path}")
    logging.info(f"Saved: {ensemble_details_path}")

    print_console_summary(base_summary_df, ensemble_summary_df, final_recipe)


if __name__ == "__main__":
    main()
