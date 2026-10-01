#!/usr/bin/env python3
"""
train.py — Train the final Nav model bundle from a CV-derived recipe
====================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ensemble_utils import compute_metrics, fit_full_model_group, predict_model_group
from model import (
    BASE_MODEL_SPECS,
    DEFAULT_XGB_MODEL_SEEDS,
    ENSEMBLE_COMPONENTS,
    RANDOM_SEED,
    SELECTED_LDA_TOP_K,
    SELECTED_PLS_PARAMS,
    SELECTED_PLS_TOP_K,
    SELECTED_RIDGE_TOP_K,
    SELECTED_XGBOOST_TOP_K,
)
from preprocess import get_target, load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SCRIPT_DIR.parent / "data" / "train.csv"
_DEFAULT_WEIGHTS = _SCRIPT_DIR / "weights"
_DEFAULT_RECIPE = _SCRIPT_DIR / "results" / "final_recipe.json"


def parse_args():
    p = argparse.ArgumentParser(description="Train final ATI models and save the Nav bundle.")
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
        "--recipe",
        default=str(_DEFAULT_RECIPE),
        help="Path to evaluate.py final_recipe.json (default: ./results/final_recipe.json)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Base random seed for model training (default: {RANDOM_SEED})",
    )
    p.add_argument(
        "--xgb-model-seeds",
        type=int,
        default=DEFAULT_XGB_MODEL_SEEDS,
        help=(
            "Number of differently-seeded XGBoost fits to average in the saved bundle "
            f"(default: {DEFAULT_XGB_MODEL_SEEDS})"
        ),
    )
    p.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=None,
        help="Override inner XGBoost threads per fit (default: model config).",
    )
    p.add_argument(
        "--xgb-top-k",
        type=int,
        default=SELECTED_XGBOOST_TOP_K,
        help=f"Number of protein features to keep for Selected XGBoost (default: {SELECTED_XGBOOST_TOP_K})",
    )
    p.add_argument(
        "--ridge-top-k",
        type=int,
        default=SELECTED_RIDGE_TOP_K,
        help=f"Number of protein features to keep for Selected Ridge LR (default: {SELECTED_RIDGE_TOP_K})",
    )
    p.add_argument(
        "--lda-top-k",
        type=int,
        default=SELECTED_LDA_TOP_K,
        help=f"Number of protein features to keep for Selected Shrinkage LDA (default: {SELECTED_LDA_TOP_K})",
    )
    p.add_argument(
        "--pls-top-k",
        type=int,
        default=SELECTED_PLS_TOP_K,
        help=f"Number of protein features to keep for Selected PLS LR (default: {SELECTED_PLS_TOP_K})",
    )
    return p.parse_args()


def save_json(path: Path, payload):
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_final_recipe(path: Path) -> dict:
    if path.exists():
        logging.info("Loading final recipe: %s", path)
        with path.open() as f:
            return json.load(f)

    logging.warning("Recipe %s not found. Falling back to legacy fixed ensemble bundle.", path)
    return {
        "recipe_type": "blend",
        "candidate_name": "legacy_fixed_blend",
        "base_model_names": [component["name"] for component in ENSEMBLE_COMPONENTS],
        "weights": [float(component["weight"]) for component in ENSEMBLE_COMPONENTS],
        "recommended_n_estimators": {},
    }


def resolve_model_spec(model_name: str, args) -> dict:
    if model_name not in BASE_MODEL_SPECS:
        raise KeyError(f"Unknown model '{model_name}'. Choose from: {sorted(BASE_MODEL_SPECS)}")
    spec = dict(BASE_MODEL_SPECS[model_name])
    top_k_overrides = {
        "Selected XGBoost": args.xgb_top_k,
        "Selected Ridge LR": args.ridge_top_k,
        "Selected Shrinkage LDA": args.lda_top_k,
        "Selected PLS LR": args.pls_top_k,
    }
    if model_name in top_k_overrides:
        spec["protein_top_k"] = top_k_overrides[model_name]
    return spec


def choose_models_to_train(recipe: dict) -> list[str]:
    model_names = set(recipe.get("base_model_names", []))
    model_names.add("Selected XGBoost")
    return [model_name for model_name in BASE_MODEL_SPECS if model_name in model_names]


def save_model_group_artifacts(
    *,
    out_dir: Path,
    model_name: str,
    model_group: dict,
):
    spec = model_group["spec"]
    feature_path = out_dir / spec["feature_file"]
    save_json(feature_path, model_group["feature_cols"])
    logging.info("Saved features: %s", feature_path)

    if spec["artifact_type"] == "xgboost":
        primary_path = out_dir / spec["model_file"]
        model_group["estimators"][0].save_model(str(primary_path))
        model_files = [primary_path.name]
        logging.info("Saved model: %s", primary_path)

        stem = Path(spec["model_file"]).stem
        suffix = Path(spec["model_file"]).suffix
        for idx, estimator in enumerate(model_group["estimators"][1:], start=2):
            extra_name = f"{stem}_seed{idx}{suffix}"
            extra_path = out_dir / extra_name
            estimator.save_model(str(extra_path))
            model_files.append(extra_path.name)
            logging.info("Saved model: %s", extra_path)

        if model_name == "Selected XGBoost":
            feature_importances = [
                estimator.feature_importances_
                for estimator in model_group["estimators"]
                if hasattr(estimator, "feature_importances_")
            ]
            if feature_importances:
                importance = pd.DataFrame(
                    {
                        "feature": model_group["feature_cols"],
                        "gain": np.mean(np.vstack(feature_importances), axis=0),
                    }
                ).sort_values("gain", ascending=False)
                importance_path = out_dir / "feature_importance.csv"
                importance.to_csv(importance_path, index=False)
                logging.info("Saved feature importance: %s", importance_path)

        return {
            "name": model_name,
            "artifact_type": spec["artifact_type"],
            "feature_file": spec["feature_file"],
            "model_file": spec["model_file"],
            "model_files": model_files,
        }

    model_path = out_dir / spec["model_file"]
    joblib.dump(model_group["estimators"][0], model_path)
    logging.info("Saved model: %s", model_path)
    return {
        "name": model_name,
        "artifact_type": spec["artifact_type"],
        "feature_file": spec["feature_file"],
        "model_file": spec["model_file"],
    }


def build_ensemble_config(recipe: dict, artifact_registry: dict[str, dict]) -> dict:
    if recipe["recipe_type"] == "single_model":
        component_names = recipe["base_model_names"]
        weights = [1.0]
    else:
        component_names = recipe["base_model_names"]
        weights = [float(value) for value in recipe["weights"]]

    components = []
    for model_name, weight in zip(component_names, weights):
        artifact = dict(artifact_registry[model_name])
        artifact["weight"] = float(weight)
        components.append(artifact)

    return {
        "prediction_mode": "weighted_average",
        "recipe_type": recipe["recipe_type"],
        "candidate_name": recipe["candidate_name"],
        "components": components,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = Path(args.recipe)
    final_recipe = load_final_recipe(recipe_path)
    models_to_train = choose_models_to_train(final_recipe)
    recommended_tree_map = final_recipe.get("recommended_n_estimators", {})

    df = load_data(args.data)
    y = get_target(df)

    trained_groups = {}
    artifact_registry = {}
    train_probabilities = {}
    training_rows = []

    for model_name in models_to_train:
        spec = resolve_model_spec(model_name, args)
        recommended_trees = recommended_tree_map.get(model_name)
        if recommended_trees is not None:
            recommended_trees = int(recommended_trees)
        logging.info(
            "Training %s on %s samples (top_k=%s, recommended_n_estimators=%s)",
            model_name,
            len(y),
            spec["protein_top_k"],
            recommended_trees,
        )
        model_group = fit_full_model_group(
            model_name=model_name,
            spec=spec,
            df_train=df,
            y_train=y,
            base_seed=args.seed,
            xgb_model_seeds=args.xgb_model_seeds,
            recommended_n_estimators=recommended_trees,
            xgb_n_jobs=args.xgb_n_jobs,
        )
        trained_groups[model_name] = model_group
        artifact_registry[model_name] = save_model_group_artifacts(
            out_dir=out_dir,
            model_name=model_name,
            model_group=model_group,
        )

        train_prob = predict_model_group(model_group, df)
        train_probabilities[model_name] = train_prob
        metrics = compute_metrics(y, train_prob)
        training_rows.append(
            {
                "model": model_name,
                "train_auc": float(metrics["roc_auc"]),
                "train_log_loss": float(metrics["log_loss"]),
                "feature_count": int(len(model_group["feature_cols"])),
                "seed_count": int(model_group["seed_count"]),
                "recommended_n_estimators": recommended_trees,
            }
        )

    if final_recipe["recipe_type"] == "single_model":
        final_prob = train_probabilities[final_recipe["base_model_names"][0]]
    else:
        final_prob = np.zeros(len(y), dtype=float)
        for model_name, weight in zip(final_recipe["base_model_names"], final_recipe["weights"]):
            final_prob += float(weight) * train_probabilities[model_name]

    final_metrics = compute_metrics(y, final_prob)
    training_rows.append(
        {
            "model": final_recipe["candidate_name"],
            "train_auc": float(final_metrics["roc_auc"]),
            "train_log_loss": float(final_metrics["log_loss"]),
            "feature_count": None,
            "seed_count": None,
            "recommended_n_estimators": None,
        }
    )

    print(f"\n{'=' * 60}")
    print(f"Samples   : {len(y)} (ATI={int(y.sum())}, No ATI={int((y == 0).sum())})")
    print(f"Recipe    : {final_recipe['candidate_name']}")
    print(f"XGB seeds : {args.xgb_model_seeds}")
    print(f"{'=' * 60}")
    for row in training_rows:
        print(
            f"{row['model']:<32} Train AUC: {row['train_auc']:.3f} | "
            f"Train LogLoss: {row['train_log_loss']:.3f}"
        )

    ensemble_config = build_ensemble_config(final_recipe, artifact_registry)
    save_json(out_dir / "ensemble_config.json", ensemble_config)

    training_summary = {
        "data_path": str(Path(args.data).resolve()),
        "recipe_path": str(recipe_path.resolve()) if recipe_path.exists() else str(recipe_path),
        "seed": int(args.seed),
        "xgb_model_seeds": int(args.xgb_model_seeds),
        "xgb_n_jobs": args.xgb_n_jobs,
        "xgb_top_k": int(args.xgb_top_k),
        "ridge_top_k": int(args.ridge_top_k),
        "lda_top_k": int(args.lda_top_k),
        "pls_top_k": int(args.pls_top_k),
        "pls_n_components": SELECTED_PLS_PARAMS["n_components"],
        "final_recipe": final_recipe,
        "artifacts": artifact_registry,
        "training_metrics": training_rows,
    }
    save_json(out_dir / "training_summary.json", training_summary)

    logging.info("Saved bundle config: %s", out_dir / "ensemble_config.json")
    logging.info("Saved training summary: %s", out_dir / "training_summary.json")

    print(f"\nWeights saved to {out_dir}/")
    print(
        "Compatibility note: starter-code inference can still use xgboost_model.json + feature_cols.json."
    )


if __name__ == "__main__":
    main()
