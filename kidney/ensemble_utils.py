#!/usr/bin/env python3
"""
ensemble_utils.py — Shared training/evaluation helpers for the ATI ensemble
===========================================================================
Centralizes fold fitting, ensemble evaluation, and final artifact assembly so
`evaluate.py`, `train.py`, and `predict.py` stay aligned.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

from model import (
    CV_FOLDS,
    EARLY_STOPPING_ROUNDS,
    PREDICTION_THRESHOLD,
    RANDOM_SEED,
    build_calibrated_meta_learner,
    build_estimator,
    build_ridge_meta_learner,
    get_lightgbm_module,
    get_model_config,
    get_seed_list,
    get_splitter,
)
from preprocess import build_matrix, select_feature_columns

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Return the core binary classification metrics used throughout."""
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    metrics = {
        "log_loss": float(log_loss(y_true, y_prob)),
        "positive_rate_at_0_5": float((y_prob >= PREDICTION_THRESHOLD).mean()),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _median_or_none(values: list[int | None]) -> int | None:
    """Return the integer median of optional values when available."""
    valid_values = [int(value) for value in values if value is not None]
    if not valid_values:
        return None
    return int(np.median(valid_values))


def fit_model_on_fold(
    *,
    model_name: str,
    df_train,
    y_train: np.ndarray,
    df_val,
    y_val: np.ndarray,
    seed: int,
    feature_cache: dict | None = None,
    xgb_n_jobs: int | None = None,
    lgbm_n_jobs: int | None = None,
    elastic_n_jobs: int | None = None,
):
    """Fit one configured model group on one fold and return validation predictions."""
    config = get_model_config(model_name)
    cache_key = (config.feature_set, seed)
    if feature_cache is not None and cache_key in feature_cache:
        feature_cols, selection_meta = feature_cache[cache_key]
    else:
        feature_cols, selection_meta = select_feature_columns(
            df_train,
            y_train,
            config.feature_set,
            random_state=seed,
        )
        if feature_cache is not None:
            feature_cache[cache_key] = (feature_cols, selection_meta)

    X_train = build_matrix(df_train, feature_cols)
    X_val = build_matrix(df_val, feature_cols)
    model_seed_list = get_seed_list(repeats=config.n_seeds, base_seed=seed)
    val_prob_parts = []
    best_iterations = []
    trees_used_values = []

    for model_seed in model_seed_list:
        estimator = build_estimator(
            model_name,
            model_seed,
            early_stopping=(config.family == "xgb"),
            xgb_n_jobs=xgb_n_jobs,
            lgbm_n_jobs=lgbm_n_jobs,
            elastic_n_jobs=elastic_n_jobs,
        )

        if config.family == "xgb":
            estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_iteration = getattr(estimator, "best_iteration", None)
            trees_used = (
                int(estimator.get_params()["n_estimators"])
                if best_iteration is None
                else int(best_iteration) + 1
            )
        elif config.family == "lgbm":
            lgb = get_lightgbm_module()
            estimator.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="binary_logloss",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            best_iteration = getattr(estimator, "best_iteration_", None)
            trees_used = getattr(estimator, "n_estimators_", None)
            if trees_used is None:
                trees_used = (
                    int(estimator.get_params()["n_estimators"])
                    if best_iteration is None
                    else int(best_iteration)
                )
        else:
            estimator.fit(X_train, y_train)
            best_iteration = None
            trees_used = None

        val_prob_parts.append(estimator.predict_proba(X_val)[:, 1])
        best_iterations.append(best_iteration)
        trees_used_values.append(trees_used)

    val_prob = np.mean(np.vstack(val_prob_parts), axis=0)
    metrics = compute_metrics(y_val, val_prob)
    return {
        "feature_cols": feature_cols,
        "selection_meta": selection_meta,
        "val_prob": val_prob,
        "metrics": metrics,
        "best_iteration": _median_or_none(best_iterations),
        "trees_used": _median_or_none(trees_used_values),
        "seed_count": int(len(model_seed_list)),
    }


def fit_full_model_group(
    *,
    model_name: str,
    df_train,
    y_train: np.ndarray,
    recommended_n_estimators: int | None = None,
    xgb_n_jobs: int | None = None,
    lgbm_n_jobs: int | None = None,
    elastic_n_jobs: int | None = None,
):
    """Fit the full-data seed-averaged version of a base model."""
    config = get_model_config(model_name)
    selected_cols, selection_meta = select_feature_columns(
        df_train,
        y_train,
        config.feature_set,
        random_state=RANDOM_SEED,
    )
    X = build_matrix(df_train, selected_cols)
    estimators = []

    for seed in get_seed_list(repeats=config.n_seeds):
        estimator = build_estimator(
            model_name,
            seed,
            n_estimators=recommended_n_estimators,
            early_stopping=False,
            xgb_n_jobs=xgb_n_jobs,
            lgbm_n_jobs=lgbm_n_jobs,
            elastic_n_jobs=elastic_n_jobs,
        )
        estimator.fit(X, y_train)
        estimators.append(estimator)

    return {
        "name": model_name,
        "family": config.family,
        "feature_set": config.feature_set,
        "feature_cols": selected_cols,
        "selection_meta": selection_meta,
        "estimators": estimators,
        "recommended_n_estimators": recommended_n_estimators,
    }


def predict_model_group(model_group: dict, df_input) -> np.ndarray:
    """Predict ATI probabilities from a fitted seed-averaged model group."""
    X = build_matrix(df_input, model_group["feature_cols"])
    probs = [estimator.predict_proba(X)[:, 1] for estimator in model_group["estimators"]]
    return np.mean(np.vstack(probs), axis=0)


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values)
    exp_values = np.exp(centered)
    return exp_values / exp_values.sum()


def optimize_blend_weights(X_meta: np.ndarray, y_true: np.ndarray):
    """Optimize convex blend weights on a meta-feature matrix."""
    if X_meta.shape[1] == 1:
        weights = np.array([1.0], dtype=float)
        return weights, {"success": True, "objective": float(log_loss(y_true, X_meta[:, 0]))}

    def objective(theta: np.ndarray) -> float:
        weights = _softmax(theta)
        prob = np.clip(X_meta @ weights, 1e-6, 1 - 1e-6)
        return float(log_loss(y_true, prob))

    result = minimize(
        objective,
        x0=np.zeros(X_meta.shape[1], dtype=float),
        method="L-BFGS-B",
    )
    weights = _softmax(result.x)
    return weights, {
        "success": bool(result.success),
        "message": result.message,
        "objective": float(result.fun),
    }


def crossfit_blend_predictions(
    X_meta: np.ndarray,
    y_true: np.ndarray,
    *,
    seed_list: list[int],
    n_folds: int = CV_FOLDS,
):
    """Evaluate a convex blend with a second-stage cross-fitting loop."""
    repeated_preds = np.full((len(y_true), len(seed_list)), np.nan, dtype=float)
    weight_rows = []

    for repeat_idx, seed in enumerate(seed_list):
        splitter = get_splitter(seed, n_splits=n_folds)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_meta, y_true), start=1):
            weights, info = optimize_blend_weights(X_meta[train_idx], y_true[train_idx])
            repeated_preds[val_idx, repeat_idx] = np.clip(
                X_meta[val_idx] @ weights,
                1e-6,
                1 - 1e-6,
            )
            weight_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold_idx),
                    "weights": [float(x) for x in weights],
                    "optimizer_success": bool(info["success"]),
                    "objective": float(info["objective"]),
                }
            )

    oof_prob = np.nanmean(repeated_preds, axis=1)
    return oof_prob, weight_rows


def crossfit_meta_predictions(
    X_meta: np.ndarray,
    y_true: np.ndarray,
    *,
    seed_list: list[int],
    calibrated: bool,
    n_folds: int = CV_FOLDS,
):
    """Cross-fit a ridge stacker, optionally with sigmoid calibration."""
    repeated_preds = np.full((len(y_true), len(seed_list)), np.nan, dtype=float)

    for repeat_idx, seed in enumerate(seed_list):
        splitter = get_splitter(seed, n_splits=n_folds)
        for train_idx, val_idx in splitter.split(X_meta, y_true):
            estimator = (
                build_calibrated_meta_learner(seed)
                if calibrated
                else build_ridge_meta_learner(seed)
            )
            estimator.fit(X_meta[train_idx], y_true[train_idx])
            repeated_preds[val_idx, repeat_idx] = estimator.predict_proba(X_meta[val_idx])[:, 1]

    return np.nanmean(repeated_preds, axis=1)


def fit_final_combiner(
    *,
    recipe_type: str,
    base_model_names: list[str],
    X_meta: np.ndarray,
    y_true: np.ndarray,
):
    """Fit the final saved ensemble combiner from OOF meta-features."""
    if recipe_type == "single_model":
        if X_meta.shape[1] != 1:
            raise ValueError("single_model combiner expects exactly one meta feature column.")
        return {
            "type": "single_model",
            "base_model_names": base_model_names,
            "training_metrics": compute_metrics(y_true, X_meta[:, 0]),
        }

    if recipe_type == "blend":
        weights, info = optimize_blend_weights(X_meta, y_true)
        prob = np.clip(X_meta @ weights, 1e-6, 1 - 1e-6)
        return {
            "type": "blend",
            "base_model_names": base_model_names,
            "weights": [float(x) for x in weights],
            "optimizer": info,
            "training_metrics": compute_metrics(y_true, prob),
        }

    if recipe_type == "stack_ridge":
        estimator = build_ridge_meta_learner(RANDOM_SEED)
        estimator.fit(X_meta, y_true)
        prob = estimator.predict_proba(X_meta)[:, 1]
        return {
            "type": "stack_ridge",
            "base_model_names": base_model_names,
            "estimator": estimator,
            "training_metrics": compute_metrics(y_true, prob),
        }

    if recipe_type == "stack_calibrated":
        estimator = build_calibrated_meta_learner(RANDOM_SEED)
        estimator.fit(X_meta, y_true)
        prob = estimator.predict_proba(X_meta)[:, 1]
        return {
            "type": "stack_calibrated",
            "base_model_names": base_model_names,
            "estimator": estimator,
            "training_metrics": compute_metrics(y_true, prob),
        }

    raise ValueError(f"Unsupported recipe_type: {recipe_type}")


def combine_from_base_matrix(base_matrix: np.ndarray, combiner: dict) -> np.ndarray:
    """Apply a saved combiner to base-model probabilities."""
    if combiner["type"] == "single_model":
        return np.clip(base_matrix[:, 0], 1e-6, 1 - 1e-6)
    if combiner["type"] == "blend":
        weights = np.asarray(combiner["weights"], dtype=float)
        return np.clip(base_matrix @ weights, 1e-6, 1 - 1e-6)
    if combiner["type"] in {"stack_ridge", "stack_calibrated"}:
        return combiner["estimator"].predict_proba(base_matrix)[:, 1]
    raise ValueError(f"Unsupported combiner type: {combiner['type']}")
