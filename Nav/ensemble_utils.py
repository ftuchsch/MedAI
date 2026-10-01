#!/usr/bin/env python3
"""
ensemble_utils.py — Shared fitting and blending helpers for Nav experiments
==========================================================================
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from model import (
    CV_FOLDS,
    DEFAULT_XGB_MODEL_SEEDS,
    RANDOM_SEED,
    build_model,
    get_seed_list,
)
from preprocess import build_feature_frame, select_feature_columns

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    metrics = {
        "log_loss": float(log_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _median_or_none(values: list[int | None]) -> int | None:
    valid_values = [int(value) for value in values if value is not None]
    if not valid_values:
        return None
    return int(np.median(valid_values))


def fit_model_on_fold(
    *,
    model_name: str,
    spec: dict,
    df_train,
    y_train: np.ndarray,
    df_val,
    y_val: np.ndarray,
    seed: int,
    xgb_model_seeds: int = DEFAULT_XGB_MODEL_SEEDS,
    xgb_n_jobs: int | None = None,
):
    feature_cols = select_feature_columns(
        df_train,
        y_train,
        protein_top_k=spec["protein_top_k"],
        feature_selector=spec.get("feature_selector", "anova"),
        include_clinical=spec.get("include_clinical", True),
    )
    X_train = build_feature_frame(df_train, feature_cols)
    X_val = build_feature_frame(df_val, feature_cols)

    seed_count = int(xgb_model_seeds) if spec["artifact_type"] == "xgboost" else 1
    model_seed_list = get_seed_list(repeats=seed_count, base_seed=seed)

    val_prob_parts = []
    best_iterations = []
    trees_used_values = []

    for model_seed in model_seed_list:
        estimator = build_model(
            spec["builder"],
            random_state=model_seed,
            n_estimators=None,
            early_stopping=(spec["artifact_type"] == "xgboost"),
            n_jobs=xgb_n_jobs,
        )

        if spec["artifact_type"] == "xgboost":
            estimator.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_iteration = getattr(estimator, "best_iteration", None)
            trees_used = (
                int(estimator.get_params()["n_estimators"])
                if best_iteration is None
                else int(best_iteration) + 1
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
        "val_prob": val_prob,
        "metrics": metrics,
        "best_iteration": _median_or_none(best_iterations),
        "trees_used": _median_or_none(trees_used_values),
        "seed_count": len(model_seed_list),
    }


def fit_full_model_group(
    *,
    model_name: str,
    spec: dict,
    df_train,
    y_train: np.ndarray,
    base_seed: int = RANDOM_SEED,
    xgb_model_seeds: int = DEFAULT_XGB_MODEL_SEEDS,
    recommended_n_estimators: int | None = None,
    xgb_n_jobs: int | None = None,
):
    feature_cols = select_feature_columns(
        df_train,
        y_train,
        protein_top_k=spec["protein_top_k"],
        feature_selector=spec.get("feature_selector", "anova"),
        include_clinical=spec.get("include_clinical", True),
    )
    X_train = build_feature_frame(df_train, feature_cols)

    seed_count = int(xgb_model_seeds) if spec["artifact_type"] == "xgboost" else 1
    model_seed_list = get_seed_list(repeats=seed_count, base_seed=base_seed)
    estimators = []

    for model_seed in model_seed_list:
        estimator = build_model(
            spec["builder"],
            random_state=model_seed,
            n_estimators=recommended_n_estimators,
            early_stopping=False,
            n_jobs=xgb_n_jobs,
        )
        estimator.fit(X_train, y_train)
        estimators.append(estimator)

    return {
        "name": model_name,
        "spec": spec,
        "feature_cols": feature_cols,
        "estimators": estimators,
        "recommended_n_estimators": recommended_n_estimators,
        "seed_count": seed_count,
    }


def predict_model_group(model_group: dict, df_input) -> np.ndarray:
    X_input = build_feature_frame(df_input, model_group["feature_cols"])
    probs = [estimator.predict_proba(X_input)[:, 1] for estimator in model_group["estimators"]]
    return np.mean(np.vstack(probs), axis=0)


def _softmax(values: np.ndarray) -> np.ndarray:
    centered = values - np.max(values)
    exp_values = np.exp(centered)
    return exp_values / exp_values.sum()


def optimize_blend_weights(X_meta: np.ndarray, y_true: np.ndarray):
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
    repeated_preds = np.full((len(y_true), len(seed_list)), np.nan, dtype=float)
    weight_rows = []

    for repeat_idx, seed in enumerate(seed_list):
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_meta, y_true), start=1):
            weights, info = optimize_blend_weights(X_meta[train_idx], y_true[train_idx])
            repeated_preds[val_idx, repeat_idx] = np.clip(X_meta[val_idx] @ weights, 1e-6, 1 - 1e-6)
            weight_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold_idx),
                    "weights": [float(value) for value in weights],
                    "optimizer_success": bool(info["success"]),
                    "objective": float(info["objective"]),
                }
            )

    oof_prob = np.nanmean(repeated_preds, axis=1)
    return oof_prob, weight_rows
