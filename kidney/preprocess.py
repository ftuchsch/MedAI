#!/usr/bin/env python3
"""
preprocess.py — Data loading, schema validation, and feature selection
======================================================================
All fold-dependent transformations are computed from training data only.
This module deliberately keeps the mechanics deterministic so evaluation,
final training, and inference use the same feature semantics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model import (
    CLINICAL_FEATURES,
    ID_COLUMN,
    PROTEIN_PREFIX,
    RANDOM_SEED,
    TARGET_COLUMN,
    get_feature_set_config,
)

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load a CSV and log its shape."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False, na_values=[".", ""])
    logger.info(f"  {len(df)} samples, {len(df.columns)} columns")
    return df


def get_protein_columns(df: pd.DataFrame) -> list[str]:
    """Return lexicographically sorted SomaScan protein feature columns."""
    protein_cols = sorted(c for c in df.columns if c.startswith(PROTEIN_PREFIX))
    if not protein_cols:
        raise ValueError("No feature_ columns found in the input data.")
    return protein_cols


def validate_schema(df: pd.DataFrame):
    """Fail fast when the required training columns are missing."""
    required = [ID_COLUMN, TARGET_COLUMN, *CLINICAL_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    return True


def resolve_clinical_columns(
    *,
    keep_clinical: bool = True,
    drop_egfr: bool = False,
    only_egfr: bool = False,
) -> list[str]:
    """Return the clinical covariates implied by a feature-set config."""
    if only_egfr:
        return ["baseline_egfr_23"]
    if not keep_clinical:
        return []
    if drop_egfr:
        return [c for c in CLINICAL_FEATURES if c != "baseline_egfr_23"]
    return list(CLINICAL_FEATURES)


def build_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Return a dense numeric matrix aligned to the provided feature order."""
    return df.reindex(columns=feature_cols).to_numpy(dtype=float)


def build_features_and_labels(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    return_sample_ids: bool = False,
):
    """Build X and y arrays without dropping rows."""
    validate_schema(df)
    if feature_cols is None:
        feature_cols = get_protein_columns(df) + list(CLINICAL_FEATURES)

    sample_ids = df[ID_COLUMN].astype(str).reset_index(drop=True)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    X = build_matrix(df, feature_cols)

    logger.info(f"Samples: {len(df)} | No ATI: {(y == 0).sum()} | ATI: {(y == 1).sum()}")
    logger.info(
        f"Features: {sum(c.startswith(PROTEIN_PREFIX) for c in feature_cols)} protein"
        f" + {sum(not c.startswith(PROTEIN_PREFIX) for c in feature_cols)} clinical"
        f" = {len(feature_cols)} total"
    )

    if return_sample_ids:
        return X, y, feature_cols, sample_ids
    return X, y, feature_cols


def rank_proteins(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    method: str = "f_classif",
    random_state: int = RANDOM_SEED,
) -> tuple[list[str], np.ndarray]:
    """Rank protein columns by univariate signal within a training fold."""
    protein_cols = get_protein_columns(df_train)
    X = build_matrix(df_train, protein_cols)

    if method == "f_classif":
        scores, _ = f_classif(X, y_train)
    elif method == "mutual_info":
        scores = mutual_info_classif(X, y_train, random_state=random_state)
    elif method == "auc":
        scores = np.zeros(len(protein_cols), dtype=float)
        for idx, col in enumerate(protein_cols):
            values = df_train[col].to_numpy(dtype=float)
            if np.nanstd(values) == 0:
                scores[idx] = 0.0
                continue
            auc = roc_auc_score(y_train, values)
            scores[idx] = abs(auc - 0.5)
    else:
        raise ValueError(f"Unknown ranking method: {method}")

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(scores)[::-1]
    ranked = [protein_cols[idx] for idx in order]
    return ranked, scores[order]


def _select_top_k_proteins(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    top_k: int,
    ranking_method: str,
    random_state: int,
) -> tuple[list[str], dict]:
    ranked, ordered_scores = rank_proteins(
        df_train,
        y_train,
        method=ranking_method,
        random_state=random_state,
    )
    selected = ranked[:top_k]
    metadata = {
        "strategy": "top_k",
        "ranking_method": ranking_method,
        "top_k": int(top_k),
        "selected_protein_count": int(len(selected)),
        "top_proteins": selected[:25],
        "top_scores": [float(x) for x in ordered_scores[:25]],
    }
    return selected, metadata


def _stability_select_proteins(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    ranking_method: str,
    prefilter_k: int,
    subsamples: int,
    sample_fraction: float,
    threshold: float,
    min_features: int,
    max_features: int,
    random_state: int,
) -> tuple[list[str], dict]:
    prefiltered, prefilter_meta = _select_top_k_proteins(
        df_train,
        y_train,
        top_k=prefilter_k,
        ranking_method=ranking_method,
        random_state=random_state,
    )
    X_prefilter = build_matrix(df_train, prefiltered)
    rng = np.random.default_rng(random_state)
    selection_counts = np.zeros(len(prefiltered), dtype=float)

    for rep in range(subsamples):
        sample_size = max(24, int(round(len(y_train) * sample_fraction)))
        row_idx = np.sort(rng.choice(len(y_train), size=sample_size, replace=False))
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=0.3,
                        l1_ratio=0.5,
                        max_iter=4000,
                        random_state=random_state + rep,
                    ),
                ),
            ]
        )
        pipeline.fit(X_prefilter[row_idx], y_train[row_idx])
        coef = np.abs(pipeline.named_steps["clf"].coef_.ravel())
        selection_counts += coef > 1e-8

    frequency = selection_counts / subsamples
    selected_idx = np.where(frequency >= threshold)[0]

    if len(selected_idx) < min_features:
        selected_idx = np.argsort(frequency)[::-1][:min_features]
    elif len(selected_idx) > max_features:
        selected_idx = selected_idx[np.argsort(frequency[selected_idx])[::-1][:max_features]]

    selected = [prefiltered[idx] for idx in selected_idx]
    ranked_selected = sorted(
        selected,
        key=lambda col: (-frequency[prefiltered.index(col)], col),
    )
    metadata = {
        "strategy": "stability",
        "ranking_method": ranking_method,
        "prefilter_k": int(prefilter_k),
        "subsamples": int(subsamples),
        "sample_fraction": float(sample_fraction),
        "threshold": float(threshold),
        "selected_protein_count": int(len(ranked_selected)),
        "top_stable_proteins": ranked_selected[:25],
        "top_stability_frequency": [
            float(frequency[prefiltered.index(col)]) for col in ranked_selected[:25]
        ],
        "prefilter_meta": prefilter_meta,
    }
    return ranked_selected, metadata


def select_feature_columns(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_set_name: str,
    *,
    random_state: int = RANDOM_SEED,
) -> tuple[list[str], dict]:
    """Return the fold-specific feature list for a configured feature set."""
    cfg = get_feature_set_config(feature_set_name)
    clinical_cols = resolve_clinical_columns(
        keep_clinical=cfg.keep_clinical,
        drop_egfr=cfg.drop_egfr,
        only_egfr=cfg.only_egfr,
    )

    for col in clinical_cols:
        if col not in df_train.columns:
            raise ValueError(f"Clinical column '{col}' is missing from training data.")

    if cfg.strategy == "clinical_only":
        feature_cols = clinical_cols
        metadata = {
            "strategy": cfg.strategy,
            "selected_protein_count": 0,
            "selected_clinical_columns": clinical_cols,
        }
    elif cfg.strategy == "all_features":
        feature_cols = get_protein_columns(df_train) + clinical_cols
        metadata = {
            "strategy": cfg.strategy,
            "selected_protein_count": int(len(feature_cols) - len(clinical_cols)),
            "selected_clinical_columns": clinical_cols,
        }
    elif cfg.strategy == "top_k":
        proteins, metadata = _select_top_k_proteins(
            df_train,
            y_train,
            top_k=int(cfg.top_k),
            ranking_method=cfg.ranking_method,
            random_state=random_state,
        )
        feature_cols = proteins + clinical_cols
        metadata["selected_clinical_columns"] = clinical_cols
    elif cfg.strategy == "stability":
        proteins, metadata = _stability_select_proteins(
            df_train,
            y_train,
            ranking_method=cfg.ranking_method,
            prefilter_k=int(cfg.stability_prefilter_k),
            subsamples=int(cfg.stability_subsamples),
            sample_fraction=float(cfg.stability_fraction),
            threshold=float(cfg.stability_threshold),
            min_features=int(cfg.stability_min_features),
            max_features=int(cfg.stability_max_features),
            random_state=random_state,
        )
        feature_cols = proteins + clinical_cols
        metadata["selected_clinical_columns"] = clinical_cols
    else:
        raise ValueError(f"Unsupported feature-set strategy: {cfg.strategy}")

    metadata.update(
        {
            "feature_set_name": cfg.name,
            "selected_feature_count": int(len(feature_cols)),
            "selected_feature_preview": feature_cols[:25],
        }
    )
    return feature_cols, metadata
