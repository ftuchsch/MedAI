#!/usr/bin/env python3
"""
preprocess.py — Data loading and fold-safe feature selection helpers
====================================================================
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedShuffleSplit

from model import (
    CLINICAL_FEATURES,
    FEATURE_SELECTOR_ANOVA,
    FEATURE_SELECTOR_STABLE_ANOVA,
    STABILITY_SELECTION_INNER_SEEDS,
)

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False, na_values=[".", ""])
    logger.info(f"  {len(df)} samples, {len(df.columns)} columns")
    return df


def get_target(df: pd.DataFrame) -> np.ndarray:
    if "ati" not in df.columns:
        raise KeyError("Expected target column 'ati' to be present.")
    y = df["ati"].astype(int).to_numpy()
    logger.info(f"Samples: {len(y)} | No ATI: {(y == 0).sum()} | ATI: {(y == 1).sum()}")
    return y


def get_protein_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("feature_")])


def get_clinical_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in CLINICAL_FEATURES if c in df.columns]


def _anova_scores(df: pd.DataFrame, protein_cols: list[str], y: np.ndarray) -> np.ndarray:
    scores, _ = f_classif(df[protein_cols], y)
    return np.nan_to_num(scores, nan=-np.inf)


def _stable_anova_top_idx(
    df: pd.DataFrame,
    y: np.ndarray,
    protein_cols: list[str],
    protein_top_k: int,
) -> np.ndarray:
    selection_counts = np.zeros(len(protein_cols), dtype=int)

    for seed in STABILITY_SELECTION_INNER_SEEDS:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
        inner_idx, _ = next(splitter.split(df, y))
        inner_scores = _anova_scores(df.iloc[inner_idx], protein_cols, y[inner_idx])
        top_idx = np.argsort(inner_scores)[-protein_top_k:]
        selection_counts[top_idx] += 1

    full_scores = _anova_scores(df, protein_cols, y)
    ranking = np.lexsort((full_scores, selection_counts))
    chosen_idx = np.sort(ranking[-protein_top_k:])
    return chosen_idx


def select_feature_columns(
    df: pd.DataFrame,
    y: np.ndarray,
    protein_top_k: int | None,
    feature_selector: str = FEATURE_SELECTOR_ANOVA,
    include_clinical: bool = True,
) -> list[str]:
    protein_cols = get_protein_columns(df)
    clinical_cols = get_clinical_columns(df)

    if protein_top_k is None or protein_top_k >= len(protein_cols):
        selected_proteins = protein_cols
    elif protein_top_k <= 0:
        selected_proteins = []
    else:
        if feature_selector == FEATURE_SELECTOR_ANOVA:
            scores = _anova_scores(df, protein_cols, y)
            top_idx = np.sort(np.argsort(scores)[-protein_top_k:])
        elif feature_selector == FEATURE_SELECTOR_STABLE_ANOVA:
            top_idx = _stable_anova_top_idx(df, y, protein_cols, protein_top_k)
        else:
            raise ValueError(
                f"Unknown feature selector '{feature_selector}'. "
                f"Expected one of: {FEATURE_SELECTOR_ANOVA}, {FEATURE_SELECTOR_STABLE_ANOVA}"
            )
        selected_proteins = [protein_cols[i] for i in top_idx]

    feature_cols = selected_proteins + (clinical_cols if include_clinical else [])
    if not feature_cols:
        raise ValueError("No feature columns were selected.")
    return feature_cols


def build_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(
            f"{len(missing)} expected feature columns missing from input. "
            f"Filled with NaN. First few: {missing[:5]}"
        )
    return df.reindex(columns=feature_cols).astype(float)


def build_features_and_labels(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    protein_top_k: int | None = None,
    feature_selector: str = FEATURE_SELECTOR_ANOVA,
    include_clinical: bool = True,
):
    y = get_target(df)
    if feature_cols is None:
        feature_cols = select_feature_columns(
            df,
            y,
            protein_top_k=protein_top_k,
            feature_selector=feature_selector,
            include_clinical=include_clinical,
        )
    X = build_feature_frame(df, feature_cols).to_numpy()
    logger.info(
        f"Features: {sum(c.startswith('feature_') for c in feature_cols)} protein"
        f" + {sum(not c.startswith('feature_') for c in feature_cols)} clinical"
        f" = {len(feature_cols)} total"
    )
    return X, y, feature_cols
