#!/usr/bin/env python3
"""
preprocess.py — Data loading and fold-safe feature selection helpers
====================================================================
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from model import CLINICAL_FEATURES

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


def select_feature_columns(
    df: pd.DataFrame,
    y: np.ndarray,
    protein_top_k: int | None,
) -> list[str]:
    protein_cols = get_protein_columns(df)
    clinical_cols = get_clinical_columns(df)

    if protein_top_k is None or protein_top_k >= len(protein_cols):
        selected_proteins = protein_cols
    elif protein_top_k <= 0:
        selected_proteins = []
    else:
        scores, _ = f_classif(df[protein_cols], y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        top_idx = np.sort(np.argsort(scores)[-protein_top_k:])
        selected_proteins = [protein_cols[i] for i in top_idx]

    feature_cols = selected_proteins + clinical_cols
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
):
    y = get_target(df)
    if feature_cols is None:
        feature_cols = select_feature_columns(df, y, protein_top_k=protein_top_k)
    X = build_feature_frame(df, feature_cols).to_numpy()
    logger.info(
        f"Features: {sum(c.startswith('feature_') for c in feature_cols)} protein"
        f" + {sum(not c.startswith('feature_') for c in feature_cols)} clinical"
        f" = {len(feature_cols)} total"
    )
    return X, y, feature_cols
