#!/usr/bin/env python3
"""
preprocess.py — Data loading and fold-safe feature selection helpers
====================================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from model import CLINICAL_FEATURES

logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load a pre-processed CSV and log its shape."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False, na_values=[".", ""])
    logger.info(f"  {len(df)} samples, {len(df.columns)} columns")
    return df


def get_target(df: pd.DataFrame) -> np.ndarray:
    """Extract the binary ATI target."""
    if "ati" not in df.columns:
        raise KeyError("Expected target column 'ati' to be present.")
    y = df["ati"].astype(int).to_numpy()
    logger.info(f"Samples: {len(y)} | No ATI: {(y == 0).sum()} | ATI: {(y == 1).sum()}")
    return y


def get_protein_columns(df: pd.DataFrame) -> list[str]:
    """Return lexicographically sorted protein columns."""
    return sorted([c for c in df.columns if c.startswith("feature_")])


def get_clinical_columns(df: pd.DataFrame) -> list[str]:
    """Return the clinical covariates that are present in the input frame."""
    return [c for c in CLINICAL_FEATURES if c in df.columns]


def score_proteins(df: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Rank proteins by univariate ANOVA F-score inside the current dataset."""
    protein_cols = get_protein_columns(df)
    if not protein_cols:
        return pd.DataFrame(columns=["feature", "score"])

    scores, _ = f_classif(df[protein_cols], y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        pd.DataFrame({"feature": protein_cols, "score": scores})
        .sort_values(["score", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def select_feature_columns(
    df: pd.DataFrame,
    y: np.ndarray,
    protein_top_k: int | None,
) -> list[str]:
    """
    Select the top-k protein columns and always keep the available clinical features.

    The ranking must be computed on training data only during CV to avoid leakage.
    """
    protein_cols = get_protein_columns(df)
    clinical_cols = get_clinical_columns(df)

    if protein_top_k is None or protein_top_k >= len(protein_cols):
        selected_proteins = protein_cols
    elif protein_top_k <= 0:
        selected_proteins = []
    else:
        ranking = score_proteins(df, y)
        selected_proteins = ranking["feature"].head(protein_top_k).tolist()

    feature_cols = selected_proteins + clinical_cols
    if not feature_cols:
        raise ValueError("No feature columns were selected.")
    return feature_cols


def build_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Align a DataFrame to the expected feature order."""
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
    """Build the feature matrix, target vector, and selected feature list."""
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
