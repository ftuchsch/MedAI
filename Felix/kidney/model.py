#!/usr/bin/env python3
"""
model.py — TabPFN model configuration and artifact constants
============================================================
Central place for the Felix kidney pipeline configuration.

This branch now deploys a single model family:
    TabPFN v2 classifier on top-k proteins + clinical covariates
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

# Clinical features included alongside the protein abundances.
CLINICAL_FEATURES = ["age", "sex", "baseline_egfr_23"]

CV_FOLDS = 5
RANDOM_SEED = 42

DEFAULT_MODEL_NAME = "TabPFN"
DEFAULT_PROTEIN_TOP_K = 100
DEFAULT_TABPFN_DEVICE = "cpu"
DEFAULT_TABPFN_N_ESTIMATORS = 8
DEFAULT_TABPFN_SOFTMAX_TEMPERATURE = 1.1
DEFAULT_TABPFN_AVERAGE_BEFORE_SOFTMAX = True
DEFAULT_TABPFN_BALANCE_PROBABILITIES = False
DEFAULT_PROBABILITY_SHRINKAGE = 0.0

MODEL_FILENAME = "tabpfn_model.joblib"
FEATURES_FILENAME = "feature_cols.json"
METADATA_FILENAME = "model_metadata.json"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    protein_top_k: int
    artifact_type: str
    model_file: str
    feature_file: str
    metadata_file: str


TABPFN_SPEC = ModelSpec(
    name=DEFAULT_MODEL_NAME,
    protein_top_k=DEFAULT_PROTEIN_TOP_K,
    artifact_type="joblib",
    model_file=MODEL_FILENAME,
    feature_file=FEATURES_FILENAME,
    metadata_file=METADATA_FILENAME,
)


def get_model_names() -> list[str]:
    """Return the active model names for evaluation loops."""
    return [TABPFN_SPEC.name]


def get_model_spec(name: str = DEFAULT_MODEL_NAME) -> ModelSpec:
    """Return the TabPFN model specification."""
    if name != TABPFN_SPEC.name:
        raise ValueError(f"Unknown model '{name}'. Choose from: {get_model_names()}")
    return TABPFN_SPEC


def get_tabpfn_version() -> str:
    """Return the installed TabPFN package version without importing the package."""
    try:
        return version("tabpfn")
    except PackageNotFoundError:
        return "not-installed"


def build_model(
    name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = DEFAULT_TABPFN_DEVICE,
    n_estimators: int = DEFAULT_TABPFN_N_ESTIMATORS,
    softmax_temperature: float = DEFAULT_TABPFN_SOFTMAX_TEMPERATURE,
    average_before_softmax: bool = DEFAULT_TABPFN_AVERAGE_BEFORE_SOFTMAX,
    balance_probabilities: bool = DEFAULT_TABPFN_BALANCE_PROBABILITIES,
):
    """Return a fresh unfitted TabPFN classifier."""
    if name != TABPFN_SPEC.name:
        raise ValueError(f"Unknown model '{name}'. Choose from: {get_model_names()}")

    # Import lazily so lightweight tasks like py_compile or metadata inspection
    # do not force a heavyweight TabPFN import.
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device=device,
        fit_mode="fit_preprocessors",
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        balance_probabilities=balance_probabilities,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


if __name__ == "__main__":
    print(f"Active model       : {DEFAULT_MODEL_NAME}")
    print(f"Clinical features  : {CLINICAL_FEATURES}")
    print(f"Default protein k  : {DEFAULT_PROTEIN_TOP_K}")
    print(f"Default estimators : {DEFAULT_TABPFN_N_ESTIMATORS}")
    print(f"Default softmax T  : {DEFAULT_TABPFN_SOFTMAX_TEMPERATURE}")
    print(f"CV folds           : {CV_FOLDS}")
    print(f"Random seed        : {RANDOM_SEED}")
    print(f"TabPFN version     : {get_tabpfn_version()}")
