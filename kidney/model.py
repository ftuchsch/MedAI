#!/usr/bin/env python3
"""
model.py — Shared configuration for the ATI ensemble pipeline
=============================================================
Defines:

- schema constants
- feature-set ladder definitions
- model registry entries
- estimator builders
- CV splitter helpers
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Shared schema constants
ID_COLUMN = "sample_id"
TARGET_COLUMN = "ati"
PROTEIN_PREFIX = "feature_"
CLINICAL_FEATURES = ["age", "sex", "baseline_egfr_23"]

# Reproducibility and evaluation defaults
CV_FOLDS = 5
CV_REPEATS = 3
RANDOM_SEED = 42
EARLY_STOPPING_ROUNDS = 100
PREDICTION_THRESHOLD = 0.5
PRIMARY_METRIC = "log_loss"
SECONDARY_METRIC = "roc_auc"


@dataclass(frozen=True)
class FeatureSetConfig:
    """Description of how a model should choose its usable columns."""

    name: str
    strategy: str
    top_k: int | None = None
    ranking_method: str = "f_classif"
    keep_clinical: bool = True
    drop_egfr: bool = False
    only_egfr: bool = False
    stability_prefilter_k: int = 1024
    stability_subsamples: int = 12
    stability_fraction: float = 0.8
    stability_threshold: float = 0.5
    stability_min_features: int = 64
    stability_max_features: int = 256


@dataclass(frozen=True)
class ModelConfig:
    """Registry entry for a trainable model family + feature set."""

    name: str
    family: str
    feature_set: str
    n_seeds: int = 1
    tree_template: str = "conservative"
    role: str = "base"
    notes: str = ""


FEATURE_SET_CONFIGS = {
    "fs0_clinical": FeatureSetConfig(
        name="fs0_clinical",
        strategy="clinical_only",
    ),
    "fs0_clinical_no_egfr": FeatureSetConfig(
        name="fs0_clinical_no_egfr",
        strategy="clinical_only",
        drop_egfr=True,
    ),
    "fs0_egfr_only": FeatureSetConfig(
        name="fs0_egfr_only",
        strategy="clinical_only",
        keep_clinical=False,
        only_egfr=True,
    ),
    "fs1_all": FeatureSetConfig(
        name="fs1_all",
        strategy="all_features",
    ),
    "fs2_top1024": FeatureSetConfig(
        name="fs2_top1024",
        strategy="top_k",
        top_k=1024,
    ),
    "fs2_top512": FeatureSetConfig(
        name="fs2_top512",
        strategy="top_k",
        top_k=512,
    ),
    "fs3_stable": FeatureSetConfig(
        name="fs3_stable",
        strategy="stability",
        stability_prefilter_k=1024,
        stability_subsamples=12,
        stability_fraction=0.8,
        stability_threshold=0.5,
        stability_min_features=64,
        stability_max_features=256,
    ),
}

# Conservative XGBoost anchor for p >> n SomaScan data.
XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_estimators": 2500,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.2,
    "reg_lambda": 8.0,
    "reg_alpha": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
}

XGB_HIGH_CAPACITY_PARAMS = {
    **XGB_BASE_PARAMS,
    "max_depth": 6,
    "min_child_weight": 8,
    "colsample_bytree": 0.15,
    "reg_lambda": 10.0,
    "reg_alpha": 0.25,
}

LGBM_BASE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.2,
    "reg_lambda": 8.0,
    "reg_alpha": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": 4,
    "verbosity": -1,
}

LGBM_HIGH_CAPACITY_PARAMS = {
    **LGBM_BASE_PARAMS,
    "num_leaves": 127,
    "min_child_samples": 30,
    "colsample_bytree": 0.15,
    "reg_lambda": 10.0,
    "reg_alpha": 0.25,
}

ELASTIC_NET_C_GRID = np.logspace(-2, 1, 5)
ELASTIC_NET_L1_RATIOS = (0.1, 0.5, 0.9)
META_C_GRID = np.logspace(-3, 2, 11)
TREE_MODEL_FAMILIES = {"xgb", "lgbm"}

MODEL_CONFIGS = {
    "xgb_fs1": ModelConfig(
        name="xgb_fs1",
        family="xgb",
        feature_set="fs1_all",
        n_seeds=3,
        tree_template="conservative",
        role="base",
        notes="All proteins plus clinical covariates.",
    ),
    "xgb_fs2": ModelConfig(
        name="xgb_fs2",
        family="xgb",
        feature_set="fs2_top1024",
        n_seeds=3,
        tree_template="conservative",
        role="base",
        notes="Within-fold filtered top-1024 proteins plus clinical covariates.",
    ),
    "xgb_fs3": ModelConfig(
        name="xgb_fs3",
        family="xgb",
        feature_set="fs3_stable",
        n_seeds=3,
        tree_template="conservative",
        role="base",
        notes="Stability-selected protein subset plus clinical covariates.",
    ),
    "lgbm_fs1": ModelConfig(
        name="lgbm_fs1",
        family="lgbm",
        feature_set="fs1_all",
        n_seeds=3,
        tree_template="conservative",
        role="base",
        notes="LightGBM on all proteins plus clinical covariates.",
    ),
    "lgbm_fs2": ModelConfig(
        name="lgbm_fs2",
        family="lgbm",
        feature_set="fs2_top1024",
        n_seeds=3,
        tree_template="conservative",
        role="base",
        notes="LightGBM on within-fold filtered top-1024 proteins plus clinical covariates.",
    ),
    "elastic_fs2": ModelConfig(
        name="elastic_fs2",
        family="elastic_net",
        feature_set="fs2_top512",
        n_seeds=1,
        role="base",
        notes="Elastic-net logistic on a within-fold filtered top-512 protein set.",
    ),
    "clinical_lr": ModelConfig(
        name="clinical_lr",
        family="elastic_net",
        feature_set="fs0_clinical",
        n_seeds=1,
        role="diagnostic",
        notes="Clinical-only benchmark.",
    ),
    "clinical_no_egfr_lr": ModelConfig(
        name="clinical_no_egfr_lr",
        family="elastic_net",
        feature_set="fs0_clinical_no_egfr",
        n_seeds=1,
        role="diagnostic",
        notes="Clinical benchmark without baseline_egfr_23.",
    ),
    "egfr_only_lr": ModelConfig(
        name="egfr_only_lr",
        family="elastic_net",
        feature_set="fs0_egfr_only",
        n_seeds=1,
        role="diagnostic",
        notes="Single-feature proxy benchmark using baseline_egfr_23 alone.",
    ),
}

DEFAULT_BASE_MODEL_NAMES = ["xgb_fs1", "xgb_fs2", "xgb_fs3", "elastic_fs2"]
DIAGNOSTIC_MODEL_NAMES = ["egfr_only_lr", "clinical_lr", "clinical_no_egfr_lr"]
DEFAULT_EVALUATION_MODEL_NAMES = DEFAULT_BASE_MODEL_NAMES + DIAGNOSTIC_MODEL_NAMES


def get_feature_set_config(name: str) -> FeatureSetConfig:
    """Return a feature-set config by name."""
    if name not in FEATURE_SET_CONFIGS:
        raise KeyError(f"Unknown feature set: {name}")
    return FEATURE_SET_CONFIGS[name]


def get_model_config(name: str) -> ModelConfig:
    """Return a model config by name."""
    if name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model config: {name}")
    return MODEL_CONFIGS[name]


def feature_set_config_to_dict(name: str) -> dict:
    """JSON-safe feature-set config export."""
    return asdict(get_feature_set_config(name))


def model_config_to_dict(name: str) -> dict:
    """JSON-safe model config export."""
    return asdict(get_model_config(name))


def get_seed_list(
    repeats: int = CV_REPEATS,
    base_seed: int = RANDOM_SEED,
    spacing: int = 101,
) -> list[int]:
    """Return deterministic shuffle seeds for repeated CV."""
    return [base_seed + spacing * idx for idx in range(repeats)]


def get_splitter(seed: int, n_splits: int = CV_FOLDS, groups=None):
    """Return a stratified splitter, upgrading to grouped CV when groups exist."""
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def get_xgb_params(template: str = "conservative", **overrides):
    """Return an XGBoost parameter dictionary by template name."""
    if template == "conservative":
        params = deepcopy(XGB_BASE_PARAMS)
    elif template == "high_capacity":
        params = deepcopy(XGB_HIGH_CAPACITY_PARAMS)
    else:
        raise KeyError(f"Unknown XGBoost template: {template}")
    params.update(overrides)
    return params


def get_lgbm_params(template: str = "conservative", **overrides):
    """Return a LightGBM parameter dictionary by template name."""
    if template == "conservative":
        params = deepcopy(LGBM_BASE_PARAMS)
    elif template == "high_capacity":
        params = deepcopy(LGBM_HIGH_CAPACITY_PARAMS)
    else:
        raise KeyError(f"Unknown LightGBM template: {template}")
    params.update(overrides)
    return params


def get_lightgbm_module():
    """Import LightGBM lazily so non-LightGBM runs still work."""
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for lgbm_* models. Install the 'lightgbm' package first."
        ) from exc
    return lgb


def build_xgb_estimator(
    seed: int,
    *,
    template: str = "conservative",
    n_estimators: int | None = None,
    early_stopping: bool = False,
    n_jobs: int | None = None,
):
    """Return a fresh XGBoost classifier."""
    params = get_xgb_params(template=template, random_state=seed)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    if early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBClassifier(**params)


def build_lgbm_estimator(
    seed: int,
    *,
    template: str = "conservative",
    n_estimators: int | None = None,
    n_jobs: int | None = None,
):
    """Return a fresh LightGBM classifier."""
    lgb = get_lightgbm_module()
    params = get_lgbm_params(template=template, random_state=seed)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    return lgb.LGBMClassifier(**params)


def build_elastic_net_estimator(seed: int, *, n_jobs: int | None = None):
    """Return an elastic-net logistic pipeline with fold-internal scaling."""
    elastic_n_jobs = -1 if n_jobs is None else int(n_jobs)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=ELASTIC_NET_C_GRID,
                    cv=3,
                    penalty="elasticnet",
                    solver="saga",
                    scoring="neg_log_loss",
                    l1_ratios=ELASTIC_NET_L1_RATIOS,
                    max_iter=4000,
                    n_jobs=elastic_n_jobs,
                    refit=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_ridge_meta_learner(seed: int):
    """Return a log-loss-tuned ridge stacker."""
    return LogisticRegressionCV(
        Cs=META_C_GRID,
        cv=3,
        penalty="l2",
        solver="liblinear",
        scoring="neg_log_loss",
        max_iter=2000,
        refit=True,
        random_state=seed,
    )


def build_calibrated_meta_learner(seed: int):
    """Return a ridge stacker wrapped in Platt scaling."""
    return CalibratedClassifierCV(
        estimator=build_ridge_meta_learner(seed),
        method="sigmoid",
        cv=3,
    )


def build_estimator(
    model_name: str,
    seed: int,
    *,
    n_estimators: int | None = None,
    early_stopping: bool = False,
    xgb_n_jobs: int | None = None,
    lgbm_n_jobs: int | None = None,
    elastic_n_jobs: int | None = None,
):
    """Return a fresh estimator from the model registry."""
    config = get_model_config(model_name)
    if config.family == "xgb":
        return build_xgb_estimator(
            seed,
            template=config.tree_template,
            n_estimators=n_estimators,
            early_stopping=early_stopping,
            n_jobs=xgb_n_jobs,
        )
    if config.family == "lgbm":
        return build_lgbm_estimator(
            seed,
            template=config.tree_template,
            n_estimators=n_estimators,
            n_jobs=lgbm_n_jobs,
        )
    if config.family == "elastic_net":
        return build_elastic_net_estimator(seed, n_jobs=elastic_n_jobs)
    raise KeyError(f"Unsupported model family: {config.family}")


if __name__ == "__main__":
    print("Feature sets:")
    for name, cfg in FEATURE_SET_CONFIGS.items():
        print(f"  {name}: {cfg}")
    print("\nModels:")
    for name, cfg in MODEL_CONFIGS.items():
        print(f"  {name}: {cfg}")
