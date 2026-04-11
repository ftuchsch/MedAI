#!/usr/bin/env python3
"""
model.py — Shared model builders and experiment configuration
==============================================================
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Clinical features included alongside the SomaScan protein abundances.
CLINICAL_FEATURES = ["age", "sex", "baseline_egfr_23"]

FEATURE_SELECTOR_ANOVA = "anova"
FEATURE_SELECTOR_STABLE_ANOVA = "stable_anova"

CV_FOLDS = 5
RANDOM_SEED = 42
STABILITY_SELECTION_INNER_SEEDS = [11, 13, 17, 19, 23]

STARTER_XGBOOST_TOP_K = None
SELECTED_XGBOOST_TOP_K = 90
SELECTED_PROTEOMICS_ONLY_XGBOOST_TOP_K = 90
SELECTED_RIDGE_TOP_K = 80
SELECTED_LDA_TOP_K = 120
SELECTED_PLS_TOP_K = 260

STARTER_XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.1,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

SELECTED_XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 1,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 4,
    "reg_alpha": 0.0,
    "reg_lambda": 3.0,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

SELECTED_RIDGE_PARAMS = {
    "C": 0.02,
    "max_iter": 5000,
}

SELECTED_LDA_PARAMS = {
    "solver": "lsqr",
    "shrinkage": "auto",
}

SELECTED_PLS_PARAMS = {
    "n_components": 14,
    "C": 0.5,
    "max_iter": 5000,
}

ENSEMBLE_COMPONENTS = [
    {
        "name": "Selected XGBoost",
        "weight": 0.41,
        "artifact_type": "xgboost",
        "feature_file": "feature_cols.json",
        "model_file": "xgboost_model.json",
    },
    {
        "name": "Selected Proteomics-Only XGBoost",
        "weight": 0.12,
        "artifact_type": "xgboost",
        "feature_file": "protein_only_feature_cols.json",
        "model_file": "protein_only_xgboost_model.json",
    },
    {
        "name": "Selected Ridge LR",
        "weight": 0.32,
        "artifact_type": "sklearn",
        "feature_file": "ridge_feature_cols.json",
        "model_file": "ridge_model.joblib",
    },
    {
        "name": "Selected PLS LR",
        "weight": 0.15,
        "artifact_type": "sklearn",
        "feature_file": "pls_feature_cols.json",
        "model_file": "pls_model.joblib",
    },
]

BASE_MODEL_SPECS = {
    "Starter XGBoost": {
        "builder": "starter_xgboost",
        "protein_top_k": STARTER_XGBOOST_TOP_K,
        "feature_selector": FEATURE_SELECTOR_ANOVA,
        "include_clinical": True,
        "artifact_type": "xgboost",
    },
    "Selected XGBoost": {
        "builder": "selected_xgboost",
        "protein_top_k": SELECTED_XGBOOST_TOP_K,
        "feature_selector": FEATURE_SELECTOR_STABLE_ANOVA,
        "include_clinical": True,
        "artifact_type": "xgboost",
    },
    "Selected Proteomics-Only XGBoost": {
        "builder": "selected_xgboost",
        "protein_top_k": SELECTED_PROTEOMICS_ONLY_XGBOOST_TOP_K,
        "feature_selector": FEATURE_SELECTOR_STABLE_ANOVA,
        "include_clinical": False,
        "artifact_type": "xgboost",
    },
    "Selected Ridge LR": {
        "builder": "selected_ridge_lr",
        "protein_top_k": SELECTED_RIDGE_TOP_K,
        "feature_selector": FEATURE_SELECTOR_ANOVA,
        "include_clinical": True,
        "artifact_type": "sklearn",
    },
    "Selected Shrinkage LDA": {
        "builder": "selected_lda",
        "protein_top_k": SELECTED_LDA_TOP_K,
        "feature_selector": FEATURE_SELECTOR_ANOVA,
        "include_clinical": True,
        "artifact_type": "sklearn",
    },
    "Selected PLS LR": {
        "builder": "selected_pls_lr",
        "protein_top_k": SELECTED_PLS_TOP_K,
        "feature_selector": FEATURE_SELECTOR_ANOVA,
        "include_clinical": True,
        "artifact_type": "sklearn",
    },
}

ENSEMBLE_SPECS = {
    "Selected Ensemble (XGB + ProtXGB + Ridge + PLS)": {
        "components": [
            {"name": "Selected XGBoost", "weight": 0.41},
            {"name": "Selected Proteomics-Only XGBoost", "weight": 0.12},
            {"name": "Selected Ridge LR", "weight": 0.32},
            {"name": "Selected PLS LR", "weight": 0.15},
        ],
    }
}


class PLSProjector(BaseEstimator, TransformerMixin):
    """Fit supervised PLS components, then expose the latent X scores to sklearn pipelines."""

    def __init__(self, n_components: int = 14):
        self.n_components = n_components
        self.model_ = None

    def fit(self, X, y):
        self.model_ = PLSRegression(n_components=self.n_components, scale=False)
        self.model_.fit(X, y)
        return self

    def transform(self, X):
        return self.model_.transform(X)


def build_starter_xgboost(random_state: int = RANDOM_SEED):
    return XGBClassifier(random_state=random_state, **STARTER_XGBOOST_PARAMS)


def build_selected_xgboost(random_state: int = RANDOM_SEED):
    return XGBClassifier(random_state=random_state, **SELECTED_XGBOOST_PARAMS)


def build_selected_ridge_lr(random_state: int = RANDOM_SEED):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_state, **SELECTED_RIDGE_PARAMS)),
        ]
    )


def build_selected_lda(random_state: int = RANDOM_SEED):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(**SELECTED_LDA_PARAMS)),
        ]
    )


def build_selected_pls_lr(random_state: int = RANDOM_SEED):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pls", PLSProjector(n_components=SELECTED_PLS_PARAMS["n_components"])),
            (
                "clf",
                LogisticRegression(
                    C=SELECTED_PLS_PARAMS["C"],
                    max_iter=SELECTED_PLS_PARAMS["max_iter"],
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_model(builder_name: str, random_state: int = RANDOM_SEED):
    builders = {
        "starter_xgboost": build_starter_xgboost,
        "selected_xgboost": build_selected_xgboost,
        "selected_ridge_lr": build_selected_ridge_lr,
        "selected_lda": build_selected_lda,
        "selected_pls_lr": build_selected_pls_lr,
    }
    if builder_name not in builders:
        raise ValueError(f"Unknown builder '{builder_name}'. Choose from: {sorted(builders)}")
    return builders[builder_name](random_state=random_state)


if __name__ == "__main__":
    for label, spec in BASE_MODEL_SPECS.items():
        model = build_model(spec["builder"])
        print(
            f"{label:<20} builder={spec['builder']:<18} "
            f"protein_top_k={spec['protein_top_k']}"
        )
        print(f"  model={type(model).__name__}")
