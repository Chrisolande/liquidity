"""
Baseline reproduction pipeline (CatBoost + HistGradientBoosting + Isotonic Calibration).
Generates standalone baseline model artifacts, out-of-fold predictions, and submission files.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import Bounds, minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    SEED,
    N_SPLITS,
    TARGET,
    ID_COL,
    FLOOR,
    CEIL,
    HAS_GPU,
    get_default_dataset_paths,
)
from src.metrics import competition_score
from src.features.monthly import (
    month_columns,
    add_monthly_summary_features,
    add_cross_feature_ratios,
)
from src.features.domain import (
    add_entropy_features,
    add_behavioral_shift_features,
    add_longitudinal_stress_features,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

FINAL_BLEND_WEIGHTS = {
    "catboost": 0.9093763221312825,
    "hist_gb": 0.09062367786871744,
}




def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Engineers monthly summary, cross-feature ratios, entropy, shift, and longitudinal stress features."""
    target_col = TARGET if TARGET in train_df.columns else "Target"
    combined = pd.concat(
        [train_df.assign(_dataset="train"), test_df.assign(_dataset="test")],
        axis=0,
        ignore_index=True,
    )
    monthly_groups = month_columns(combined)
    combined = add_monthly_summary_features(combined, monthly_groups)
    combined = add_cross_feature_ratios(combined)
    monthly_groups = month_columns(combined)
    combined = add_entropy_features(combined)
    monthly_groups = month_columns(combined)
    combined = add_behavioral_shift_features(combined, monthly_groups)
    combined = add_longitudinal_stress_features(combined)

    categorical_cols = [c for c in ["gender", "region", "smartphone", "segment", "earning_pattern"] if c in combined.columns]
    for col in categorical_cols:
        combined[col] = combined[col].astype("category")

    train_fe = combined.loc[combined["_dataset"] == "train"].drop(columns=["_dataset"]).reset_index(drop=True)
    test_fe = combined.loc[combined["_dataset"] == "test"].drop(columns=["_dataset"]).reset_index(drop=True)

    feature_cols = [c for c in train_fe.columns if c not in {target_col, ID_COL}]
    numeric_cols = train_fe[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return train_fe, test_fe, categorical_cols, numeric_cols


def build_logistic_anchor(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")),
        ]
    )


def build_hist_model(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_depth=6,
                    max_iter=300,
                    min_samples_leaf=40,
                    l2_regularization=0.2,
                    max_leaf_nodes=31,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_catboost() -> CatBoostClassifier:
    task_type = "GPU" if HAS_GPU else "CPU"
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=400,
        learning_rate=0.05,
        depth=5,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bagging_temperature=0.0,
        task_type=task_type,
        random_seed=SEED,
        verbose=False,
    )


def fit_fold_model(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_cols: List[str],
) -> object:
    if model_name == "catboost":
        cat_idx = [x_train.columns.get_loc(col) for col in categorical_cols if col in x_train.columns]
        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            cat_features=cat_idx,
            use_best_model=True,
            early_stopping_rounds=50,
        )
        return model

    model.fit(x_train, y_train)
    return model


def cross_validate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, List[object]]]:
    target_col = TARGET if TARGET in train_df.columns else "Target"
    feature_cols = [c for c in train_df.columns if c not in {target_col, ID_COL}]
    X = train_df[feature_cols]
    y = train_df[target_col]
    X_test = test_df[feature_cols]

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    model_builders = {
        "catboost": lambda: build_catboost(),
        "hist_gb": lambda: build_hist_model(numeric_cols, categorical_cols),
        "logistic_anchor": lambda: build_logistic_anchor(numeric_cols, categorical_cols),
    }

    oof = pd.DataFrame(index=train_df.index)
    test_preds = pd.DataFrame(index=test_df.index)
    score_map: Dict[str, Dict[str, float]] = {}
    saved_models: Dict[str, List[object]] = {}

    for model_name, builder in model_builders.items():
        print(f"Training baseline model: {model_name}", flush=True)
        model_oof = np.zeros(len(train_df))
        fold_test_preds = []
        saved_models[model_name] = []
        fold_scores = []

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            x_train, x_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = builder()
            model = fit_fold_model(model_name, model, x_train, y_train, x_valid, y_valid, categorical_cols)

            val_pred = np.clip(model.predict_proba(x_valid)[:, 1], 1e-5, 1.0 - 1e-5)
            test_pred = np.clip(model.predict_proba(X_test)[:, 1], 1e-5, 1.0 - 1e-5)

            model_oof[valid_idx] = val_pred
            fold_test_preds.append(test_pred)
            ll, auc, comp = competition_score(y_valid.to_numpy(), val_pred)
            print(f"  Fold {fold} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Comp: {comp:.4f}", flush=True)
            fold_scores.append(comp)
            saved_models[model_name].append(model)

        ll, auc, comp = competition_score(y.to_numpy(), model_oof)
        print(
            f"{model_name} OOF | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Comp: {comp:.4f} | "
            f"CV mean+-std: {np.mean(fold_scores):.4f} +- {np.std(fold_scores):.4f}",
            flush=True,
        )
        oof[model_name] = model_oof
        test_preds[model_name] = np.mean(fold_test_preds, axis=0)
        score_map[model_name] = {
            "logloss": ll,
            "auc": auc,
            "competition_score": comp,
        }

    return oof, test_preds, score_map, saved_models


def optimize_blend_weights(oof_preds: pd.DataFrame, y_true: pd.Series) -> np.ndarray:
    initial = np.full(oof_preds.shape[1], 1.0 / oof_preds.shape[1])

    def objective(weights: np.ndarray) -> float:
        weights = np.clip(weights, 0, 1)
        weights = weights / (weights.sum() + EPS)
        blended = np.dot(oof_preds.to_numpy(), weights)
        _, _, comp = competition_score(y_true.to_numpy(), blended)
        return -comp

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = Bounds(0.0, 1.0)
    result = minimize(objective, x0=initial, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0, 1)
    weights = weights / (weights.sum() + EPS)
    return weights


def isotonic_calibrate(
    oof_blend: np.ndarray,
    y_true: pd.Series,
    test_blend: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], object]:
    sigmoid_calibrator = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="sigmoid", cv=5)
    isotonic_calibrator = CalibratedClassifierCV(LogisticRegression(max_iter=2000), method="isotonic", cv=5)

    raw_feature = oof_blend.reshape(-1, 1)
    test_feature = test_blend.reshape(-1, 1)

    sigmoid_calibrator.fit(raw_feature, y_true)
    sigmoid_oof = sigmoid_calibrator.predict_proba(raw_feature)[:, 1]
    _, _, sigmoid_score = competition_score(y_true.to_numpy(), sigmoid_oof)

    isotonic_calibrator.fit(raw_feature, y_true)
    isotonic_oof = isotonic_calibrator.predict_proba(raw_feature)[:, 1]
    isotonic_test = isotonic_calibrator.predict_proba(test_feature)[:, 1]
    _, _, isotonic_score = competition_score(y_true.to_numpy(), isotonic_oof)

    metrics = {
        "sigmoid": sigmoid_score,
        "isotonic": isotonic_score,
    }
    return isotonic_oof, isotonic_test, metrics, isotonic_calibrator


def run_baseline(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    models_dir: str = "models",
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Executes the self-contained baseline pipeline:
    Fits CatBoost + HistGB, blends weights, calibrates via isotonic regression,
    and saves checkpoints/oof_champ_train.npy & submissions/submission_best_0.73731.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_df.columns else "Target"

    print("Engineering baseline features", flush=True)
    train_fe, test_fe, categorical_cols, numeric_cols = engineer_features(train_df, test_df)

    oof_preds, test_preds, model_scores, cv_models = cross_validate_models(
        train_fe,
        test_fe,
        numeric_cols,
        categorical_cols,
    )

    optimized_weights = optimize_blend_weights(oof_preds[["catboost", "hist_gb"]], train_fe[target_col])
    w_cb = optimized_weights.get("catboost", FINAL_BLEND_WEIGHTS["catboost"])
    w_hgb = optimized_weights.get("hist_gb", FINAL_BLEND_WEIGHTS["hist_gb"])
    w_sum = w_cb + w_hgb + 1e-9
    w_cb, w_hgb = w_cb / w_sum, w_hgb / w_sum
    print(f"Dynamic baseline blend weights: catboost={w_cb:.4f}, hist_gb={w_hgb:.4f}")

    blend_oof = w_cb * oof_preds["catboost"].to_numpy() + w_hgb * oof_preds["hist_gb"].to_numpy()
    blend_test = w_cb * test_preds["catboost"].to_numpy() + w_hgb * test_preds["hist_gb"].to_numpy()

    calibrated_oof, calibrated_test, calibration_metrics, calibrator = isotonic_calibrate(
        blend_oof,
        train_fe[target_col],
        blend_test,
    )
    ll, auc, comp = competition_score(train_fe[target_col].to_numpy(), calibrated_oof)
    print(f"Baseline OOF: Comp: {comp:.5f} | AUC: {auc:.5f} | LL: {ll:.5f}", flush=True)

    # Export canonical baseline tournament anchor artifacts
    np.save(os.path.join(output_dir, "oof_champ_train.npy"), calibrated_oof)
    np.save(os.path.join(output_dir, "y_true.npy"), train_fe[target_col].to_numpy())

    sub_path = os.path.join(sub_dir, "submission_best_0.73731.csv")
    sub_df = pd.DataFrame({
        ID_COL: test_fe[ID_COL],
        "Target": np.clip(calibrated_test, FLOOR, CEIL),
    })
    sub_df.to_csv(sub_path, index=False)
    sub_df.to_csv(os.path.join(sub_dir, "submission_baseline.csv"), index=False)

    metadata = {
        "feature_count": len([c for c in train_fe.columns if c not in {target_col, ID_COL}]),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "n_splits": N_SPLITS,
        "model_scores": model_scores,
        "selected_strategy": {
            "name": "weighted_top2_calibrated",
            "weights": FINAL_BLEND_WEIGHTS,
            "optimized_weights": {
                "catboost": float(optimized_weights[0]),
                "hist_gb": float(optimized_weights[1]),
            },
        },
        "selected_oof": {"logloss": ll, "auc": auc, "competition_score": comp},
        "calibration_candidates": calibration_metrics,
        "submission_path": str(sub_path),
    }

    with open(os.path.join(output_dir, "run_summary_baseline.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(
        {"selected_strategy": "weighted_top2_calibrated", "metadata": metadata, "calibrator": calibrator},
        os.path.join(models_dir, "ensemble_baseline.joblib"),
    )
    for model_name, models in cv_models.items():
        joblib.dump(models, os.path.join(models_dir, f"{model_name}_cv_baseline.joblib"))

    return comp, calibrated_oof, calibrated_test


if __name__ == "__main__":
    run_baseline()
