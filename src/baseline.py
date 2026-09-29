"""
Baseline tournament reproduction pipeline (CatBoost GPU + XGBoost GPU + HistGB + Beta/Platt Calibration).
Integrates feature_engine selection, 3-model diversity, dynamic SLSQP blending, and leakage-free calibration.
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
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.optimize import Bounds, minimize
from sklearn.model_selection import StratifiedKFold

from src.config import (
    SEED,
    N_SPLITS,
    TARGET,
    ID_COL,
    FLOOR,
    CEIL,
    HAS_GPU,
    EPS,
    get_default_dataset_paths,
)
from src.metrics import competition_score
from src.features.pipeline import engineer_features
from src.features.selection import run_feature_engine_selection
from src.ensemble.calibration import beta_calibrate

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def build_catboost(seed: int = SEED) -> CatBoostClassifier:
    task_type = "GPU" if HAS_GPU else "CPU"
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=800,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=10.0,
        random_strength=0.5,
        bagging_temperature=0.0,
        task_type=task_type,
        random_seed=seed,
        verbose=False,
    )


def build_xgboost(seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        tree_method="hist",
        device="cuda" if HAS_GPU else "cpu",
        random_state=seed,
        seed=seed,
        eval_metric="logloss",
    )


def build_hist_model(cat_indices: List[int], seed: int = SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.035,
        max_depth=6,
        max_iter=350,
        min_samples_leaf=30,
        l2_regularization=0.5,
        categorical_features=cat_indices,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=25,
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
            early_stopping_rounds=50,
            verbose=False,
        )
        return model

    if model_name == "xgboost":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        return model

    if model_name == "hist_gb":
        # Convert category dtypes to numeric integer codes for HistGB
        x_tr_h = x_train.copy()
        for col in categorical_cols:
            if col in x_tr_h.columns:
                x_tr_h[col] = x_tr_h[col].cat.codes.replace(-1, np.nan)
        model.fit(x_tr_h, y_train)
        return model

    model.fit(x_train, y_train)
    return model


def predict_model_probs(
    model_name: str,
    model: object,
    X: pd.DataFrame,
    categorical_cols: List[str],
) -> np.ndarray:
    if model_name == "hist_gb":
        X_h = X.copy()
        for col in categorical_cols:
            if col in X_h.columns:
                X_h[col] = X_h[col].cat.codes.replace(-1, np.nan)
        return np.clip(model.predict_proba(X_h)[:, 1], FLOOR, CEIL)
    return np.clip(model.predict_proba(X)[:, 1], FLOOR, CEIL)


def cross_validate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, List[object]]]:
    target_col = TARGET if TARGET in train_df.columns else "Target"
    feature_cols = [c for c in train_df.columns if c not in {target_col, ID_COL}]
    X = train_df[feature_cols].copy()
    y = train_df[target_col]
    X_test = test_df[feature_cols].copy()

    # Ensure category dtype for tree models
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    cat_idx = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    model_builders = {
        "catboost": lambda fold: build_catboost(seed=seed + fold),
        "xgboost": lambda fold: build_xgboost(seed=seed + fold),
        "hist_gb": lambda fold: build_hist_model(cat_indices=cat_idx, seed=seed + fold),
    }

    oof = pd.DataFrame(index=train_df.index)
    test_preds = pd.DataFrame(index=test_df.index)
    score_map: Dict[str, Dict[str, float]] = {}
    saved_models: Dict[str, List[object]] = {}

    for model_name, builder in model_builders.items():
        print(f"  [seed={seed}] Training model: {model_name}", flush=True)
        model_oof = np.zeros(len(train_df))
        fold_test_preds = []
        saved_models[model_name] = []
        fold_scores = []

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            x_train, x_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = builder(fold)
            model = fit_fold_model(model_name, model, x_train, y_train, x_valid, y_valid, categorical_cols)

            val_pred = predict_model_probs(model_name, model, x_valid, categorical_cols)
            test_pred = predict_model_probs(model_name, model, X_test, categorical_cols)

            model_oof[valid_idx] = val_pred
            fold_test_preds.append(test_pred)
            ll, auc, comp = competition_score(y_valid.to_numpy(), val_pred)
            print(f"    Fold {fold} | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Comp: {comp:.4f}", flush=True)
            fold_scores.append(comp)
            saved_models[model_name].append(model)

        ll, auc, comp = competition_score(y.to_numpy(), model_oof)
        print(
            f"  {model_name} OOF | LogLoss: {ll:.4f} | AUC: {auc:.4f} | Comp: {comp:.4f} | "
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
    n_models = oof_preds.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

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


def run_baseline(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    models_dir: str = "models",
    k_top_features: int = 60,
    seeds: Tuple[int, ...] = (42, 100, 2024, 777),
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Executes the upgraded baseline pipeline with multi-seed averaging:
    1. Full feature engineering (monthly + domain + stress + solvency + interactions)
    2. feature_engine selection (DropConstant + DropDuplicate + SmartCorrelatedSelection + CV importance)
    3. 3-way diverse GBDT ensemble (CatBoost GPU + XGBoost GPU + HistGB) × N seeds
    4. Per-seed SLSQP dynamic blend optimization, then raw OOF/test averaged across seeds
    5. Non-degrading Beta/Platt calibration (replaces AUC-damaging isotonic)
    6. Saves canonical anchor artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_df.columns else "Target"

    print("Step 1: Engineering comprehensive feature set", flush=True)
    train_fe, test_fe, categorical_cols, numeric_cols = engineer_features(train_df, test_df)

    y_train = train_fe[target_col].to_numpy(int)
    X_tr_raw = train_fe.drop(columns=[target_col, ID_COL])
    X_te_raw = test_fe.drop(columns=[c for c in [ID_COL] if c in test_fe.columns])

    print(f"Step 2: Running feature_engine selection (targeting ~{k_top_features} features)", flush=True)
    X_tr_sel, X_te_sel, selected_features = run_feature_engine_selection(
        X_tr_raw, y_train, X_te_raw, cat_cols=categorical_cols, k_top=k_top_features, corr_threshold=0.98, seed=SEED
    )

    # Reconstruct dataframes for cross_validate_models
    train_selected = X_tr_sel.copy()
    train_selected[target_col] = y_train
    test_selected = X_te_sel.copy()
    test_selected[ID_COL] = test_fe[ID_COL].values

    active_cats = [c for c in categorical_cols if c in selected_features]
    active_nums = [c for c in selected_features if c not in active_cats]

    print(
        f"Step 3: Multi-seed cross-validation — seeds={list(seeds)}, "
        f"features={len(selected_features)} ({len(active_nums)} numeric + {len(active_cats)} categorical)",
        flush=True,
    )

    seed_raw_oof: List[np.ndarray] = []
    seed_raw_test: List[np.ndarray] = []

    for i, seed in enumerate(seeds, start=1):
        print(f"\n--- Seed {i}/{len(seeds)} : seed={seed} ---", flush=True)
        oof_preds, test_preds, model_scores, cv_models = cross_validate_models(
            train_selected,
            test_selected,
            active_nums,
            active_cats,
            seed=seed,
        )

        # Per-seed SLSQP blend
        model_names = ["catboost", "xgboost", "hist_gb"]
        optimized_weights = optimize_blend_weights(oof_preds[model_names], pd.Series(y_train))
        weights_dict = {name: float(w) for name, w in zip(model_names, optimized_weights)}

        s_oof = np.zeros(len(train_df))
        s_test = np.zeros(len(test_df))
        for name, w in weights_dict.items():
            s_oof += w * oof_preds[name].to_numpy()
            s_test += w * test_preds[name].to_numpy()

        s_ll, s_auc, s_comp = competition_score(y_train, s_oof)
        print(
            f"  [seed={seed}] Blend OOF: Comp={s_comp:.5f} | AUC={s_auc:.5f} | LL={s_ll:.5f} | "
            f"weights={weights_dict}",
            flush=True,
        )
        seed_raw_oof.append(s_oof)
        seed_raw_test.append(s_test)

        # Save per-seed models
        for model_name, models in cv_models.items():
            joblib.dump(models, os.path.join(models_dir, f"{model_name}_seed{seed}_baseline.joblib"))

    # Average across seeds
    raw_oof = np.mean(seed_raw_oof, axis=0)
    raw_test = np.mean(seed_raw_test, axis=0)

    raw_ll, raw_auc, raw_comp = competition_score(y_train, raw_oof)
    print(f"\nStep 4: {len(seeds)}-seed averaged blend OOF: Comp={raw_comp:.5f} | AUC={raw_auc:.5f} | LL={raw_ll:.5f}", flush=True)

    # Step 5: Beta calibration (strictly more expressive than Platt — superset with 3 params vs 2)
    print("Step 5: Beta calibration", flush=True)
    calibrated_oof, calibrated_test, calibrator = beta_calibrate(raw_oof, raw_test, y_train, n_splits=5, seed=SEED)
    best_ll, best_auc, best_comp = competition_score(y_train, calibrated_oof)
    print(f">>> Beta Calibrated: Comp={best_comp:.5f} | AUC={best_auc:.5f} | LL={best_ll:.5f}", flush=True)

    # Export canonical baseline tournament anchor artifacts
    np.save(os.path.join(output_dir, "oof_champ_train.npy"), calibrated_oof)
    np.save(os.path.join(output_dir, "y_true.npy"), y_train)
    np.save(os.path.join(output_dir, "oof_raw_blend.npy"), raw_oof)  # pre-calibration checkpoint

    sub_path = os.path.join(sub_dir, "submission_baseline.csv")
    sub_df = pd.DataFrame({
        ID_COL: test_fe[ID_COL],
        "Target": np.clip(calibrated_test, FLOOR, CEIL),
    })
    sub_df.to_csv(sub_path, index=False)

    metadata = {
        "seeds": list(seeds),
        "feature_count": len(selected_features),
        "numeric_features": len(active_nums),
        "categorical_features": len(active_cats),
        "n_splits": N_SPLITS,
        "selected_strategy": {
            "name": "multiseed_slsqp_blend_beta_calibrated",
        },
        "oof_scores": {
            "raw_blend": {"comp": raw_comp, "auc": raw_auc, "ll": raw_ll},
            "beta_calibrated": {"comp": best_comp, "auc": best_auc, "ll": best_ll},
        },
        "submission_path": str(sub_path),
    }

    with open(os.path.join(output_dir, "run_summary_baseline.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(
        {"selected_strategy": f"multiseed_slsqp_blend_{best_cal_name}_calibrated", "metadata": metadata, "calibrator": calibrator},
        os.path.join(models_dir, "ensemble_baseline.joblib"),
    )

    return best_comp, calibrated_oof, calibrated_test


if __name__ == "__main__":
    run_baseline()

