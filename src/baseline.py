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
import lightgbm as lgb
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


def build_composite_strata(df: pd.DataFrame, target_col: str = "Target", n_splits: int = 10) -> pd.Series:
    """
    Constructs joint multi-strata cohort keys:
    Target x Customer Segment x Balance Quartile x Earning Pattern.
    Groups rare combinations (< n_splits members) into fallback strata.
    """
    y_str = df[target_col].astype(str) if target_col in df.columns else pd.Series("0", index=df.index)
    seg_str = df["segment"].astype(str) if "segment" in df.columns else pd.Series("unk", index=df.index)

    earn_col = None
    for cand in ["earning_pattern", "financial_earning_indicator"]:
        if cand in df.columns:
            earn_col = cand
            break
    earn_str = df[earn_col].astype(str) if earn_col else pd.Series("unk", index=df.index)

    bal_col = None
    for cand in ["m1_daily_avg_bal", "agg_daily_avg_bal_mean", "agg_daily_avg_bal_recent3_to_old3_ratio"]:
        if cand in df.columns:
            bal_col = cand
            break
    if bal_col:
        bal_str = pd.qcut(df[bal_col].rank(method="first"), q=4, labels=["q1", "q2", "q3", "q4"]).astype(str)
    else:
        bal_str = pd.Series("unk", index=df.index)

    composite = y_str + "_" + seg_str + "_" + bal_str + "_" + earn_str
    counts = composite.value_counts()
    rare_strata = counts[counts < n_splits].index
    if len(rare_strata) > 0:
        composite = composite.mask(composite.isin(rare_strata), y_str + "_rare")
    return composite


def build_catboost(seed: int = SEED) -> CatBoostClassifier:
    task_type = "GPU" if HAS_GPU else "CPU"
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=900,
        learning_rate=0.045,         # Optuna tuned
        depth=6,                     # Optuna tuned (d=6 prevents over-partitioning)
        l2_leaf_reg=10.0,            # Optuna tuned (lighter reg restores gradient steps)
        random_strength=1.5,         # Optuna tuned
        bagging_temperature=0.4,     # Optuna tuned
        border_count=128,
        task_type=task_type,
        random_seed=seed,
        verbose=False,
    )


def build_xgboost(seed: int = SEED) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=800,
        learning_rate=0.025,         # Optuna tuned
        max_depth=4,                 # Optuna tuned (d=4 decisively beat d=5/6)
        min_child_weight=3.0,        # Optuna tuned
        subsample=0.75,              # Optuna tuned
        colsample_bytree=0.70,       # Optuna tuned
        reg_lambda=2.5,              # Optuna tuned
        reg_alpha=0.45,              # Optuna tuned (higher L1 penalty ignores noisy splits)
        enable_categorical=True,
        tree_method="hist",
        device="cuda" if HAS_GPU else "cpu",
        random_state=seed,
        seed=seed,
        eval_metric="logloss",
    )


def build_lightgbm(seed: int = SEED) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.030,
        num_leaves=35,               # Optuna tuned (optimal balance without leaf sparsity)
        min_child_samples=30,        # Optuna tuned
        colsample_bytree=0.70,       # Optuna tuned
        subsample=0.85,              # Optuna tuned
        subsample_freq=1,
        reg_alpha=0.25,              # Optuna tuned
        reg_lambda=1.0,              # Optuna tuned
        random_state=seed,
        seed=seed,
        bagging_seed=seed + 11,
        feature_fraction_seed=seed + 22,
        extra_seed=seed + 33,
        data_random_seed=seed + 44,
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        n_jobs=-1,
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
            cat_features=cat_idx if cat_idx else None,
            early_stopping_rounds=75,
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

    if model_name == "lightgbm":
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        return model

    model.fit(x_train, y_train)
    return model


def predict_model_probs(
    model_name: str,
    model: object,
    X: pd.DataFrame,
    categorical_cols: List[str],
) -> np.ndarray:
    return np.clip(model.predict_proba(X)[:, 1], FLOOR, CEIL)


def cross_validate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    strata: pd.Series = None,
    n_splits: int = 10,
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

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    split_target = strata if strata is not None else y

    model_builders = {
        "catboost": lambda fold: build_catboost(seed=seed + fold),
        "xgboost": lambda fold: build_xgboost(seed=seed + fold),
        "lightgbm": lambda fold: build_lightgbm(seed=seed + fold),
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

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, split_target), start=1):
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
    n_splits: int = 10,
    seeds: Tuple[int, ...] = (42,),
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Executes the upgraded baseline pipeline with family-level multi-seed averaging:
    1. Full feature engineering (monthly + domain + stress + solvency + interactions)
    2. feature_engine selection (DropConstant + DropDuplicate + SmartCorrelatedSelection + CV importance)
    3. 3-way diverse GBDT ensemble (CatBoost GPU + XGBoost GPU + HistGB) × N seeds in 10-fold CV
    4. Per-family seed averaging to denoise model streams
    5. Single unified constrained SLSQP simplex blend over the 3 model families
    6. Cross-fitted Beta calibration with protective fallback gating
    7. Saves canonical anchor artifacts
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

    # Construct multi-strata joint cohort key to balance difficulty across folds
    strata = build_composite_strata(train_fe, target_col=target_col, n_splits=n_splits)
    print(f"Built multi-strata cohort keys ({strata.nunique()} unique strata) for balanced CV splitting", flush=True)

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
        f"Step 3: Multi-seed cross-validation — {n_splits}-fold CV, seeds={list(seeds)}, "
        f"features={len(selected_features)} ({len(active_nums)} numeric + {len(active_cats)} categorical)",
        flush=True,
    )

    model_names = ["catboost", "xgboost", "lightgbm"]
    family_oof: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}
    family_test: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}

    for i, seed in enumerate(seeds, start=1):
        print(f"\n--- Seed {i}/{len(seeds)} : seed={seed} ---", flush=True)
        oof_preds, test_preds, model_scores, cv_models = cross_validate_models(
            train_selected,
            test_selected,
            active_nums,
            active_cats,
            strata=strata,
            n_splits=n_splits,
            seed=seed,
        )

        for name in model_names:
            family_oof[name].append(oof_preds[name].to_numpy())
            family_test[name].append(test_preds[name].to_numpy())

        # Save per-seed models
        for model_name, models in cv_models.items():
            joblib.dump(models, os.path.join(models_dir, f"{model_name}_seed{seed}_baseline.joblib"))

    # Compute family-averaged OOF and Test predictions
    print("\n--- Model Family Averaging Across Seeds ---", flush=True)
    mean_family_oof_dict = {}
    mean_family_test_dict = {}
    for name in model_names:
        mean_family_oof_dict[name] = np.mean(family_oof[name], axis=0)
        mean_family_test_dict[name] = np.mean(family_test[name], axis=0)
        f_ll, f_auc, f_comp = competition_score(y_train, mean_family_oof_dict[name])
        print(
            f"  {name:10s} ({len(seeds)} seeds avg) | LogLoss: {f_ll:.4f} | AUC: {f_auc:.4f} | Comp: {f_comp:.4f}",
            flush=True,
        )

    # Diagnostic: Leave-One-Out Seed Evaluation for CatBoost
    if len(seeds) > 2:
        print("\n--- Seed Sensitivity Check (CatBoost Leave-One-Out) ---", flush=True)
        for idx, s in enumerate(seeds):
            cb_without_s = np.mean([family_oof["catboost"][j] for j in range(len(seeds)) if j != idx], axis=0)
            loo_ll, loo_auc, loo_comp = competition_score(y_train, cb_without_s)
            print(f"  CatBoost without seed {s:4d} | LogLoss: {loo_ll:.4f} | AUC: {loo_auc:.4f} | Comp: {loo_comp:.4f}", flush=True)

    mean_family_oof = pd.DataFrame(mean_family_oof_dict)
    mean_family_test = pd.DataFrame(mean_family_test_dict)

    # Step 4: One unified constrained simplex blend across the 3 family averages
    print("\nStep 4: Unified Constrained Family Blend (SLSQP)", flush=True)
    optimized_weights = optimize_blend_weights(mean_family_oof[model_names], pd.Series(y_train))
    weights_dict = {name: float(w) for name, w in zip(model_names, optimized_weights)}
    print(f"  Optimized family weights: {weights_dict}", flush=True)

    raw_oof = np.zeros(len(train_df))
    raw_test = np.zeros(len(test_df))
    for name, w in weights_dict.items():
        raw_oof += w * mean_family_oof[name].to_numpy()
        raw_test += w * mean_family_test[name].to_numpy()

    raw_ll, raw_auc, raw_comp = competition_score(y_train, raw_oof)
    print(
        f"  Unified Blend Raw OOF: Comp={raw_comp:.5f} | AUC={raw_auc:.5f} | LL={raw_ll:.5f}",
        flush=True,
    )

    # Step 5: Beta calibration with protective fallback gating
    print("Step 5: Cross-fitted Beta calibration", flush=True)
    calibrated_oof, calibrated_test, calibrator = beta_calibrate(raw_oof, raw_test, y_train, n_splits=5, seed=SEED)
    cal_ll, cal_auc, cal_comp = competition_score(y_train, calibrated_oof)
    print(f"  Beta Calibrated: Comp={cal_comp:.5f} | AUC={cal_auc:.5f} | LL={cal_ll:.5f}", flush=True)

    # Protect against calibration degradation
    if cal_comp >= raw_comp:
        final_oof = calibrated_oof
        final_test = calibrated_test
        selected_strategy_name = "multiseed_slsqp_family_blend_beta_calibrated"
        best_comp, best_auc, best_ll = cal_comp, cal_auc, cal_ll
        print("  --> Adopted Beta Calibration.", flush=True)
    else:
        final_oof = raw_oof
        final_test = raw_test
        selected_strategy_name = "multiseed_slsqp_family_blend_raw"
        best_comp, best_auc, best_ll = raw_comp, raw_auc, raw_ll
        print("  --> Preserved Raw Blend (Calibration did not improve composite).", flush=True)

    # Export canonical baseline tournament anchor artifacts
    np.save(os.path.join(output_dir, "oof_champ_train.npy"), final_oof)
    np.save(os.path.join(output_dir, "y_true.npy"), y_train)
    np.save(os.path.join(output_dir, "oof_raw_blend.npy"), raw_oof)

    sub_path = os.path.join(sub_dir, "submission_baseline.csv")
    sub_df = pd.DataFrame({
        ID_COL: test_fe[ID_COL],
        "Target": np.clip(final_test, FLOOR, CEIL),
    })
    sub_df.to_csv(sub_path, index=False)

    metadata = {
        "seeds": list(seeds),
        "n_splits": n_splits,
        "feature_count": len(selected_features),
        "numeric_features": len(active_nums),
        "categorical_features": len(active_cats),
        "blend_weights": weights_dict,
        "selected_strategy": {
            "name": selected_strategy_name,
        },
        "oof_scores": {
            "raw_blend": {"comp": raw_comp, "auc": raw_auc, "ll": raw_ll},
            "calibrated": {"comp": cal_comp, "auc": cal_auc, "ll": cal_ll},
            "final": {"comp": best_comp, "auc": best_auc, "ll": best_ll},
        },
        "submission_path": str(sub_path),
    }

    with open(os.path.join(output_dir, "run_summary_baseline.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(
        {"selected_strategy": selected_strategy_name, "metadata": metadata, "calibrator": calibrator, "weights": weights_dict},
        os.path.join(models_dir, "ensemble_baseline.joblib"),
    )

    return best_comp, final_oof, final_test


if __name__ == "__main__":
    run_baseline()

