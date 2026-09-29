"""
Optuna Bayesian Hyperparameter Sweep for Stage 1 Base Models.
Tunes CatBoost, LightGBM, and XGBoost on the exact 79 curated features
maximizing the exact competition composite score with multi-strata stratification.
Outputs progress to optuna_sweep.log and saves checkpoints/optuna_best_stage1.json.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from src.config import (
    ID_COL,
    TARGET,
    FLOOR,
    CEIL,
    SEED,
    get_default_dataset_paths,
)
from src.metrics import competition_score
from src.features.pipeline import engineer_features
from src.features.selection import run_feature_engine_selection
from src.baseline import build_composite_strata

# Setup logging
log_path = "optuna_sweep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("optuna_stage1")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_sweep(n_trials_per_model: int = 15, n_splits: int = 5):
    logger.info("=" * 70)
    logger.info("STARTING STAGE 1 OPTUNA BAYESIAN HYPERPARAMETER SWEEP")
    logger.info(f"Trials per model: {n_trials_per_model} | CV folds: {n_splits}")
    logger.info("=" * 70)

    train_path, test_path = get_default_dataset_paths()
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_raw.columns else "Target"
    y_true = train_raw[target_col].to_numpy(int)

    # 1. Feature Engineering & Selection (uses cache if available)
    logger.info("Loading / engineering feature set...")
    train_fe, test_fe, cat_cols, _ = engineer_features(train_raw, test_raw)
    X_raw = train_fe.drop(columns=[target_col, ID_COL])
    X_test_raw = test_fe.drop(columns=[c for c in [ID_COL] if c in test_fe.columns])

    logger.info("Building multi-strata cohort keys...")
    strata = build_composite_strata(train_fe, target_col=target_col, n_splits=n_splits)

    logger.info("Filtering features via feature_engine (variance-based)...")
    X_sel, _, selected_cols = run_feature_engine_selection(
        X_raw, y_true, X_test_raw, cat_cols=cat_cols, k_top=60, corr_threshold=0.98, seed=SEED
    )
    active_cats = [c for c in cat_cols if c in selected_cols]
    active_nums = [c for c in selected_cols if c not in active_cats]
    logger.info(f"Active features: {len(selected_cols)} ({len(active_nums)} numeric, {len(active_cats)} categoricals)")

    # Ensure categoricals are category dtype
    for c in active_cats:
        X_sel[c] = X_sel[c].astype("category")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_sel, strata))

    best_results: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. TUNE CATBOOST
    # -------------------------------------------------------------
    logger.info("\n>>> [1/3] Tuning CatBoost (GPU) <<<")
    cat_idx = [X_sel.columns.get_loc(c) for c in active_cats]

    def objective_cb(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 6, 8)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 10.0, 30.0, step=2.0)
        learning_rate = trial.suggest_float("learning_rate", 0.025, 0.045, step=0.005)
        random_strength = trial.suggest_float("random_strength", 0.5, 2.0, step=0.25)
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 0.5, step=0.1)

        oof_preds = np.zeros(len(y_true))
        for trn_idx, val_idx in splits:
            x_tr, y_tr = X_sel.iloc[trn_idx], y_true[trn_idx]
            x_va, y_va = X_sel.iloc[val_idx], y_true[val_idx]

            cb = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                iterations=850,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_strength=random_strength,
                bagging_temperature=bagging_temperature,
                border_count=128,
                task_type="GPU",
                random_seed=SEED,
                verbose=False,
            )
            cb.fit(
                x_tr, y_tr,
                eval_set=(x_va, y_va),
                cat_features=cat_idx,
                early_stopping_rounds=40,
                verbose=False,
            )
            oof_preds[val_idx] = cb.predict_proba(x_va)[:, 1]

        ll, auc, comp = competition_score(y_true, oof_preds)
        trial.set_user_attr("auc", float(auc))
        trial.set_user_attr("logloss", float(ll))
        logger.info(f"  [CB Trial {trial.number:2d}] Comp: {comp:.5f} | AUC: {auc:.5f} | LL: {ll:.5f} | d={depth} l2={l2_leaf_reg:.1f} lr={learning_rate:.3f}")
        return float(comp)

    study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cb.optimize(objective_cb, n_trials=n_trials_per_model, timeout=450)
    best_results["catboost"] = {
        "best_comp": study_cb.best_value,
        "best_params": study_cb.best_params,
        "auc": study_cb.best_trial.user_attrs.get("auc"),
        "logloss": study_cb.best_trial.user_attrs.get("logloss"),
    }
    logger.info(f"✓ CatBoost Best Comp: {study_cb.best_value:.5f} with params: {study_cb.best_params}")

    # -------------------------------------------------------------
    # 2. TUNE LIGHTGBM
    # -------------------------------------------------------------
    logger.info("\n>>> [2/3] Tuning LightGBM <<<")

    def objective_lgb(trial: optuna.Trial) -> float:
        num_leaves = trial.suggest_int("num_leaves", 35, 65, step=5)
        min_child_samples = trial.suggest_int("min_child_samples", 20, 50, step=5)
        learning_rate = trial.suggest_float("learning_rate", 0.025, 0.040, step=0.005)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.60, 0.85, step=0.05)
        subsample = trial.suggest_float("subsample", 0.70, 0.90, step=0.05)
        reg_lambda = trial.suggest_float("reg_lambda", 1.0, 5.0, step=1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.05, 0.40, step=0.05)

        oof_preds = np.zeros(len(y_true))
        for trn_idx, val_idx in splits:
            x_tr, y_tr = X_sel.iloc[trn_idx], y_true[trn_idx]
            x_va, y_va = X_sel.iloc[val_idx], y_true[val_idx]

            lgb_model = lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                colsample_bytree=colsample_bytree,
                subsample=subsample,
                subsample_freq=1,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(
                x_tr, y_tr,
                eval_set=[(x_va, y_va)],
                eval_metric="logloss",
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            oof_preds[val_idx] = lgb_model.predict_proba(x_va)[:, 1]

        ll, auc, comp = competition_score(y_true, oof_preds)
        trial.set_user_attr("auc", float(auc))
        trial.set_user_attr("logloss", float(ll))
        logger.info(f"  [LGB Trial {trial.number:2d}] Comp: {comp:.5f} | AUC: {auc:.5f} | LL: {ll:.5f} | leaves={num_leaves} min_child={min_child_samples} lr={learning_rate:.3f}")
        return float(comp)

    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(objective_lgb, n_trials=n_trials_per_model, timeout=400)
    best_results["lightgbm"] = {
        "best_comp": study_lgb.best_value,
        "best_params": study_lgb.best_params,
        "auc": study_lgb.best_trial.user_attrs.get("auc"),
        "logloss": study_lgb.best_trial.user_attrs.get("logloss"),
    }
    logger.info(f"✓ LightGBM Best Comp: {study_lgb.best_value:.5f} with params: {study_lgb.best_params}")

    # -------------------------------------------------------------
    # 3. TUNE XGBOOST
    # -------------------------------------------------------------
    logger.info("\n>>> [3/3] Tuning XGBoost (GPU) <<<")

    def objective_xgb(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 4, 6)
        learning_rate = trial.suggest_float("learning_rate", 0.025, 0.040, step=0.005)
        min_child_weight = trial.suggest_float("min_child_weight", 3.0, 8.0, step=1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.65, 0.85, step=0.05)
        subsample = trial.suggest_float("subsample", 0.70, 0.90, step=0.05)
        reg_lambda = trial.suggest_float("reg_lambda", 1.5, 5.0, step=0.5)
        reg_alpha = trial.suggest_float("reg_alpha", 0.05, 0.5, step=0.1)

        oof_preds = np.zeros(len(y_true))
        for trn_idx, val_idx in splits:
            x_tr, y_tr = X_sel.iloc[trn_idx], y_true[trn_idx]
            x_va, y_va = X_sel.iloc[val_idx], y_true[val_idx]

            xgb_model = XGBClassifier(
                n_estimators=800,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                reg_alpha=reg_alpha,
                enable_categorical=True,
                tree_method="hist",
                device="cuda",
                random_state=SEED,
                eval_metric="logloss",
                early_stopping_rounds=40,
            )
            xgb_model.fit(
                x_tr, y_tr,
                eval_set=[(x_va, y_va)],
                verbose=False,
            )
            oof_preds[val_idx] = xgb_model.predict_proba(x_va)[:, 1]

        ll, auc, comp = competition_score(y_true, oof_preds)
        trial.set_user_attr("auc", float(auc))
        trial.set_user_attr("logloss", float(ll))
        logger.info(f"  [XGB Trial {trial.number:2d}] Comp: {comp:.5f} | AUC: {auc:.5f} | LL: {ll:.5f} | d={max_depth} lr={learning_rate:.3f} reg_l={reg_lambda}")
        return float(comp)

    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(objective_xgb, n_trials=n_trials_per_model, timeout=400)
    best_results["xgboost"] = {
        "best_comp": study_xgb.best_value,
        "best_params": study_xgb.best_params,
        "auc": study_xgb.best_trial.user_attrs.get("auc"),
        "logloss": study_xgb.best_trial.user_attrs.get("logloss"),
    }
    logger.info(f"✓ XGBoost Best Comp: {study_xgb.best_value:.5f} with params: {study_xgb.best_params}")

    # Save summary
    os.makedirs("checkpoints", exist_ok=True)
    summary_path = "checkpoints/optuna_best_stage1.json"
    with open(summary_path, "w") as f:
        json.dump(best_results, f, indent=2)
    logger.info(f"\n========================================================")
    logger.info(f"SWEEP COMPLETE! Results saved to {summary_path}")
    logger.info(f"Summary:\n{json.dumps(best_results, indent=2)}")
    logger.info(f"========================================================")


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    run_sweep(n_trials_per_model=trials, n_splits=5)
