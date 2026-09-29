"""
Stage 2: 4-Seed 10-Fold Domain GBDT Zoo across 5 tree architectures.
Generates checkpoints/gbdt_zoo_4seed.npz and submission_s2_gbdt_zoo_0.73733.csv.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from src.config import (
    ID_COL,
    TARGET,
    get_default_dataset_paths,
)
from src.features.pipeline import engineer_features
from src.features.encoding import extract_domain_feature_subsets, screen_features
from src.models.gbdt import fit_gbdt_runner
from src.ensemble.calibration import platt_scaling_calibrate
from src.metrics import competition_score


def run_stage2(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    seeds: Tuple[int, ...] = (42, 100, 2024, 777),
    n_splits: int = 10,
) -> float:
    """Executes Stage 2 4-Seed 10-Fold Domain GBDT Zoo."""
    print("Stage 2: 4-Seed 10-Fold Domain GBDT Zoo")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    X_train_feat, X_test_feat, cat_cols, _ = engineer_features(train_raw, test_raw)
    target_col = TARGET if TARGET in X_train_feat.columns else "Target"
    y_true = X_train_feat[target_col].to_numpy(int)

    drop_meta = [c for c in [ID_COL, target_col] if c in X_train_feat.columns]
    X_train_clean = X_train_feat.drop(columns=drop_meta)
    X_test_clean = X_test_feat.drop(columns=[c for c in [ID_COL] if c in X_test_feat.columns])

    print("Screening top 300 features on numeric signals", flush=True)
    selected_cols = screen_features(X_train_clean, y_true, cat_cols, k_top=300, seed=42)
    X_tr = X_train_clean[selected_cols].copy()
    X_te = X_test_clean[selected_cols].copy()
    active_cats = [c for c in cat_cols if c in selected_cols]

    dom2_cols, dom3_cols, dom_triage_cols = extract_domain_feature_subsets(list(X_tr.columns))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X_tr, y_true))

    architectures = [
        ("cb_d7_dom26", CatBoostClassifier, {"depth": 7, "learning_rate": 0.038, "l2_leaf_reg": 20.0, "iterations": 600, "feature_cols": dom2_cols}),
        ("cb_d6_dom35", CatBoostClassifier, {"depth": 6, "learning_rate": 0.038, "l2_leaf_reg": 10.0, "iterations": 600, "feature_cols": dom3_cols}),
        ("xgb_d4_dom35", XGBClassifier, {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5.0, "max_delta_step": 5.0, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 2.0, "gamma": 1.5, "n_estimators": 600, "feature_cols": dom3_cols}),
        ("xgb_d4_triage", XGBClassifier, {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5.0, "max_delta_step": 5.0, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 2.0, "gamma": 1.5, "n_estimators": 600, "feature_cols": dom_triage_cols}),
        ("lgb_extra", lgb.LGBMClassifier, {"num_leaves": 45, "learning_rate": 0.030, "min_child_samples": 100, "colsample_bytree": 0.60, "subsample": 0.75, "subsample_freq": 1, "reg_lambda": 5.0, "n_estimators": 600}),
    ]

    zoo_oof: Dict[str, np.ndarray] = {}
    zoo_test: Dict[str, np.ndarray] = {}
    fold_cache: Dict[int, Any] = {}

    for name, cls_, params in architectures:
        oof, test_pred, _ = fit_gbdt_runner(
            name=name,
            model_cls=cls_,
            base_params=params,
            X_train=X_tr,
            y_train=y_true,
            X_test=X_te,
            folds=folds,
            cat_cols=active_cats,
            seeds=seeds,
            fold_data_cache=fold_cache,
        )
        zoo_oof[name] = oof
        zoo_test[name] = test_pred

    # Optimal weight optimization to strictly minimize LogLoss
    from scipy.optimize import minimize
    names = list(zoo_oof.keys())
    init_w = np.ones(len(names)) / len(names)

    def logloss_obj(w):
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-7)
        blend = sum(w[i] * zoo_oof[n] for i, n in enumerate(names))
        return competition_score(y_true, blend)[0]  # return logloss

    res = minimize(logloss_obj, init_w, method="SLSQP", bounds=[(0.0, 1.0)] * len(names), constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    weights = res.x if res.success else init_w
    weights = np.clip(weights, 0, 1)
    weights /= weights.sum()

    print("Stage 2 Learned Model Weights (LogLoss Minimization):")
    for n, w in zip(names, weights):
        print(f"  {n:<16}: {w:.4f}", flush=True)

    raw_oof = sum(weights[i] * zoo_oof[n] for i, n in enumerate(names))
    raw_test = sum(weights[i] * zoo_test[n] for i, n in enumerate(names))
    cal_oof, cal_test, _ = platt_scaling_calibrate(raw_oof, raw_test, y_true)

    ll, auc, final_comp = competition_score(y_true, cal_oof)
    print(f"Stage 2 Calibrated Ensemble: Score: {final_comp:.5f} | AUC: {auc:.5f} | LogLoss: {ll:.5f}")

    s4_path = os.path.join(output_dir, "gbdt_zoo_4seed.npz")
    np.savez_compressed(s4_path, oof_s4_tree_zoo=cal_oof, test_s4_tree_zoo=cal_test, **zoo_oof)

    sub_path = os.path.join(sub_dir, "submission_s2_gbdt_zoo_0.73733.csv")
    pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": cal_test}).to_csv(sub_path, index=False)
    print(f"Saved {s4_path} and {sub_path}")
    return final_comp


if __name__ == "__main__":
    run_stage2()
