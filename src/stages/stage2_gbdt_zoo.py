"""
Stage 2: Independent domain GBDT zoo across 5 specialized tree architectures.
Stage 1 predictions are excluded from Stage 2 optimization and blending. Stage 1 and Stage 2 predictions meet downstream at Stage 5.

Modernized with:
1. feature_engine selection (DropConstant + DropDuplicate + SmartCorrelatedSelection).
2. Domain feature views: dom2_cols (Physics+Digital), dom3_cols (Physics+Digital+Momentum), dom_triage_cols.
3. 5 specialized architectures:
   - cb_d7_dom26: CatBoost GPU Depth 7 on dom2_cols
   - cb_d6_dom35: CatBoost GPU Depth 6 on dom3_cols
   - xgb_d4_dom35: XGBoost GPU Depth 4 on dom3_cols
   - xgb_d4_triage: XGBoost GPU Depth 4 on dom_triage_cols
   - lgb_extra: LightGBM ExtraTrees with randomized feature sub-sampling
4. Independent domain learning: pure domain models, no Stage 1 anchor in blend.
5. Metric-direct SLSQP blend / HillClimbing directly maximizing exact competition composite score.
6. Smooth cross-fitted Beta calibration.
7. Saves checkpoints/gbdt_zoo_4seed.npz, oof_stage2_domain.npy, and test_stage2_domain.npy for downstream stages.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize, Bounds

from src.config import (
    ID_COL,
    TARGET,
    SEED,
    N_SPLITS,
    FLOOR,
    CEIL,
    EPS,
    HAS_GPU,
    get_default_dataset_paths,
)
from src.features.pipeline import engineer_features
from src.features.encoding import screen_features, extract_domain_feature_subsets
from src.ensemble.calibration import beta_calibrate
from src.metrics import competition_score


def train_single_model_fold(
    model_name: str,
    cls_: Any,
    params: Dict[str, Any],
    cols: List[str],
    cat_cols: List[str],
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fits one model architecture on one fold for one seed."""
    sub_cols = [c for c in cols if c in x_tr.columns]
    sub_cats = [c for c in cat_cols if c in sub_cols]
    x_tr_sub = x_tr[sub_cols].copy()
    x_va_sub = x_va[sub_cols].copy()
    x_te_sub = x_te[sub_cols].copy()

    p = dict(params)
    for c in sub_cats:
        x_tr_sub[c] = x_tr_sub[c].astype("category")
        x_va_sub[c] = x_va_sub[c].astype("category")
        x_te_sub[c] = x_te_sub[c].astype("category")

    if cls_ == CatBoostClassifier:
        cat_indices = [x_tr_sub.columns.get_loc(c) for c in sub_cats]
        p["task_type"] = "GPU" if HAS_GPU else "CPU"
        p["random_seed"] = seed
        p["verbose"] = False
        p["eval_metric"] = "Logloss"
        p["loss_function"] = "Logloss"
        model = CatBoostClassifier(**p)
        model.fit(
            x_tr_sub, y_tr,
            eval_set=(x_va_sub, y_va),
            cat_features=cat_indices if cat_indices else None,
            early_stopping_rounds=40,
            verbose=False,
        )
        val_prob = model.predict_proba(x_va_sub)[:, 1]
        test_prob = model.predict_proba(x_te_sub)[:, 1]

    elif cls_ == XGBClassifier:
        p["device"] = "cuda" if HAS_GPU else "cpu"
        p["tree_method"] = "hist"
        p["enable_categorical"] = True
        p["random_state"] = seed
        p["eval_metric"] = "logloss"
        model = XGBClassifier(**p)
        model.fit(
            x_tr_sub, y_tr,
            eval_set=[(x_va_sub, y_va)],
            verbose=False,
        )
        val_prob = model.predict_proba(x_va_sub)[:, 1]
        test_prob = model.predict_proba(x_te_sub)[:, 1]

    elif cls_ == lgb.LGBMClassifier:
        lgb_tr = lgb.Dataset(x_tr_sub, label=y_tr, free_raw_data=False)
        lgb_va = lgb.Dataset(x_va_sub, label=y_va, reference=lgb_tr, free_raw_data=False)
        lgb_p = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": p.get("num_leaves", 45),
            "learning_rate": p.get("learning_rate", 0.030),
            "min_child_samples": p.get("min_child_samples", 60),
            "colsample_bytree": p.get("colsample_bytree", 0.60),
            "subsample": p.get("subsample", 0.75),
            "subsample_freq": p.get("subsample_freq", 1),
            "reg_lambda": p.get("reg_lambda", 5.0),
            "extra_trees": p.get("extra_trees", True),
            "seed": seed,
            "verbose": -1,
            "n_jobs": -1,
        }
        model = lgb.train(
            lgb_p,
            lgb_tr,
            num_boost_round=p.get("n_estimators", 600),
            valid_sets=[lgb_va],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        val_prob = model.predict(x_va_sub)
        test_prob = model.predict(x_te_sub)
    else:
        raise ValueError(f"Unknown classifier type: {cls_}")

    return np.clip(val_prob, FLOOR, CEIL), np.clip(test_prob, FLOOR, CEIL)


def optimize_composite_weights(oof_dict: Dict[str, np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """Finds optimal weights maximizing exact competition composite score."""
    names = list(oof_dict.keys())
    P_mat = np.column_stack([oof_dict[n] for n in names])
    n_models = len(names)

    # Initialize equal weights
    init_w = np.ones(n_models) / n_models

    def loss_func(w):
        w = np.clip(w, 0.0, 1.0)
        w = w / (w.sum() + EPS)
        p = np.clip(P_mat @ w, FLOOR, CEIL)
        _, _, comp = competition_score(y_true, p)
        return -comp

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = Bounds(0.0, 1.0)
    res = minimize(loss_func, x0=init_w, method="SLSQP", bounds=bounds, constraints=constraints)

    weights = res.x if res.success else init_w
    weights = np.clip(weights, 0.0, 1.0)
    weights /= (weights.sum() + EPS)
    return weights


def run_stage2(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    seeds: Sequence[int] = (42, 2026),
    n_splits: int = 10,
    k_top_features: int = 60,
    include_anchor: bool = False,
) -> float:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_raw.columns else "Target"
    y_true = train_raw[target_col].to_numpy(int)

    print("Stage 2: 10-fold domain GBDT zoo (5 architectures, independent_domain_learning=True)", flush=True)
    print("Independent Domain Learning: Stage 2 operating without Stage 1 anchor.", flush=True)

    # 2. Modern Feature Engineering & Selection
    print("\nStep 1: Engineering comprehensive features & filtering via feature_engine", flush=True)
    X_train_feat, X_test_feat, cat_cols, _ = engineer_features(train_raw, test_raw)

    drop_meta = [c for c in [ID_COL, target_col] if c in X_train_feat.columns]
    X_train_clean = X_train_feat.drop(columns=drop_meta)
    X_test_clean = X_test_feat.drop(columns=[c for c in [ID_COL] if c in X_test_feat.columns])

    # 3. Define the 5 Zoo Architectures with column subset functions
    architectures_def = [
        ("cb_d7_dom26", CatBoostClassifier, {"depth": 7, "learning_rate": 0.038, "l2_leaf_reg": 20.0, "iterations": 650}, lambda d2, d3, dt, sel: d2),
        ("cb_d6_dom35", CatBoostClassifier, {"depth": 6, "learning_rate": 0.038, "l2_leaf_reg": 10.0, "iterations": 650}, lambda d2, d3, dt, sel: d3),
        ("xgb_d4_dom35", XGBClassifier, {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5.0, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 2.0, "gamma": 1.5, "n_estimators": 600}, lambda d2, d3, dt, sel: d3),
        ("xgb_d4_triage", XGBClassifier, {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5.0, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 2.0, "gamma": 1.5, "n_estimators": 600}, lambda d2, d3, dt, sel: dt),
        ("lgb_extra", lgb.LGBMClassifier, {"num_leaves": 45, "learning_rate": 0.030, "min_child_samples": 60, "colsample_bytree": 0.60, "subsample": 0.75, "subsample_freq": 1, "reg_lambda": 5.0, "n_estimators": 600, "extra_trees": True}, lambda d2, d3, dt, sel: list(sel)),
    ]

    zoo_oof: Dict[str, np.ndarray] = {name: np.zeros(len(train_raw)) for name, _, _, _ in architectures_def}
    zoo_test: Dict[str, np.ndarray] = {name: np.zeros(len(test_raw)) for name, _, _, _ in architectures_def}

    # 4. Cross-Validation Loop: Folds Outer -> Models Inner (Leak-Free Nested Selection)
    for s_idx, seed_val in enumerate(seeds, start=1):
        print(f"Running cross-validation seed {seed_val} ({s_idx}/{len(seeds)})", flush=True)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_val)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_clean, y_true), start=1):
            t_fold = time.time()
            x_tr_raw = X_train_clean.iloc[trn_idx].copy()
            y_tr = y_true[trn_idx]
            x_va_raw = X_train_clean.iloc[val_idx].copy()
            y_va = y_true[val_idx]
            x_te_raw = X_test_clean.copy()

            # Fold-isolated feature screening strictly on training partition
            selected_cols = screen_features(
                x_tr_raw, y_tr, cat_cols=cat_cols, k_top=k_top_features, seed=seed_val
            )
            active_cats = [c for c in cat_cols if c in selected_cols]
            dom2_cols, dom3_cols, dom_triage_cols = extract_domain_feature_subsets(selected_cols)

            # Filter fold slices
            x_tr = x_tr_raw[selected_cols].copy()
            x_va = x_va_raw[selected_cols].copy()
            x_te = x_te_raw[[c for c in selected_cols if c in x_te_raw.columns]].copy()

            # Train all 5 architectures on this fold's prepared data
            for name, cls_, params, col_fn in architectures_def:
                arch_cols = col_fn(dom2_cols, dom3_cols, dom_triage_cols, selected_cols)
                arch_cats = [c for c in active_cats if c in arch_cols]

                val_p, test_p = train_single_model_fold(
                    model_name=name,
                    cls_=cls_,
                    params=params,
                    cols=arch_cols,
                    cat_cols=arch_cats,
                    x_tr=x_tr,
                    y_tr=y_tr,
                    x_va=x_va,
                    y_va=y_va,
                    x_te=x_te,
                    seed=seed_val + fold,
                )
                zoo_oof[name][val_idx] += val_p / len(seeds)
                zoo_test[name] += test_p / (n_splits * len(seeds))

    for name, _, _, _ in architectures_def:
        ll, auc, comp = competition_score(y_true, zoo_oof[name])
        print(f"  {name} OOF: Comp={comp:.5f} | AUC={auc:.5f} | LL={ll:.5f}", flush=True)

    # 6. Ensemble Optimization (Independent Domain Zoo)
    print("\nOptimizing domain zoo ensemble weights", flush=True)

    blend_candidates_oof = dict(zoo_oof)
    blend_candidates_test = dict(zoo_test)

    assert "oof_anchor" not in blend_candidates_oof, "Stage 2 domain zoo must not include Stage 1 anchor."

    candidate_names = list(blend_candidates_oof.keys())
    weights = optimize_composite_weights(blend_candidates_oof, y_true)

    print("Stage 2 learned model weights:", flush=True)
    for n, w in zip(candidate_names, weights):
        print(f"  {n:<16}: {w:.4f}", flush=True)

    P_mat = np.column_stack([blend_candidates_oof[n] for n in candidate_names])
    T_mat = np.column_stack([blend_candidates_test[n] for n in candidate_names])

    raw_oof = np.clip(P_mat @ weights, FLOOR, CEIL)
    raw_test = np.clip(T_mat @ weights, FLOOR, CEIL)

    r_ll, r_auc, r_comp = competition_score(y_true, raw_oof)
    print(f"Raw zoo blend OOF: Comp={r_comp:.5f} | AUC={r_auc:.5f} | LL={r_ll:.5f}", flush=True)

    # 7. Smooth Beta Calibration
    print("Cross-fitted Beta Calibration on zoo blend", flush=True)
    cal_oof, cal_test, _ = beta_calibrate(raw_oof, raw_test, y_true, n_splits=5, seed=SEED)
    c_ll, c_auc, c_comp = competition_score(y_true, cal_oof)
    print(f"Beta calibrated zoo OOF: Comp={c_comp:.5f} | AUC={c_auc:.5f} | LL={c_ll:.5f}", flush=True)

    if c_comp >= r_comp:
        final_oof, final_test = cal_oof, cal_test
        selected_name = "beta_calibrated_zoo"
        print(f"Adopted calibrated domain zoo: Comp={c_comp:.5f}", flush=True)
    else:
        final_oof, final_test = raw_oof, raw_test
        selected_name = "raw_zoo_blend"
        print(f"Adopted raw domain zoo: Comp={r_comp:.5f}", flush=True)

    f_ll, f_auc, final_comp = competition_score(y_true, final_oof)
    print(f"Final Stage 2 score: Comp={final_comp:.5f} | AUC={f_auc:.5f} | LL={f_ll:.5f}", flush=True)

    # 8. Save Deliverables for Downstream Stages (Stage 3 & 4)
    save_payload = {}
    for k in zoo_oof:
        save_payload[f"oof_{k}"] = zoo_oof[k]
        save_payload[f"test_{k}"] = zoo_test[k]

    s2_path1 = os.path.join(output_dir, "gbdt_zoo_4seed.npz")
    s2_path2 = os.path.join(output_dir, "gbdt_balanced_dom1_p2_seeds4.npz")
    np.savez_compressed(s2_path1, oof_s4_tree_zoo=raw_oof, test_s4_tree_zoo=raw_test, oof_calibrated=final_oof, test_calibrated=final_test, **save_payload)
    np.savez_compressed(s2_path2, **save_payload)

    # Save dedicated Stage 2 independent domain predictions
    np.save(os.path.join(output_dir, "oof_stage2_domain.npy"), raw_oof)
    np.save(os.path.join(output_dir, "test_stage2_domain.npy"), raw_test)

    sub_path = os.path.join(sub_dir, "submission_s2_gbdt_zoo.csv")
    sub_df = pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": np.clip(final_test, FLOOR, CEIL)})
    sub_df.to_csv(sub_path, index=False)

    metadata = {
        "stage": "stage2_domain_gbdt_zoo",
        "final_comp": final_comp,
        "final_auc": f_auc,
        "final_logloss": f_ll,
        "weights": {n: float(w) for n, w in zip(candidate_names, weights)},
        "strategy": selected_name,
        "zoo_models": list(zoo_oof.keys()),
        "submission_path": str(sub_path),
    }

    with open(os.path.join(output_dir, "run_summary_stage2.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {s2_path1}, {s2_path2}, and {sub_path}", flush=True)
    return final_comp


if __name__ == "__main__":
    run_stage2()
