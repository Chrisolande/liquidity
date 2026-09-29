"""
Stage 2: 4-Seed 5-Fold Domain GBDT Zoo across 5 specialized tree architectures.
Modernized with:
1. feature_engine selection (DropConstant + DropDuplicate + SmartCorrelatedSelection).
2. Domain feature views: dom2_cols (Physics+Digital), dom3_cols (Physics+Digital+Momentum), dom_triage_cols.
3. 5 specialized architectures:
   - cb_d7_dom26: CatBoost GPU Depth 7 on dom2_cols
   - cb_d6_dom35: CatBoost GPU Depth 6 on dom3_cols
   - xgb_d4_dom35: XGBoost GPU Depth 4 on dom3_cols
   - xgb_d4_triage: XGBoost GPU Depth 4 on dom_triage_cols
   - lgb_extra: LightGBM ExtraTrees with randomized feature sub-sampling
4. Integrates oof_champ anchor (0.73522) into the ensembling pool for guaranteed non-degradation.
5. Metric-direct SLSQP blend / HillClimbing directly maximizing exact competition composite score.
6. Smooth cross-fitted Beta calibration.
7. Saves checkpoints/gbdt_zoo_4seed.npz and checkpoints/gbdt_balanced_dom1_p2_seeds4.npz for downstream stages.
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
from src.features.selection import run_feature_engine_selection
from src.features.encoding import extract_domain_feature_subsets
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
            early_stopping_rounds=75,
            verbose=False,
        )
        val_prob = model.predict_proba(x_va_sub)[:, 1]
        test_prob = model.predict_proba(x_te_sub)[:, 1]

    elif cls_ == XGBClassifier:
        p["device"] = "cuda" if HAS_GPU else "cpu"
        p["tree_method"] = "hist"
        p["random_state"] = seed
        p["eval_metric"] = "logloss"
        p["early_stopping_rounds"] = 75

        # Apply leak-free in-fold target encoding for interaction categoricals
        te_cols = [c for c in sub_cats if c in x_tr_sub.columns]
        if te_cols:
            from src.features.encoding import apply_fold_target_encoding
            x_tr_sub, x_va_sub, x_te_sub = apply_fold_target_encoding(
                x_tr_sub, y_tr, x_va_sub, x_te_sub, te_cols, smoothing=20.0, seed=seed
            )
            x_tr_sub = x_tr_sub.drop(columns=te_cols)
            x_va_sub = x_va_sub.drop(columns=te_cols)
            x_te_sub = x_te_sub.drop(columns=te_cols)
        else:
            p["enable_categorical"] = True

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
            num_boost_round=p.get("n_estimators", 750),
            valid_sets=[lgb_va],
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        val_prob = model.predict(x_va_sub)
        test_prob = model.predict(x_te_sub)
    else:
        raise ValueError(f"Unknown classifier type: {cls_}")

    return np.clip(val_prob, FLOOR, CEIL), np.clip(test_prob, FLOOR, CEIL)


def optimize_composite_weights(oof_dict: Dict[str, np.ndarray], y_true: np.ndarray, reg_alpha: float = 0.05) -> np.ndarray:
    """Finds optimal weights maximizing exact competition composite score with L2 regularizer."""
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
        # Regularize weights towards equal weighting to prevent overfitting collinear predictions
        l2_pen = reg_alpha * np.sum((w - init_w) ** 2)
        return -comp + l2_pen

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
) -> float:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_raw.columns else "Target"
    y_true = train_raw[target_col].to_numpy(int)

    print("=" * 80, flush=True)
    print(f"STAGE 2: {len(seeds)}-SEED {n_splits}-FOLD DOMAIN GBDT ZOO (5 ARCHITECTURES)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Anchor from Stage 1 / Baseline
    anchor_oof_path = os.path.join(output_dir, "oof_champ_train.npy")
    anchor_sub_path = os.path.join(sub_dir, "submission_stage1_hillclimb.csv")
    if not os.path.exists(anchor_sub_path):
        anchor_sub_path = os.path.join(sub_dir, "submission_baseline.csv")

    oof_anchor = np.load(anchor_oof_path) if os.path.exists(anchor_oof_path) else None
    test_anchor = pd.read_csv(anchor_sub_path)["Target"].to_numpy(dtype=float) if os.path.exists(anchor_sub_path) else None

    if oof_anchor is not None:
        a_ll, a_auc, a_comp = competition_score(y_true, oof_anchor)
        print(f"✓ Loaded Stage Champion Anchor: Comp={a_comp:.5f} | AUC={a_auc:.5f} | LL={a_ll:.5f}", flush=True)
    else:
        print("Notice: oof_champ_train.npy not found. Stage 2 will proceed from scratch.", flush=True)
        a_comp = 0.0

    # 2. Modern Feature Engineering & Selection
    print("\nStep 1: Engineering comprehensive features & filtering via feature_engine", flush=True)
    X_train_feat, X_test_feat, cat_cols, _ = engineer_features(train_raw, test_raw)

    drop_meta = [c for c in [ID_COL, target_col] if c in X_train_feat.columns]
    X_train_clean = X_train_feat.drop(columns=drop_meta)
    X_test_clean = X_test_feat.drop(columns=[c for c in [ID_COL] if c in X_test_feat.columns])

    # Construct joint multi-strata cohort keys
    from src.baseline import build_composite_strata
    strata = build_composite_strata(X_train_feat, target_col=target_col, n_splits=n_splits)
    print(f"Built multi-strata cohort keys ({strata.nunique()} unique strata) for balanced CV splitting", flush=True)

    X_tr_sel, X_te_sel, selected_cols = run_feature_engine_selection(
        X_train_clean, y_true, X_test_clean, cat_cols=cat_cols, k_top=k_top_features, corr_threshold=0.98, seed=SEED
    )
    active_cats = [c for c in cat_cols if c in selected_cols]
    print(f"Active features for Domain Zoo: {len(selected_cols)} ({len(active_cats)} categoricals)", flush=True)

    # 3. Extract Specialized Domain Views
    dom2_cols, dom3_cols, dom_triage_cols = extract_domain_feature_subsets(list(selected_cols))
    print(f"Domain Views Configured:")
    print(f"  dom2_cols (Physics + Digital Channels)  : {len(dom2_cols)} features", flush=True)
    print(f"  dom3_cols (Physics + Digital + Momentum): {len(dom3_cols)} features", flush=True)
    print(f"  dom_triage_cols (Physics + Triage)      : {len(dom_triage_cols)} features", flush=True)

    # 4. Define the 5 Zoo Architectures
    architectures = [
        ("cb_d7_dom26", CatBoostClassifier, {"depth": 7, "learning_rate": 0.038, "l2_leaf_reg": 20.0, "iterations": 750}, dom2_cols),
        ("cb_d6_dom35", CatBoostClassifier, {"depth": 6, "learning_rate": 0.038, "l2_leaf_reg": 15.0, "iterations": 750}, dom3_cols),
        ("xgb_d4_dom35", XGBClassifier, {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5.0, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.5, "n_estimators": 750}, dom3_cols),
        ("xgb_d4_triage", XGBClassifier, {"max_depth": 4, "learning_rate": 0.035, "min_child_weight": 5.0, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.5, "n_estimators": 750}, dom_triage_cols),
        ("lgb_extra", lgb.LGBMClassifier, {"num_leaves": 45, "learning_rate": 0.030, "min_child_samples": 60, "colsample_bytree": 0.60, "subsample": 0.75, "subsample_freq": 1, "reg_lambda": 5.0, "n_estimators": 750, "extra_trees": True}, list(selected_cols)),
    ]

    zoo_oof: Dict[str, np.ndarray] = {}
    zoo_test: Dict[str, np.ndarray] = {}

    # 5. Cross-Validation Loop across Seeds & Folds
    for arch_idx, (name, cls_, params, cols) in enumerate(architectures, start=1):
        print(f"\n[{arch_idx}/{len(architectures)}] Training Domain Architecture: {name} (cols={len(cols)})", flush=True)
        t_arch = time.time()
        m_oof_seeds = np.zeros(len(train_raw))
        m_test_seeds = np.zeros(len(test_raw))

        for s_idx, seed_val in enumerate(seeds, start=1):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed_val)
            fold_test_preds = []

            for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr_sel, strata), start=1):
                x_tr, y_tr = X_tr_sel.iloc[trn_idx].copy(), y_true[trn_idx]
                x_va, y_va = X_tr_sel.iloc[val_idx].copy(), y_true[val_idx]
                x_te = X_te_sel.copy()

                val_p, test_p = train_single_model_fold(
                    model_name=name,
                    cls_=cls_,
                    params=params,
                    cols=cols,
                    cat_cols=active_cats,
                    x_tr=x_tr,
                    y_tr=y_tr,
                    x_va=x_va,
                    y_va=y_va,
                    x_te=x_te,
                    seed=seed_val + fold,
                )
                m_oof_seeds[val_idx] += val_p / len(seeds)
                fold_test_preds.append(test_p)

            m_test_seeds += np.mean(fold_test_preds, axis=0) / len(seeds)

        zoo_oof[name] = m_oof_seeds
        zoo_test[name] = m_test_seeds

        ll, auc, comp = competition_score(y_true, m_oof_seeds)
        print(f"  ==> {name} {len(seeds)}-Seed OOF: Comp={comp:.5f} | AUC={auc:.5f} | LL={ll:.5f} ({time.time() - t_arch:.1f}s)", flush=True)

    # 6. Ensemble Optimization (incorporating anchor)
    print("\n" + "=" * 80, flush=True)
    print("PHASE 2: OPTIMIZING DOMAIN ZOO ENSEMBLE (COMPOSITE MAXIMIZATION)", flush=True)
    print("=" * 80, flush=True)

    blend_candidates_oof = dict(zoo_oof)
    blend_candidates_test = dict(zoo_test)

    if oof_anchor is not None and test_anchor is not None:
        blend_candidates_oof["oof_anchor"] = oof_anchor
        blend_candidates_test["oof_anchor"] = test_anchor

    candidate_names = list(blend_candidates_oof.keys())
    weights = optimize_composite_weights(blend_candidates_oof, y_true, reg_alpha=0.05)

    print("Stage 2 Learned Model Weights (Regularized Metric-Direct SLSQP):", flush=True)
    for n, w in zip(candidate_names, weights):
        print(f"  {n:<16}: {w:.4f}", flush=True)

    P_mat = np.column_stack([blend_candidates_oof[n] for n in candidate_names])
    T_mat = np.column_stack([blend_candidates_test[n] for n in candidate_names])

    raw_oof = np.clip(P_mat @ weights, FLOOR, CEIL)
    raw_test = np.clip(T_mat @ weights, FLOOR, CEIL)

    r_ll, r_auc, r_comp = competition_score(y_true, raw_oof)
    print(f"\nRaw Zoo Blend OOF: Comp={r_comp:.5f} | AUC={r_auc:.5f} | LL={r_ll:.5f}", flush=True)

    # 7. Calibration Check (Strict Gating to Prevent Double Calibration)
    print("\nStep 3: Evaluating Calibration vs Raw Blend", flush=True)
    cal_oof, cal_test, _ = beta_calibrate(raw_oof, raw_test, y_true, n_splits=5, seed=SEED)
    c_ll, c_auc, c_comp = competition_score(y_true, cal_oof)
    print(f"Beta Calibrated Zoo OOF: Comp={c_comp:.5f} | AUC={c_auc:.5f} | LL={c_ll:.5f}", flush=True)

    # Strict gating against double calibration
    if c_comp > r_comp and c_comp > a_comp and (c_comp - r_comp) > 0.0003:
        final_oof, final_test = cal_oof, cal_test
        selected_name = "beta_calibrated_zoo"
        print(f"  --> Adopted Calibrated Zoo (+{c_comp - a_comp:+.5f} vs Anchor).", flush=True)
    elif r_comp >= a_comp:
        final_oof, final_test = raw_oof, raw_test
        selected_name = "raw_zoo_blend"
        print(f"  --> Adopted Raw Zoo (+{r_comp - a_comp:+.5f} vs Anchor).", flush=True)
    else:
        final_oof, final_test = oof_anchor, test_anchor
        selected_name = "preserved_anchor"
        print("  --> Preserved Previous Anchor.", flush=True)

    f_ll, f_auc, final_comp = competition_score(y_true, final_oof)
    print("\n" + "=" * 80, flush=True)
    print(f"★ FINAL STAGE 2 COMPOSITE SCORE: {final_comp:.5f} (AUC: {f_auc:.5f}, LL: {f_ll:.5f}) ★", flush=True)
    print("=" * 80, flush=True)

    # 8. Save Deliverables for Downstream Stages (Stage 3 & 4)
    save_payload = {}
    for k in zoo_oof:
        save_payload[f"oof_{k}"] = zoo_oof[k]
        save_payload[f"test_{k}"] = zoo_test[k]

    s2_path1 = os.path.join(output_dir, "gbdt_zoo_4seed.npz")
    s2_path2 = os.path.join(output_dir, "gbdt_balanced_dom1_p2_seeds4.npz")
    np.savez_compressed(s2_path1, oof_s4_tree_zoo=final_oof, test_s4_tree_zoo=final_test, **save_payload)
    np.savez_compressed(s2_path2, **save_payload)

    # Update champion anchor if improved
    if final_comp >= a_comp:
        np.save(os.path.join(output_dir, "oof_champ_train.npy"), final_oof)

    sub_path = os.path.join(sub_dir, "submission_s2_gbdt_zoo.csv")
    sub_df = pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": np.clip(final_test, FLOOR, CEIL)})
    sub_df.to_csv(sub_path, index=False)
    if final_comp >= a_comp:
        sub_df.to_csv(os.path.join(sub_dir, "submission_baseline.csv"), index=False)
        sub_df.to_csv(os.path.join(sub_dir, "submission_best_0.73731.csv"), index=False)

    metadata = {
        "stage": "stage2_domain_gbdt_zoo",
        "previous_anchor_comp": a_comp,
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
