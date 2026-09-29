"""
Stage 1: Metric-Direct Stepwise Hill Climbing Ensembling over Baseline Anchor + Diverse GBDTs.
Replicates lawsof(1).py Cell 12:
- Anchor: oof_champ (from baseline 4-seed GBDT ensemble, Comp ~0.73495)
- Candidate 1: CatBoost GPU Tuned Depth 7 (Multi-Seed [42, 2026])
- Candidate 2: LightGBM Leaf-Wise Tuned (num_leaves=45, min_child_samples=40)
- Candidate 3: LightGBM ExtraTrees (extra_trees=True, colsample=0.60)
- Ensembling: hillclimbers.climb_hill maximizing exact competition composite score
- Calibration: Cross-fitted Beta calibration on blended OOF/test
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

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
from src.features.pipeline import engineer_features
from src.features.selection import run_feature_engine_selection
from src.ensemble.calibration import beta_calibrate

try:
    from hillclimbers import climb_hill
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "hillclimbers"])
    from hillclimbers import climb_hill


def comp_metric_eval(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact competition composite score for hillclimbers."""
    _, _, comp = competition_score(y_true, y_pred)
    return float(comp)


def run_tuned_catboost_ms(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    cat_cols: List[str],
    params: Dict,
    seeds: List[int] = (42, 2026),
) -> Tuple[np.ndarray, np.ndarray]:
    cat_idx = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
    val_preds_seeds = []
    test_preds_seeds = []
    task_type = "GPU" if HAS_GPU else "CPU"

    for s in seeds:
        cb = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=params.get("iterations", 900),
            learning_rate=params.get("learning_rate", 0.035),
            depth=params.get("depth", 7),
            l2_leaf_reg=params.get("l2_leaf_reg", 25.0),
            random_strength=params.get("random_strength", 1.0),
            bagging_temperature=params.get("bagging_temperature", 0.3),
            task_type=task_type,
            random_seed=s,
            verbose=False,
        )
        cb.fit(
            X_tr, y_tr,
            eval_set=(X_va, y_va),
            cat_features=cat_idx,
            early_stopping_rounds=40,
            verbose=False,
        )
        val_preds_seeds.append(cb.predict_proba(X_va)[:, 1])
        test_preds_seeds.append(cb.predict_proba(X_te)[:, 1])

    return np.mean(val_preds_seeds, axis=0), np.mean(test_preds_seeds, axis=0)


def run_tuned_lightgbm(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    X_te: pd.DataFrame,
    cat_cols: List[str],
    params: Dict,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    cat_cols_present = [c for c in cat_cols if c in X_tr.columns]
    X_tr_lgb = X_tr.copy()
    X_va_lgb = X_va.copy()
    X_te_lgb = X_te.copy()

    for c in cat_cols_present:
        X_tr_lgb[c] = X_tr_lgb[c].astype("category")
        X_va_lgb[c] = X_va_lgb[c].astype("category")
        X_te_lgb[c] = X_te_lgb[c].astype("category")

    lgb_tr = lgb.Dataset(X_tr_lgb, label=y_tr, free_raw_data=False)
    lgb_va = lgb.Dataset(X_va_lgb, label=y_va, reference=lgb_tr, free_raw_data=False)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": params.get("num_leaves", 45),
        "max_depth": params.get("max_depth", -1),
        "learning_rate": params.get("learning_rate", 0.030),
        "min_child_samples": params.get("min_child_samples", 40),
        "feature_fraction": params.get("feature_fraction", 0.70),
        "bagging_fraction": params.get("bagging_fraction", 0.80),
        "bagging_freq": params.get("bagging_freq", 1),
        "lambda_l1": params.get("lambda_l1", 0.5),
        "lambda_l2": params.get("lambda_l2", 5.0),
        "extra_trees": params.get("extra_trees", False),
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        lgb_params,
        lgb_tr,
        num_boost_round=params.get("num_boost_round", 800),
        valid_sets=[lgb_va],
        callbacks=[lgb.early_stopping(params.get("early_stopping_rounds", 40), verbose=False)],
    )
    val_prob = model.predict(X_va_lgb)
    test_prob = model.predict(X_te_lgb)
    return np.clip(val_prob, FLOOR, CEIL), np.clip(test_prob, FLOOR, CEIL)


def run_stage1_hillclimb(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    k_top_features: int = 60,
    n_splits: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_df.columns else "Target"

    print("=" * 80, flush=True)
    print("STAGE 1: METRIC-DIRECT STEPWISE HILL CLIMBING ENSEMBLING", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Anchor from Baseline (Auto-triggers baseline if missing)
    anchor_oof_path = os.path.join(output_dir, "oof_champ_train.npy")
    anchor_sub_path = os.path.join(sub_dir, "submission_baseline.csv")
    if not os.path.exists(anchor_sub_path):
        anchor_sub_path = os.path.join(sub_dir, "submission_best_0.73731.csv")

    if not (os.path.exists(anchor_oof_path) and os.path.exists(anchor_sub_path)):
        print("Anchor artifacts not found on disk. Automatically running baseline first...", flush=True)
        from src.baseline import run_baseline
        run_baseline(
            train_path=train_path,
            test_path=test_path,
            output_dir=output_dir,
            sub_dir=sub_dir,
            k_top_features=k_top_features,
            n_splits=n_splits,
        )
        if not os.path.exists(anchor_sub_path):
            anchor_sub_path = os.path.join(sub_dir, "submission_baseline.csv")

    assert os.path.exists(anchor_oof_path), f"Missing anchor OOF at {anchor_oof_path}!"
    assert os.path.exists(anchor_sub_path), f"Missing anchor submission at {anchor_sub_path}!"

    oof_anchor = np.load(anchor_oof_path)
    sub_anchor_df = pd.read_csv(anchor_sub_path)
    test_anchor = sub_anchor_df["Target"].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(int)

    a_ll, a_auc, a_comp = competition_score(y_train, oof_anchor)
    print(f"✓ Loaded Baseline Anchor: Comp={a_comp:.5f} | AUC={a_auc:.5f} | LL={a_ll:.5f}", flush=True)

    # 2. Prepare Features via feature_engine
    print("\nStep 1: Engineering & selecting clean feature subset", flush=True)
    train_fe, test_fe, categorical_cols, _ = engineer_features(train_df, test_df)
    X_tr_raw = train_fe.drop(columns=[target_col, ID_COL])
    X_te_raw = test_fe.drop(columns=[c for c in [ID_COL] if c in test_fe.columns])

    X_tr_sel, X_te_sel, selected_features = run_feature_engine_selection(
        X_tr_raw, y_train, X_te_raw, cat_cols=categorical_cols, k_top=k_top_features, corr_threshold=0.98, seed=SEED
    )

    active_cats = [c for c in categorical_cols if c in selected_features]
    print(f"Selected {len(selected_features)} features ({len(active_cats)} categoricals)", flush=True)

    # 3. Define Diverse Model Candidates
    model_specs = [
        (
            "cb_tuned_d7_ms",
            "CatBoost GPU Tuned Depth 7 (Multi-Seed [42, 2026])",
            lambda xtr, ytr, xva, yva, xte: run_tuned_catboost_ms(
                xtr, ytr, xva, yva, xte, active_cats,
                {"depth": 7, "learning_rate": 0.035, "l2_leaf_reg": 25.0, "random_strength": 1.0, "iterations": 900},
                seeds=[42, 2026],
            ),
        ),
        (
            "lgb_leaf_wise",
            "LightGBM Leaf-Wise (num_leaves=45, min_child=40)",
            lambda xtr, ytr, xva, yva, xte: run_tuned_lightgbm(
                xtr, ytr, xva, yva, xte, active_cats,
                {"num_leaves": 45, "learning_rate": 0.030, "min_child_samples": 40, "feature_fraction": 0.70, "bagging_fraction": 0.80},
                seed=SEED,
            ),
        ),
        (
            "lgb_extra_trees",
            "LightGBM ExtraTrees (colsample=0.60, min_child=60)",
            lambda xtr, ytr, xva, yva, xte: run_tuned_lightgbm(
                xtr, ytr, xva, yva, xte, active_cats,
                {"num_leaves": 40, "learning_rate": 0.030, "min_child_samples": 60, "feature_fraction": 0.60, "bagging_fraction": 0.75, "extra_trees": True},
                seed=2026,
            ),
        ),
    ]

    candidate_oof: Dict[str, np.ndarray] = {"oof_anchor": oof_anchor}
    candidate_test: Dict[str, np.ndarray] = {"oof_anchor": test_anchor}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_indices = list(skf.split(X_tr_sel, y_train))

    for key, label, trainer in model_specs:
        print(f"\n>>> Training Candidate: {label} <<<", flush=True)
        t0 = time.time()
        m_oof = np.zeros(len(train_df))
        test_preds = []

        for fold, (trn_idx, val_idx) in enumerate(fold_indices, start=1):
            x_tr, y_tr = X_tr_sel.iloc[trn_idx].copy(), y_train[trn_idx]
            x_va, y_va = X_tr_sel.iloc[val_idx].copy(), y_train[val_idx]
            x_te = X_te_sel.copy()

            val_p, test_p = trainer(x_tr, y_tr, x_va, y_va, x_te)
            m_oof[val_idx] = val_p
            test_preds.append(test_p)

            f_ll, f_auc, f_comp = competition_score(y_va, val_p)
            print(f"  Fold {fold}/{N_SPLITS} | Comp: {f_comp:.5f} | AUC: {f_auc:.5f} | LL: {f_ll:.5f}", flush=True)

        m_test = np.mean(test_preds, axis=0)
        m_ll, m_auc, m_comp = competition_score(y_train, m_oof)
        print(f"  ==> {key} OOF: Comp={m_comp:.5f} | AUC={m_auc:.5f} | LL={m_ll:.5f} ({time.time() - t0:.1f}s)", flush=True)

        candidate_oof[key] = m_oof
        candidate_test[key] = m_test

    # 4. Metric-Direct Stepwise Hill Climbing
    print("\n" + "=" * 80, flush=True)
    print("PHASE 2: METRIC-DIRECT STEPWISE HILL CLIMBING ENSEMBLING (via hillclimbers)", flush=True)
    print("=" * 80, flush=True)

    oof_cand_df = pd.DataFrame(candidate_oof)
    test_cand_df = pd.DataFrame(candidate_test)
    train_label_df = pd.DataFrame({TARGET: y_train})

    climb_res = climb_hill(
        train=train_label_df,
        oof_pred_df=oof_cand_df,
        test_pred_df=test_cand_df,
        target=TARGET,
        objective="maximize",
        eval_metric=partial(comp_metric_eval),
        negative_weights=False,
        precision=0.01,
        plot_hill=False,
        plot_hist=False,
        return_oof_preds=True,
    )

    blended_test, blended_oof = climb_res
    b_ll, b_auc, b_comp = competition_score(y_train, blended_oof)
    print(f"\nRaw HillClimbers Blend OOF: Comp={b_comp:.5f} | AUC={b_auc:.5f} | LL={b_ll:.5f}", flush=True)

    # 5. Beta Calibration
    print("\nStep 3: Cross-fitted Beta Calibration on HillClimber Output", flush=True)
    cal_oof, cal_test, calibrator = beta_calibrate(blended_oof, blended_test, y_train, n_splits=5, seed=SEED)
    c_ll, c_auc, c_comp = competition_score(y_train, cal_oof)
    print(f"Beta Calibrated OOF: Comp={c_comp:.5f} | AUC={c_auc:.5f} | LL={c_ll:.5f}", flush=True)

    # Gating check: only adopt if >= raw blend and >= anchor
    if c_comp >= b_comp and c_comp >= a_comp:
        final_oof = cal_oof
        final_test = cal_test
        strategy_name = "hillclimb_beta_calibrated"
        print(f"  --> Adopted Calibrated HillClimber (+{c_comp - a_comp:+.5f} vs Anchor).", flush=True)
    elif b_comp >= a_comp:
        final_oof = blended_oof
        final_test = blended_test
        strategy_name = "hillclimb_raw_blend"
        print(f"  --> Adopted Raw HillClimber (+{b_comp - a_comp:+.5f} vs Anchor).", flush=True)
    else:
        final_oof = oof_anchor
        final_test = test_anchor
        strategy_name = "preserved_anchor"
        print("  --> Preserved Baseline Anchor (HillClimber did not improve).", flush=True)

    final_ll, final_auc, final_comp = competition_score(y_train, final_oof)
    print("\n" + "=" * 80, flush=True)
    print(f"★ FINAL STAGE 1 COMPOSITE SCORE: {final_comp:.5f} (AUC: {final_auc:.5f}, LL: {final_ll:.5f}) ★", flush=True)
    print("=" * 80, flush=True)

    # 6. Save Artifacts & Update Champion
    np.save(os.path.join(output_dir, "oof_champ_train.npy"), final_oof)
    np.save(os.path.join(output_dir, "oof_stage1_candidates.npy"), oof_cand_df.to_numpy())

    sub_path = os.path.join(sub_dir, "submission_stage1_hillclimb.csv")
    sub_df = pd.DataFrame({
        ID_COL: test_fe[ID_COL],
        "Target": np.clip(final_test, FLOOR, CEIL),
    })
    sub_df.to_csv(sub_path, index=False)
    # Also update baseline submission if improved
    if final_comp >= a_comp:
        sub_df.to_csv(os.path.join(sub_dir, "submission_baseline.csv"), index=False)
        sub_df.to_csv(os.path.join(sub_dir, "submission_best_0.73731.csv"), index=False)

    metadata = {
        "stage": "stage1_hillclimb",
        "previous_anchor_comp": a_comp,
        "final_comp": final_comp,
        "final_auc": final_auc,
        "final_logloss": final_ll,
        "strategy": strategy_name,
        "candidates": list(candidate_oof.keys()),
        "submission_path": str(sub_path),
    }

    with open(os.path.join(output_dir, "run_summary_stage1.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return final_comp, final_oof, final_test


if __name__ == "__main__":
    run_stage1_hillclimb()
