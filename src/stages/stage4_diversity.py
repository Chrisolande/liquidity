"""
Stage 4: Diversity Base Learners Pipeline.
Executes:
  1. 10-Fold 70k Distillation Students (LightGBM & XGBoost regressors on soft temperature-sharpened teacher targets)
     -> checkpoints/distill_student_10fold.npz
  2. 10-Fold PyTorch Neural TabMLP -> checkpoints/tab_mlp_fast.npz
  3. 5-Fold MultiStrata Iterative Stratification Ensemble -> checkpoints/multistrata.npz
"""

import os
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import (
    SEED,
    FLOOR,
    CEIL,
    TARGET,
    ID_COL,
    get_default_dataset_paths,
)
from src.features.pipeline import engineer_features
from src.features.encoding import extract_domain_feature_subsets
from src.models.neural import fit_mlp_runner
from src.models.multistrata import run_multistrata_pipeline
from src.models.distillation import temperature_sharpen
from src.ensemble.stacking import blend_and_calibrate
from src.metrics import competition_score


def run_stage4(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    n_splits: int = 10,
) -> float:
    """Executes Stage 4 Diversity Pipeline."""
    print("Stage 4: Diversity Base Learners Pipeline (10-Fold CV)", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_raw.columns else "Target"
    y_true = train_raw[target_col].to_numpy(dtype=float)
    n_train = len(train_raw)
    n_test = len(test_raw)

    X_tr_feat, X_te_feat, cat_cols, _ = engineer_features(train_raw, test_raw)
    drop_meta = [c for c in [ID_COL, target_col] if c in X_tr_feat.columns]
    X_tr_feat = X_tr_feat.drop(columns=drop_meta)
    X_te_feat = X_te_feat.drop(columns=[c for c in [ID_COL] if c in X_te_feat.columns])

    dom2_cols, dom3_cols, dom_triage_cols = extract_domain_feature_subsets(list(X_tr_feat.columns))
    dom3_cols = [c for c in dom3_cols if c in X_tr_feat.columns]
    if len(dom3_cols) == 0:
        dom3_cols = list(X_tr_feat.columns)[:35]

    X_tr_dom35 = X_tr_feat[dom3_cols].to_numpy(dtype=np.float32)
    X_te_dom35 = X_te_feat[dom3_cols].to_numpy(dtype=np.float32)

    # Impute NaNs with column means
    col_means = np.nanmean(X_tr_dom35, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds_tr = np.where(np.isnan(X_tr_dom35))
    X_tr_dom35[inds_tr] = np.take(col_means, inds_tr[1])
    inds_te = np.where(np.isnan(X_te_dom35))
    X_te_dom35[inds_te] = np.take(col_means, inds_te[1])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_tr_dom35, y_true))

    # =========================================================================
    # STAGE 4A: 10-FOLD 70k SELF-TRAINING DISTILLATION STUDENTS
    # =========================================================================
    # Ensure MultiStrata exists BEFORE building Grand Teacher
    ms_path = os.path.join(output_dir, "multistrata.npz")
    if not os.path.exists(ms_path):
        print("Stage 4 Pre-requisite: MultiStrata Iterative Stratification Ensemble", flush=True)
        run_multistrata_pipeline(train_path=train_path, test_path=test_path, output_dir=output_dir, sub_dir=sub_dir)

    # Load Teacher Predictions from Stage 2 Checkpoints
    p6_candidates = [
        os.path.join(output_dir, "gbdt_zoo_4seed.npz"),
        os.path.join(output_dir, "gbdt_balanced_dom1_p2_seeds4.npz"),
        os.path.join(output_dir, "gbdt_balanced_dom1_p2_seeds2.npz"),
    ]
    p6_path = next((p for p in p6_candidates if os.path.exists(p)), None)

    oof_teacher_dict: Dict[str, np.ndarray] = {}
    test_teacher_dict: Dict[str, np.ndarray] = {}

    if p6_path:
        p6_cache = np.load(p6_path)
        for k in p6_cache.files:
            if k.startswith("oof_") and k != "oof_s4_tree_zoo":
                m_name = k.replace("oof_", "")
                oof_teacher_dict[m_name] = p6_cache[k]
                t_key = f"test_{m_name}" if f"test_{m_name}" in p6_cache else k
                test_teacher_dict[m_name] = p6_cache[t_key] if t_key in p6_cache else p6_cache[k]
            elif k in ["cb_d7_dom26", "cb_d6_dom35", "xgb_d4_dom35", "xgb_d4_triage", "lgb_extra"]:
                oof_teacher_dict[k] = p6_cache[k]
                t_key = f"test_{k}" if f"test_{k}" in p6_cache else k
                test_teacher_dict[k] = p6_cache[t_key] if t_key in p6_cache else p6_cache[k]
        if "oof_s4_tree_zoo" in p6_cache and len(oof_teacher_dict) == 0:
            oof_teacher_dict["zoo_ensemble"] = p6_cache["oof_s4_tree_zoo"]
            test_teacher_dict["zoo_ensemble"] = p6_cache["test_s4_tree_zoo"]

    if os.path.exists(ms_path):
        ms = np.load(ms_path)
        k_ms = "oof_raw" if "oof_raw" in ms else ("oof_multistrata" if "oof_multistrata" in ms else "p_ms_oof")
        k_te = "test_raw" if "test_raw" in ms else ("test_multistrata" if "test_multistrata" in ms else "p_ms_test")
        oof_teacher_dict["multistrata"] = ms[k_ms]
        test_teacher_dict["multistrata"] = ms[k_te]

    pfn_path = os.path.join(output_dir, "tabpfn.npz")
    if os.path.exists(pfn_path):
        pfn = np.load(pfn_path)
        for k in ["pfn_phys", "pfn_champ", "tabpfn"]:
            oof_k = f"oof_{k}" if f"oof_{k}" in pfn else (f"{k}_oof" if f"{k}_oof" in pfn else None)
            test_k = f"test_{k}" if f"test_{k}" in pfn else (f"{k}_test" if f"{k}_test" in pfn else None)
            if oof_k and test_k:
                oof_teacher_dict[k] = pfn[oof_k]
                test_teacher_dict[k] = pfn[test_k]

    champ_oof_p = os.path.join(output_dir, "oof_champ_train.npy")
    champ_sub_p = os.path.join(sub_dir, "submission_best_0.73731.csv")
    if os.path.exists(champ_oof_p) and os.path.exists(champ_sub_p):
        oof_teacher_dict["oof_champ"] = np.load(champ_oof_p)
        test_teacher_dict["oof_champ"] = pd.read_csv(champ_sub_p)["Target"].values
        print("Integrated Tournament Champion Anchor into Grand Teacher", flush=True)

    # Blend teacher predictions to obtain soft pseudo labels
    if oof_teacher_dict:
        _, _, _, oof_t_cal, test_t_cal = blend_and_calibrate(
            oof_teacher_dict, test_teacher_dict, y_true, n_splits=n_splits, seed=SEED
        )
        teacher_ll, teacher_auc, teacher_comp = competition_score(y_true, oof_t_cal)
        print(f"Teacher Calibrated OOF: Comp={teacher_comp:.5f} | AUC={teacher_auc:.5f} | LL={teacher_ll:.5f}", flush=True)
    else:
        # Fallback baseline teacher if no prior stage checkpoint exists
        test_t_cal = np.full(n_test, float(y_true.mean()))

    # Temperature Sharpening (T = 0.85)
    p_sharp = temperature_sharpen(test_t_cal, temperature=0.85)

    oof_student_lgb = np.zeros(n_train, dtype=np.float64)
    test_student_lgb = np.zeros(n_test, dtype=np.float64)
    oof_student_xgb = np.zeros(n_train, dtype=np.float64)
    test_student_xgb = np.zeros(n_test, dtype=np.float64)
    pseudo_weight = 0.10

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_fold_tr = np.vstack([X_tr_dom35[trn_idx], X_te_dom35])
        y_fold_tr = np.r_[y_true[trn_idx], p_sharp]
        w_fold_tr = np.r_[np.ones(len(trn_idx), dtype=np.float32), np.full(n_test, pseudo_weight, dtype=np.float32)]
        X_fold_val = X_tr_dom35[val_idx]
        y_fold_val = y_true[val_idx]

        # Student 1: LightGBM Regressor (cross_entropy objective)
        lgb_student = LGBMRegressor(
            objective="cross_entropy",
            n_estimators=900,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED + fold,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_student.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
        oof_student_lgb[val_idx] = np.clip(lgb_student.predict(X_fold_val), FLOOR, CEIL)
        test_student_lgb += np.clip(lgb_student.predict(X_te_dom35), FLOOR, CEIL) / float(n_splits)

        # Student 2: XGBoost Regressor (binary:logistic objective)
        xgb_student = XGBRegressor(
            objective="binary:logistic",
            n_estimators=700,
            learning_rate=0.035,
            max_depth=4,
            max_leaves=24,
            grow_policy="lossguide",
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=SEED + fold,
            tree_method="hist",
            n_jobs=-1,
        )
        xgb_student.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
        oof_student_xgb[val_idx] = np.clip(xgb_student.predict(X_fold_val), FLOOR, CEIL)
        test_student_xgb += np.clip(xgb_student.predict(X_te_dom35), FLOOR, CEIL) / float(n_splits)

    lgb_ll, lgb_auc, lgb_comp = competition_score(y_true, oof_student_lgb)
    xgb_ll, xgb_auc, xgb_comp = competition_score(y_true, oof_student_xgb)
    print(f"LightGBM Student: Comp={lgb_comp:.5f} | AUC={lgb_auc:.5f} | LL={lgb_ll:.5f}", flush=True)
    print(f"XGBoost Student:  Comp={xgb_comp:.5f} | AUC={xgb_auc:.5f} | LL={xgb_ll:.5f}", flush=True)

    np.savez_compressed(
        os.path.join(output_dir, "distill_student_10fold.npz"),
        oof_lgb_student=oof_student_lgb,
        test_lgb_student=test_student_lgb,
        oof_xgb_student=oof_student_xgb,
        test_xgb_student=test_student_xgb,
    )

    # =========================================================================
    # STAGE 4B: PYTORCH NEURAL TABMLP
    # =========================================================================
    print("Stage 4B: PyTorch Neural TabMLP Base Learner", flush=True)
    oof_mlp, test_mlp, mlp_comp = fit_mlp_runner(
        X_train=pd.DataFrame(X_tr_dom35, columns=dom3_cols),
        y_train=y_true,
        X_test=pd.DataFrame(X_te_dom35, columns=dom3_cols),
        folds=folds,
        feature_cols=dom3_cols,
        epochs=15,
    )
    np.savez_compressed(os.path.join(output_dir, "tab_mlp_fast.npz"), oof_mlp=oof_mlp, test_mlp=test_mlp)

    # Combine Diversity Learners (PyTorch TabMLP + Distillation Students)
    raw_div_oof = (oof_mlp + oof_student_lgb + oof_student_xgb) / 3.0
    raw_div_test = (test_mlp + test_student_lgb + test_student_xgb) / 3.0
    final_ll, final_auc, final_comp = competition_score(y_true, raw_div_oof)
    print(f"Stage 4 Diversity Final: Comp={final_comp:.5f} | AUC={final_auc:.5f} | LL={final_ll:.5f}", flush=True)

    sub_path = os.path.join(sub_dir, "submission_stage4_diversity.csv")
    pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": np.clip(raw_div_test, FLOOR, CEIL)}).to_csv(sub_path, index=False)
    return final_comp


if __name__ == "__main__":
    run_stage4()
