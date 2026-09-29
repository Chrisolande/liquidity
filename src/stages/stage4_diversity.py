"""
Stage 4: Diversity Base Learners Pipeline.
Executes:
  1. 10-Fold 70k Distillation Students (LightGBM & XGBoost regressors on soft temperature-sharpened teacher targets)
     -> checkpoints/distill_student_10fold.npz
  2. 10-Fold PyTorch Neural TabMLP -> checkpoints/tab_mlp_fast.npz
  3. 5-Fold MultiStrata Iterative Stratification Ensemble -> checkpoints/multistrata.npz
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Ensure repo root is in sys.path for direct script execution by reviewers
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import (
    SEED,
    FLOOR,
    CEIL,
    TARGET,
    ID_COL,
    HAS_GPU,
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
    use_tabpfn: bool = False,
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

    # Stage 1 equivalent selection (~87 features: protected anchors + top numeric signals)
    feat_cache_candidates = [
        os.path.join(output_dir, "stage1_selected_features.json"),
        os.path.join("pulled", "stage1_selected_features.json"),
        os.path.join("checkpoints", "stage1_selected_features.json"),
    ]
    feat_cache = next((p for p in feat_cache_candidates if os.path.exists(p)), None)
    if feat_cache:
        print(f"Loading cached Stage 1 feature set from {feat_cache}...", flush=True)
        with open(feat_cache) as f:
            student_cols = json.load(f)
        student_cols = [c for c in student_cols if c in X_tr_feat.columns]
        X_tr_sel = X_tr_feat[student_cols]
        X_te_sel = X_te_feat[student_cols]
    else:
        print("Selecting Stage 1 high-signal feature set (~87 features) for students...", flush=True)
        from src.features.selection import run_feature_engine_selection
        X_tr_sel, X_te_sel, student_cols = run_feature_engine_selection(
            X_tr_feat, y_true, X_te_feat, cat_cols=cat_cols, k_top=60, corr_threshold=0.98, seed=SEED
        )
        try:
            cache_out = os.path.join(output_dir, "stage1_selected_features.json")
            with open(cache_out, "w") as f:
                json.dump(student_cols, f)
        except Exception:
            pass

    print(f"Features for Stage 4 students and TabMLP: {len(student_cols)} ({student_cols[:5]}...)", flush=True)

    # Encode categoricals to integer codes and convert to float32 matrix
    X_tr_num = X_tr_sel.copy()
    X_te_num = X_te_sel.copy()
    for col in X_tr_num.columns:
        if X_tr_num[col].dtype.name in ("category", "object") or pd.api.types.is_object_dtype(X_tr_num[col]):
            X_tr_num[col] = X_tr_num[col].astype("category").cat.codes.replace(-1, np.nan)
            X_te_num[col] = X_te_num[col].astype("category").cat.codes.replace(-1, np.nan)
        else:
            X_tr_num[col] = pd.to_numeric(X_tr_num[col], errors="coerce")
            X_te_num[col] = pd.to_numeric(X_te_num[col], errors="coerce")

    X_tr_arr = X_tr_num.to_numpy(dtype=np.float32)
    X_te_arr = X_te_num.to_numpy(dtype=np.float32)

    # Impute NaNs with column means
    col_means = np.nanmean(X_tr_arr, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds_tr = np.where(np.isnan(X_tr_arr))
    X_tr_arr[inds_tr] = np.take(col_means, inds_tr[1])
    inds_te = np.where(np.isnan(X_te_arr))
    X_te_arr[inds_te] = np.take(col_means, inds_te[1])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_tr_arr, y_true))

    # Stage 4A: 10-Fold Distillation Students
    # Ensure MultiStrata exists BEFORE building Grand Teacher
    ms_path = os.path.join(output_dir, "multistrata.npz")
    if not os.path.exists(ms_path):
        print("Stage 4 Pre-requisite: MultiStrata Iterative Stratification Ensemble", flush=True)
        run_multistrata_pipeline(train_path=train_path, test_path=test_path, output_dir=output_dir, sub_dir=sub_dir)

    # Load Teacher Predictions
    # Note: Combine ONLY distinct streams: Champion Anchor (oof_champ) + MultiStrata + TabPFN
    # Do NOT double-count individual GBDT zoo trees if oof_champ is already loaded!
    oof_teacher_dict: Dict[str, np.ndarray] = {}
    test_teacher_dict: Dict[str, np.ndarray] = {}

    # 1. Champion Anchor (contains Stage 1 + Stage 2 Zoo)
    champ_oof_candidates = [
        os.path.join(output_dir, "oof_champ_train.npy"),
        os.path.join("pulled", "oof_champ_train.npy"),
    ]
    champ_oof_p = next((p for p in champ_oof_candidates if os.path.exists(p)), None)

    champ_sub_candidates = [
        os.path.join(sub_dir, "submission_stage1_hillclimb.csv"),
        os.path.join("pulled", "submission_stage1_hillclimb.csv"),
        os.path.join(sub_dir, "submission_baseline.csv"),
        os.path.join("pulled", "submission_baseline.csv"),
    ]
    champ_sub_p = next((p for p in champ_sub_candidates if os.path.exists(p)), None)

    if champ_oof_p and champ_sub_p:
        oof_teacher_dict["oof_champ"] = np.load(champ_oof_p)
        test_teacher_dict["oof_champ"] = pd.read_csv(champ_sub_p)["Target"].values
        print(f"Integrated Champion Anchor into Grand Teacher from {champ_oof_p}", flush=True)
    else:
        # Fallback to individual GBDT zoo only if no champion anchor exists
        p6_candidates = [
            os.path.join(output_dir, "gbdt_zoo_4seed.npz"),
            os.path.join(output_dir, "gbdt_balanced_dom1_p2_seeds4.npz"),
            os.path.join(output_dir, "gbdt_balanced_dom1_p2_seeds2.npz"),
            os.path.join("pulled", "gbdt_zoo_4seed.npz"),
            os.path.join("pulled", "gbdt_balanced_dom1_p2_seeds4.npz"),
        ]
        p6_path = next((p for p in p6_candidates if os.path.exists(p)), None)
        if p6_path:
            p6_cache = np.load(p6_path)
            for k in p6_cache.files:
                if k.startswith("oof_") and k != "oof_s4_tree_zoo":
                    m_name = k.replace("oof_", "")
                    oof_teacher_dict[m_name] = p6_cache[k]
                    t_key = f"test_{m_name}" if f"test_{m_name}" in p6_cache else k
                    test_teacher_dict[m_name] = p6_cache[t_key] if t_key in p6_cache else p6_cache[k]
            if "oof_s4_tree_zoo" in p6_cache and len(oof_teacher_dict) == 0:
                oof_teacher_dict["zoo_ensemble"] = p6_cache["oof_s4_tree_zoo"]
                test_teacher_dict["zoo_ensemble"] = p6_cache["test_s4_tree_zoo"]

    # 2. MultiStrata Iterative Stratification Stream
    ms_candidates = [
        os.path.join(output_dir, "multistrata.npz"),
        os.path.join("pulled", "multistrata.npz"),
    ]
    ms_path = next((p for p in ms_candidates if os.path.exists(p)), None)
    if ms_path:
        ms = np.load(ms_path)
        k_ms = next((x for x in ["oof_raw", "oof_multistrata", "p_ms_oof"] if x in ms), None)
        k_te = next((x for x in ["test_raw", "test_multistrata", "p_ms_test"] if x in ms), None)
        if k_ms and k_te:
            oof_teacher_dict["multistrata"] = ms[k_ms]
            test_teacher_dict["multistrata"] = ms[k_te]
            print(f"Integrated MultiStrata stream into Grand Teacher from {ms_path}", flush=True)

    # 3. TabPFN Non-Tree Foundation Prior Stream
    if use_tabpfn:
        pfn_candidates = [
            os.path.join(output_dir, "tabpfn.npz"),
            os.path.join("pulled", "tabpfn.npz"),
            os.path.join(output_dir, "tabpfn_priors.npz"),
            os.path.join(output_dir, "experiments", "tabpfn_view_sweep.npz"),
            os.path.join("pulled", "tabpfn_view_sweep.npz"),
        ]
        pfn_path = next((p for p in pfn_candidates if os.path.exists(p)), None)
        if pfn_path:
            pfn = np.load(pfn_path)
            for k in ["pfn_phys", "pfn_champ", "tabpfn", "pfn_phys_legacy", "pfn_sweetspot_50", "pfn_solvency_runway"]:
                oof_k = next((x for x in [f"oof_{k}", f"{k}_oof", k] if x in pfn), None)
                test_k = next((x for x in [f"test_{k}", f"{k}_test", f"te_{k}"] if x in pfn), None)
                if oof_k:
                    oof_teacher_dict[k] = pfn[oof_k]
                    if test_k:
                        test_teacher_dict[k] = pfn[test_k]
                    else:
                        test_teacher_dict[k] = np.full(n_test, float(pfn[oof_k].mean()))
                    print(f"Integrated TabPFN stream '{k}' into Grand Teacher from {pfn_path}", flush=True)
    else:
        print("[Stage 4] Skipping TabPFN priors inclusion (use_tabpfn=False).", flush=True)

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

    oof_student_cb = np.zeros(n_train, dtype=np.float64)
    test_student_cb = np.zeros(n_test, dtype=np.float64)
    oof_student_xgb = np.zeros(n_train, dtype=np.float64)
    test_student_xgb = np.zeros(n_test, dtype=np.float64)

    cb_task = "GPU" if HAS_GPU else "CPU"

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_fold_tr = X_tr_arr[trn_idx]
        y_fold_tr = y_true[trn_idx]
        w_fold_tr = np.ones(len(trn_idx), dtype=np.float32)
        X_fold_val = X_tr_arr[val_idx]
        y_fold_val = y_true[val_idx]

        # Student 1: CatBoost Regressor (RMSE on soft continuous teacher targets)
        cb_student = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=900,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=5.0,
            task_type=cb_task,
            random_seed=SEED + fold,
            verbose=False,
        )
        cb_student.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
        oof_student_cb[val_idx] = np.clip(cb_student.predict(X_fold_val), FLOOR, CEIL)
        test_student_cb += np.clip(cb_student.predict(X_te_arr), FLOOR, CEIL) / float(n_splits)

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
            seed=SEED + fold,
            tree_method="hist",
            n_jobs=-1,
        )
        xgb_student.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
        oof_student_xgb[val_idx] = np.clip(xgb_student.predict(X_fold_val), FLOOR, CEIL)
        test_student_xgb += np.clip(xgb_student.predict(X_te_arr), FLOOR, CEIL) / float(n_splits)

    cb_ll, cb_auc, cb_comp = competition_score(y_true, oof_student_cb)
    xgb_ll, xgb_auc, xgb_comp = competition_score(y_true, oof_student_xgb)
    print(f"CatBoost Student: Comp={cb_comp:.5f} | AUC={cb_auc:.5f} | LL={cb_ll:.5f}", flush=True)
    print(f"XGBoost Student:  Comp={xgb_comp:.5f} | AUC={xgb_auc:.5f} | LL={xgb_ll:.5f}", flush=True)

    np.savez_compressed(
        os.path.join(output_dir, "distill_student_10fold.npz"),
        oof_cb_student=oof_student_cb,
        test_cb_student=test_student_cb,
        oof_lgb_student=oof_student_cb,  # for backward compatibility
        test_lgb_student=test_student_cb,
        oof_xgb_student=oof_student_xgb,
        test_xgb_student=test_student_xgb,
    )

    # Stage 4B: PyTorch Neural TabMLP
    print("Stage 4B: PyTorch Neural TabMLP Base Learner", flush=True)
    oof_mlp, test_mlp, mlp_comp = fit_mlp_runner(
        X_train=pd.DataFrame(X_tr_arr, columns=student_cols),
        y_train=y_true,
        X_test=pd.DataFrame(X_te_arr, columns=student_cols),
        folds=folds,
        feature_cols=student_cols,
        epochs=15,
    )
    np.savez_compressed(os.path.join(output_dir, "tab_mlp_fast.npz"), oof_mlp=oof_mlp, test_mlp=test_mlp)

    # Combine Diversity Learners (PyTorch TabMLP + Distillation Students)
    raw_div_oof = (oof_mlp + oof_student_cb + oof_student_xgb) / 3.0
    raw_div_test = (test_mlp + test_student_cb + test_student_xgb) / 3.0
    final_ll, final_auc, final_comp = competition_score(y_true, raw_div_oof)
    print(f"Stage 4 Diversity Final: Comp={final_comp:.5f} | AUC={final_auc:.5f} | LL={final_ll:.5f}", flush=True)

    sub_path = os.path.join(sub_dir, "submission_stage4_diversity.csv")
    pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": np.clip(raw_div_test, FLOOR, CEIL)}).to_csv(sub_path, index=False)
    return final_comp


if __name__ == "__main__":
    run_stage4()
