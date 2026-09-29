"""
Multi-Strata / Iterative Stratification training pipeline and cohort stratification utilities.
Guarantees balanced joint distributions across Target x Customer Segment x Balance Quartile x Earning Pattern.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from src.ensemble.calibration import beta_calibrate

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ImportError:
    MultilabelStratifiedKFold = None

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
from src.features.pipeline import engineer_features
from src.features.encoding import apply_fold_target_encoding
from src.metrics import competition_score
from src.models.distillation import run_pseudo_student
from src.ensemble.pseudo_labels import build_pseudo_labels


def build_multilabel_stratification_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Constructs multi-label stratification matrix:
    Target x Customer Segment x Balance Quartile x Earning Pattern.
    """
    target_col = TARGET if TARGET in df.columns else "Target"
    y_target = pd.get_dummies(df[target_col], prefix="target")
    y_segment = pd.get_dummies(df["segment"], prefix="seg")
    y_bal_q = pd.get_dummies(pd.qcut(df["m1_daily_avg_bal"].rank(method="first"), 4), prefix="bal_q")
    earn_col = "earning_pattern" if "earning_pattern" in df.columns else "financial_earning_indicator"
    y_earn = pd.get_dummies(df[earn_col], prefix="earn")
    return pd.concat([y_target, y_segment, y_bal_q, y_earn], axis=1).to_numpy()


def run_catboost_multistrata(
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
    params: dict,
    cat_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    task_type = "GPU" if HAS_GPU else "CPU"
    seeds = params.get("seeds", [42, 2026])
    val_preds_seeds = []
    test_preds_seeds = []
    obj_cols = xtr.select_dtypes(include=['category', 'object']).columns.tolist()
    all_cat_idx = sorted(list(set(cat_indices).union([xtr.columns.get_loc(c) for c in obj_cols if c in xtr.columns])))
    for s in seeds:
        cb = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=params.get("iterations", 750),
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params.get("random_strength", 1.0),
            bagging_temperature=params.get("bagging_temperature", 0.5),
            border_count=params.get("border_count", 128),
            task_type=task_type,
            random_seed=s,
            verbose=False,
        )
        cb.fit(
            xtr,
            ytr,
            eval_set=(xva, yva),
            cat_features=all_cat_idx if all_cat_idx else None,
            early_stopping_rounds=params.get("early_stopping_rounds", 40),
            verbose=False,
        )
        val_preds_seeds.append(cb.predict_proba(xva)[:, 1])
        test_preds_seeds.append(cb.predict_proba(xte)[:, 1])
    return np.mean(val_preds_seeds, axis=0), np.mean(test_preds_seeds, axis=0)


def run_xgboost_multistrata(
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
    params: dict,
    cat_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    xtr_xgb = xtr.copy()
    xva_xgb = xva.copy()
    xte_xgb = xte.copy()
    all_cats = list(set(cat_cols).union(xtr_xgb.select_dtypes(include=['category', 'object']).columns))
    for c in all_cats:
        if c in xtr_xgb.columns:
            all_vals = pd.concat([xtr_xgb[c], xva_xgb[c], xte_xgb[c]]).astype(str).unique().tolist()
            mapping = {v: i for i, v in enumerate(all_vals)}
            xtr_xgb[c] = xtr_xgb[c].astype(str).map(mapping).fillna(-1).astype(int)
            xva_xgb[c] = xva_xgb[c].astype(str).map(mapping).fillna(-1).astype(int)
            xte_xgb[c] = xte_xgb[c].astype(str).map(mapping).fillna(-1).astype(int)

    seeds = params.get("seeds", [42, 2026])
    val_preds_seeds = []
    test_preds_seeds = []

    dtrain = xgb.DMatrix(xtr_xgb, label=ytr, enable_categorical=True)
    dval = xgb.DMatrix(xva_xgb, label=yva, enable_categorical=True)
    dtest = xgb.DMatrix(xte_xgb, enable_categorical=True)

    for s in seeds:
        p = {
            "tree_method": "hist",
            "device": "cuda" if HAS_GPU else "cpu",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params.get("subsample", 0.80),
            "colsample_bytree": params.get("colsample_bytree", 0.70),
            "reg_lambda": params.get("reg_lambda", 5.0),
            "reg_alpha": params.get("reg_alpha", 0.5),
            "seed": s,
            "random_state": s,
        }
        bst = xgb.train(
            p,
            dtrain,
            num_boost_round=params.get("num_boost_round", 800),
            evals=[(dval, "val")],
            early_stopping_rounds=params.get("early_stopping_rounds", 40),
            verbose_eval=False,
        )
        val_preds_seeds.append(bst.predict(dval))
        test_preds_seeds.append(bst.predict(dtest))

    return np.mean(val_preds_seeds, axis=0), np.mean(test_preds_seeds, axis=0)


def run_lightgbm_multistrata(
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
    params: dict,
    categorical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    all_cats = list(set(categorical_cols).union(xtr.select_dtypes(include=['category', 'object']).columns))
    cat_cols_present = [c for c in all_cats if c in xtr.columns]
    xtr_lgb = xtr.copy()
    xva_lgb = xva.copy()
    xte_lgb = xte.copy()
    for c in cat_cols_present:
        xtr_lgb[c] = xtr_lgb[c].astype("category")
        xva_lgb[c] = xva_lgb[c].astype("category")
        xte_lgb[c] = xte_lgb[c].astype("category")

    lgb_tr = lgb.Dataset(xtr_lgb, label=ytr, categorical_feature=cat_cols_present, free_raw_data=False)
    lgb_va = lgb.Dataset(xva_lgb, label=yva, reference=lgb_tr, categorical_feature=cat_cols_present, free_raw_data=False)
    seeds = params.get("seeds", [42, 2026])
    val_preds_seeds = []
    test_preds_seeds = []
    for s in seeds:
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": params["num_leaves"],
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "min_child_samples": params.get("min_child_samples", 40),
            "feature_fraction": params.get("feature_fraction", 0.65),
            "bagging_fraction": params.get("bagging_fraction", 0.80),
            "bagging_freq": params.get("bagging_freq", 1),
            "bagging_seed": s + 11,
            "feature_fraction_seed": s + 22,
            "extra_seed": s + 33,
            "data_random_seed": s + 44,
            "deterministic": True,
            "force_col_wise": True,
            "lambda_l1": params.get("lambda_l1", 0.5),
            "lambda_l2": params.get("lambda_l2", 5.0),
            "seed": s,
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
        val_preds_seeds.append(model.predict(xva_lgb))
        test_preds_seeds.append(model.predict(xte_lgb))
    return np.mean(val_preds_seeds, axis=0), np.mean(test_preds_seeds, axis=0)


def run_multistrata_pipeline(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    pseudo_file: str = "pseudo_labels_iter3.csv",
) -> float:
    """Executes the full 5-fold iterative stratification multi-strata ensemble pipeline."""
    total_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_raw.columns else "Target"

    Y_multi = build_multilabel_stratification_matrix(train_raw)
    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    train_fe, test_fe, categorical_cols, numeric_cols = engineer_features(train_raw, test_raw)
    feature_cols = [c for c in train_fe.columns if c not in {target_col, ID_COL}]

    X = train_fe[feature_cols].copy()
    y = train_fe[target_col].to_numpy(dtype=int)
    X_test = test_fe[feature_cols].copy()

    # Discover ALL categorical and object columns in the dataset
    detected_cats = train_fe[feature_cols].select_dtypes(include=['category', 'object']).columns.tolist()
    categorical_cols = list(dict.fromkeys(list(categorical_cols) + detected_cats))

    # Pre-convert categoricals to clean string objects
    clean_cat_cols = []
    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype(str).replace({"nan": "missing", "None": "missing"}).fillna("missing")
            X_test[c] = X_test[c].astype(str).replace({"nan": "missing", "None": "missing"}).fillna("missing")
            clean_cat_cols.append(c)

    te_targets = [c for c in ["segment", "region", "gender", "seg_earn", "reg_seg", "gen_earn"] if c in X.columns]

    fold_splits = []
    for fold, (trn_idx, val_idx) in enumerate(mskf.split(X, Y_multi), start=1):
        x_tr, y_tr = X.iloc[trn_idx].copy(), y[trn_idx]
        x_va, y_va = X.iloc[val_idx].copy(), y[val_idx]
        x_te = X_test.copy()
        x_tr, x_va, x_te = apply_fold_target_encoding(x_tr, y_tr, x_va, x_te, te_targets, smoothing=20.0, seed=SEED + fold)
        fold_splits.append((trn_idx, val_idx, x_tr, y_tr, x_va, y_va, x_te))

    fold_feature_cols = list(fold_splits[0][2].columns)
    cat_indices = [fold_feature_cols.index(c) for c in clean_cat_cols if c in fold_feature_cols]

    models_oof: Dict[str, np.ndarray] = {}
    models_test: Dict[str, np.ndarray] = {}

    model_specs = [
        ("ms_cb_w2_d5", "MultiStrata CatBoost GPU Depth 5",
         lambda xtr, ytr, xva, yva, xte: run_catboost_multistrata(
             xtr, ytr, xva, yva, xte,
             {"depth": 5, "learning_rate": 0.035, "l2_leaf_reg": 45.0, "random_strength": 1.5, "bagging_temperature": 0.30, "seeds": [42, 2026]},
             cat_indices)),
        ("ms_cb_w2_d6", "MultiStrata CatBoost GPU Depth 6",
         lambda xtr, ytr, xva, yva, xte: run_catboost_multistrata(
             xtr, ytr, xva, yva, xte,
             {"depth": 6, "learning_rate": 0.032, "l2_leaf_reg": 35.0, "random_strength": 1.0, "bagging_temperature": 0.35, "seeds": [42, 2026]},
             cat_indices)),
        ("ms_cb_w2_d7", "MultiStrata CatBoost GPU Depth 7",
         lambda xtr, ytr, xva, yva, xte: run_catboost_multistrata(
             xtr, ytr, xva, yva, xte,
             {"depth": 7, "learning_rate": 0.030, "l2_leaf_reg": 25.0, "random_strength": 0.8, "bagging_temperature": 0.25, "seeds": [42, 2026]},
             cat_indices)),
        ("ms_xgb_d4", "MultiStrata XGBoost GPU Depth 4",
         lambda xtr, ytr, xva, yva, xte: run_xgboost_multistrata(
             xtr, ytr, xva, yva, xte,
             {"max_depth": 4, "learning_rate": 0.035, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.5, "seeds": [42, 2026]},
             categorical_cols)),
        ("ms_xgb_d5", "MultiStrata XGBoost GPU Depth 5",
         lambda xtr, ytr, xva, yva, xte: run_xgboost_multistrata(
             xtr, ytr, xva, yva, xte,
             {"max_depth": 5, "learning_rate": 0.032, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 5.0, "reg_alpha": 0.5, "seeds": [42, 2026]},
             categorical_cols)),
        ("ms_lgb_d4", "MultiStrata LightGBM Depth 4 Leaves 15",
         lambda xtr, ytr, xva, yva, xte: run_lightgbm_multistrata(
             xtr, ytr, xva, yva, xte,
             {"max_depth": 4, "num_leaves": 15, "learning_rate": 0.035, "min_child_samples": 40, "feature_fraction": 0.65, "bagging_fraction": 0.80, "lambda_l2": 5.0},
             categorical_cols)),
    ]

    for key, label, trainer in model_specs:
        oof_vec = np.zeros(len(train_fe))
        test_preds = []
        for fold, (trn_idx, val_idx, x_tr, y_tr, x_va, y_va, x_te) in enumerate(fold_splits, start=1):
            val_p, test_p = trainer(x_tr, y_tr, x_va, y_va, x_te)
            oof_vec[val_idx] = val_p
            test_preds.append(test_p)
        models_oof[key] = oof_vec
        models_test[key] = np.mean(test_preds, axis=0)

    # Multi-strata student models on pseudo-labels
    p_path = Path(pseudo_file)
    if not p_path.exists():
        try:
            build_pseudo_labels(test_path=test_path, output_dir=output_dir, sub_dir=sub_dir, out_path=str(p_path))
        except Exception as e:
            print(f"MultiStrata note: Unable to build pseudo-labels ({e}), skipping pseudo students.", flush=True)

    if p_path.exists():
        df_p = pd.read_csv(p_path).set_index(ID_COL).loc[test_fe[ID_COL]]
        y_pseudo = np.clip(df_p["Target"].to_numpy(dtype=float), FLOOR, CEIL)
        for s_key, s_label, mtype in [
            ("ms_cb_student", "MultiStrata CatBoost Student", "catboost"),
            ("ms_xgb_student", "MultiStrata XGBoost Student", "xgboost"),
        ]:
            s_oof_vec = np.zeros(len(train_fe))
            s_test_preds = []
            for fold, (trn_idx, val_idx, x_tr, y_tr, x_va, y_va, x_te) in enumerate(fold_splits, start=1):
                val_prob, test_prob = run_pseudo_student(x_tr, y_tr, x_va, y_va, x_te, y_pseudo, cat_indices, categorical_cols, model_type=mtype)
                s_oof_vec[val_idx] = val_prob
                s_test_preds.append(test_prob)
            models_oof[s_key] = s_oof_vec
            models_test[s_key] = np.mean(s_test_preds, axis=0)

    # Optimize Ensemble
    oof_df = pd.DataFrame(models_oof)
    test_df = pd.DataFrame(models_test)
    P_mat = oof_df.to_numpy(dtype=float)
    T_mat = test_df.to_numpy(dtype=float)
    n_models = P_mat.shape[1]

    def nm_objective(theta):
        w = softmax(theta)
        p = np.clip(P_mat @ w, FLOOR, CEIL)
        auc = roc_auc_score(y, p)
        ll = log_loss(y, p)
        return -(0.40 * auc + 0.60 * (1.0 - ll / 0.595))

    restarts = [("Equal Weights", np.zeros(n_models))]
    best_solo = np.argmax([competition_score(y, P_mat[:, i])[2] for i in range(n_models)])
    th_solo = np.full(n_models, -3.0)
    th_solo[best_solo] = 3.0
    restarts.append(("Top Solo Warmstart", th_solo))

    best_nm_comp = -1.0
    best_w = None
    for _, th_init in restarts:
        opt = minimize(nm_objective, th_init, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-5})
        c = -opt.fun
        if c > best_nm_comp:
            best_nm_comp = c
            best_w = softmax(opt.x)

    raw_oof = np.clip(P_mat @ best_w, FLOOR, CEIL)
    raw_test = np.clip(T_mat @ best_w, FLOOR, CEIL)

    # 5-fold cross-calibrated Beta calibration (smooth, monotonic, no step-plateaus)
    cal_oof, cal_test, _ = beta_calibrate(raw_oof, raw_test, y, n_splits=5, seed=SEED)

    _, _, cal_comp = competition_score(y, cal_oof)

    np.savez_compressed(
        os.path.join(output_dir, "multistrata.npz"),
        oof_multistrata=cal_oof,
        test_multistrata=cal_test,
        oof_raw=raw_oof,
        test_raw=raw_test,
    )
    pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": np.clip(cal_test, FLOOR, CEIL)}).to_csv(
        os.path.join(sub_dir, "submission_multistrata.csv"), index=False
    )
    return cal_comp
