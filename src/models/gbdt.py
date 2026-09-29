"""
GBDT model training runners, Optuna hyperparameter tuners, and distillation self-training.
"""

import time
from typing import Any, Dict, List, Sequence, Tuple, Type
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from xgboost import XGBClassifier

from src.config import HAS_GPU, SEED
from src.features.encoding import apply_fold_target_encoding
from src.metrics import competition_score


def fit_gbdt_runner(
    name: str,
    model_cls: Type[Any],
    base_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    seeds: Sequence[int] = (42,),
    fold_data_cache: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """Unified cross-validation, bagging, fold target encoding, and prediction runner."""
    t0 = time.time()
    n_tr, n_te = len(X_train), len(X_test)
    oof_pred = np.zeros(n_tr)
    test_pred = np.zeros(n_te)

    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        if fold_data_cache is not None and fold_idx in fold_data_cache:
            x_tr, x_va, x_te = fold_data_cache[fold_idx]
            y_tr = y_train[trn_idx]
            y_va = y_train[val_idx]
        else:
            x_tr = X_train.iloc[trn_idx]
            y_tr = y_train[trn_idx]
            x_va = X_train.iloc[val_idx]
            y_va = y_train[val_idx]
            x_te = X_test.copy()

            x_tr, x_va, x_te = apply_fold_target_encoding(
                x_tr, y_tr, x_va, x_te, cat_cols, seed=seeds[0] + fold_idx
            )

            # Standardize categorical columns as pandas category dtype with uniform category mapping
            active_cat_cols = [c for c in cat_cols if c in x_tr.columns]
            for c in active_cat_cols:
                x_tr[c] = x_tr[c].astype(str).fillna("__missing__").astype("category")
                x_va[c] = pd.Categorical(x_va[c].astype(str).fillna("__missing__"), categories=x_tr[c].cat.categories)
                x_te[c] = pd.Categorical(x_te[c].astype(str).fillna("__missing__"), categories=x_tr[c].cat.categories)

            # Retain numeric columns and categorical columns
            num_cols = x_tr.select_dtypes(include="number").columns.tolist()
            keep_cols = num_cols + active_cat_cols
            x_tr = x_tr[keep_cols]
            x_va = x_va[keep_cols]
            x_te = x_te[keep_cols]

            if fold_data_cache is not None:
                fold_data_cache[fold_idx] = (x_tr, x_va, x_te)

        fold_val_p = np.zeros(len(val_idx))
        fold_test_p = np.zeros(n_te)

        for s in seeds:
            p = dict(base_params)
            fcols = p.pop("feature_cols", None)
            cur_tr, cur_va, cur_te = x_tr, x_va, x_te
            if fcols is not None:
                avail = [c for c in fcols if c in cur_tr.columns]
                # CRITICAL: ensure target-encoded signals and native categories reach all domain models
                avail += [c for c in cur_tr.columns if c.endswith("_te") and c not in avail]
                avail += [c for c in active_cat_cols if c in cur_tr.columns and c not in avail]
                cur_tr = cur_tr[avail]
                cur_va = cur_va[avail]
                cur_te = cur_te[avail]

            model_cats = [c for c in active_cat_cols if c in cur_tr.columns]
            cls_name = model_cls.__name__
            if cls_name == "CatBoostClassifier":
                p["random_seed"] = s
                p["verbose"] = False
                p["early_stopping_rounds"] = 50
                if HAS_GPU:
                    p["task_type"] = "GPU"
                m = model_cls(**p)
                m.fit(cur_tr, y_tr, eval_set=(cur_va, y_va), cat_features=model_cats if model_cats else None, verbose=False)
            elif cls_name in ["LGBMClassifier", "LGBMRegressor"]:
                p["random_state"] = s
                p["seed"] = s
                p.setdefault("bagging_seed", s + 11)
                p.setdefault("feature_fraction_seed", s + 22)
                p.setdefault("extra_seed", s + 33)
                p.setdefault("data_random_seed", s + 44)
                p.setdefault("deterministic", True)
                p.setdefault("force_col_wise", True)
                p["verbose"] = -1
                p["n_jobs"] = -1
                m = model_cls(**p)
                m.fit(
                    cur_tr,
                    y_tr,
                    eval_set=[(cur_va, y_va)],
                    categorical_feature=model_cats if model_cats else "auto",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            elif cls_name in ["XGBClassifier", "XGBRegressor"]:
                p["random_state"] = s
                p["seed"] = s
                p["verbosity"] = 0
                p["early_stopping_rounds"] = 50
                p["enable_categorical"] = True
                if HAS_GPU:
                    p["tree_method"] = "hist"
                    p["device"] = "cuda"
                m = model_cls(**p)
                m.fit(cur_tr, y_tr, eval_set=[(cur_va, y_va)], verbose=False)
            else:
                m = model_cls(**p)
                m.fit(cur_tr, y_tr)

            va_p = np.clip(m.predict_proba(cur_va)[:, 1], 1e-5, 1.0 - 1e-5)
            te_p = np.clip(m.predict_proba(cur_te)[:, 1], 1e-5, 1.0 - 1e-5)
            fold_val_p += va_p / len(seeds)
            fold_test_p += te_p / len(seeds)

        oof_pred[val_idx] = fold_val_p
        test_pred += fold_test_p / len(folds)
        f_ll, f_auc, f_comp = competition_score(y_va, fold_val_p)
        print(f"  [{name}] Fold {fold_idx + 1}/{len(folds)} | LogLoss: {f_ll:.4f} | AUC: {f_auc:.4f} | Comp: {f_comp:.4f}", flush=True)

    # Calibrate individual model predictions via Platt scaling to ensure minimum LogLoss
    from src.ensemble.calibration import platt_scaling_calibrate
    cal_oof, cal_test, _ = platt_scaling_calibrate(oof_pred, test_pred, y_train)

    ll, auc, comp = competition_score(y_train, cal_oof)
    print(f"[{name}] Full OOF Calibrated | LogLoss: {ll:.5f} | AUC: {auc:.5f} | Comp: {comp:.5f} ({time.time() - t0:.1f}s)", flush=True)
    return cal_oof, cal_test, (ll, auc, comp)



def run_catboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a standard multi-fold CatBoost model."""
    oof_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        x_tr, y_tr = x_train.iloc[trn_idx], y_train[trn_idx]
        x_va, y_va = x_train.iloc[val_idx], y_train[val_idx]
        x_te = x_test.copy()

        x_tr, x_va, x_te = apply_fold_target_encoding(x_tr, y_tr, x_va, x_te, cat_cols, seed=SEED + fold)

        model = CatBoostClassifier(**params)
        model.fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=0)
        oof_preds[val_idx] = model.predict_proba(x_va)[:, 1]
        test_preds += model.predict_proba(x_te)[:, 1] / len(folds)

    return oof_preds, test_preds


def run_xgboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a standard multi-fold XGBoost model."""
    oof_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        x_tr, y_tr = x_train.iloc[trn_idx], y_train[trn_idx]
        x_va, y_va = x_train.iloc[val_idx], y_train[val_idx]
        x_te = x_test.copy()

        x_tr, x_va, x_te = apply_fold_target_encoding(x_tr, y_tr, x_va, x_te, cat_cols, seed=SEED + fold)

        model = XGBClassifier(**params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        oof_preds[val_idx] = model.predict_proba(x_va)[:, 1]
        test_preds += model.predict_proba(x_te)[:, 1] / len(folds)

    return oof_preds, test_preds


def run_lightgbm_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs a standard multi-fold LightGBM model."""
    oof_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        x_tr, y_tr = x_train.iloc[trn_idx], y_train[trn_idx]
        x_va, y_va = x_train.iloc[val_idx], y_train[val_idx]
        x_te = x_test.copy()

        x_tr, x_va, x_te = apply_fold_target_encoding(x_tr, y_tr, x_va, x_te, cat_cols, seed=SEED + fold)

        model = lgb.LGBMClassifier(**params)
        model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)])
        oof_preds[val_idx] = model.predict_proba(x_va)[:, 1]
        test_preds += model.predict_proba(x_te)[:, 1] / len(folds)

    return oof_preds, test_preds


def run_pseudo_student(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    test_pseudo_probs: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cat_cols: List[str],
    model_type: str = "lgb",
    threshold: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-training distillation runner using confident pseudo-labels from test data.
    """
    conf_pos = test_pseudo_probs >= threshold
    conf_neg = test_pseudo_probs <= (1.0 - threshold)
    conf_mask = conf_pos | conf_neg

    if not np.any(conf_mask):
        print("Pseudo-Student: No pseudo labels met confidence threshold.")
        if model_type == "lgb":
            return run_lightgbm_model(x_train, y_train, x_test, folds, cat_cols, dict(n_estimators=500, verbose=-1))
        return run_catboost_model(x_train, y_train, x_test, folds, cat_cols, dict(iterations=500, verbose=0))

    pseudo_x = x_test[conf_mask].copy()
    pseudo_y = np.where(conf_pos[conf_mask], 1, 0)
    print(f"Pseudo-Student: Augmented train set with {len(pseudo_x)} high-confidence test samples.")

    oof_preds = np.zeros(len(x_train))
    test_preds = np.zeros(len(x_test))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        x_tr = pd.concat([x_train.iloc[trn_idx], pseudo_x], ignore_index=True)
        y_tr = np.concatenate([y_train[trn_idx], pseudo_y])
        x_va = x_train.iloc[val_idx]
        x_te = x_test.copy()

        x_tr, x_va, x_te = apply_fold_target_encoding(x_tr, y_tr, x_va, x_te, cat_cols, seed=SEED + fold)

        if model_type == "lgb":
            m = lgb.LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                random_state=SEED + fold,
                seed=SEED + fold,
                bagging_seed=SEED + fold + 11,
                feature_fraction_seed=SEED + fold + 22,
                extra_seed=SEED + fold + 33,
                data_random_seed=SEED + fold + 44,
                deterministic=True,
                force_col_wise=True,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            m = CatBoostClassifier(iterations=600, learning_rate=0.03, random_seed=SEED + fold, verbose=0)
            if HAS_GPU:
                m.set_params(task_type="GPU")

        m.fit(x_tr, y_tr)
        oof_preds[val_idx] = m.predict_proba(x_va)[:, 1]
        test_preds += m.predict_proba(x_te)[:, 1] / len(folds)

    return oof_preds, test_preds
