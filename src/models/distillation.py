"""
Knowledge distillation and pseudo-label student routines.
Implements:
1. Temperature-sharpened soft-label knowledge distillation students (LightGBM & XGBoost regressors).
2. Multi-Strata student models trained with sample weights on soft pseudo-labels.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from catboost import CatBoostClassifier

from src.config import SEED, FLOOR, CEIL, HAS_GPU


def temperature_sharpen(probs: np.ndarray, temperature: float = 0.85) -> np.ndarray:
    """
    Applies temperature sharpening to soft probabilities:
    P_sharp = P^(1/T) / (P^(1/T) + (1-P)^(1/T)).
    """
    p_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    p_pow = p_clipped ** (1.0 / temperature)
    p_sharp = p_pow / (p_pow + (1.0 - p_clipped) ** (1.0 / temperature))
    return np.clip(p_sharp, FLOOR, CEIL)


def train_distillation_students(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    teacher_test_probs: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    temperature: float = 0.85,
    pseudo_weight: float = 0.10,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """
    Trains 10-fold self-training distillation students (LightGBM & XGBoost regressors)
    learning from soft teacher probability targets on the unlabeled test set.
    """
    n_train = len(X_train)
    n_test = len(X_test)
    n_folds = len(folds)

    p_sharp = temperature_sharpen(teacher_test_probs, temperature=temperature)

    oof_student_lgb = np.zeros(n_train, dtype=np.float64)
    test_student_lgb = np.zeros(n_test, dtype=np.float64)
    oof_student_xgb = np.zeros(n_train, dtype=np.float64)
    test_student_xgb = np.zeros(n_test, dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        X_fold_tr = np.vstack([X_train[trn_idx], X_test])
        y_fold_tr = np.r_[y_train[trn_idx], p_sharp]
        w_fold_tr = np.r_[
            np.ones(len(trn_idx), dtype=np.float32),
            np.full(n_test, pseudo_weight, dtype=np.float32),
        ]
        X_fold_val = X_train[val_idx]

        # Student 1: LightGBM Regressor with cross_entropy objective
        lgb_student = LGBMRegressor(
            objective="cross_entropy",
            n_estimators=900,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed + fold,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_student.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
        oof_student_lgb[val_idx] = np.clip(lgb_student.predict(X_fold_val), FLOOR, CEIL)
        test_student_lgb += np.clip(lgb_student.predict(X_test), FLOOR, CEIL) / float(n_folds)

        # Student 2: XGBoost Regressor with binary:logistic objective
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
            random_state=seed + fold,
            tree_method="hist",
            device="cuda" if HAS_GPU else "cpu",
            n_jobs=-1,
        )
        xgb_student.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
        oof_student_xgb[val_idx] = np.clip(xgb_student.predict(X_fold_val), FLOOR, CEIL)
        test_student_xgb += np.clip(xgb_student.predict(X_test), FLOOR, CEIL) / float(n_folds)

    return {
        "oof_lgb_student": oof_student_lgb,
        "test_lgb_student": test_student_lgb,
        "oof_xgb_student": oof_student_xgb,
        "test_xgb_student": test_student_xgb,
    }


def run_pseudo_student(
    xtr: pd.DataFrame,
    ytr: np.ndarray,
    xva: pd.DataFrame,
    yva: np.ndarray,
    xte: pd.DataFrame,
    y_pseudo: np.ndarray,
    cat_indices: List[int],
    cat_cols: List[str],
    model_type: str = "catboost",
    pseudo_weight: float = 0.6,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Strata pseudo-label student runner using CatBoost CrossEntropy or XGBoost binary:logistic.
    """
    X_aug = pd.concat([xtr, xte], ignore_index=True)
    y_aug = np.concatenate([ytr.astype(float), y_pseudo.astype(float)])
    weights = np.concatenate([np.ones(len(ytr)), np.full(len(y_pseudo), pseudo_weight)])

    if model_type == "catboost":
        task_type = "GPU" if HAS_GPU else "CPU"
        cb = CatBoostClassifier(
            loss_function="CrossEntropy",
            eval_metric="CrossEntropy",
            iterations=850,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=40.0,
            random_strength=1.5,
            bagging_temperature=0.3,
            border_count=128,
            task_type=task_type,
            random_seed=seed,
            verbose=False,
        )
        cb.fit(
            X_aug,
            y_aug,
            sample_weight=weights,
            eval_set=(xva, yva),
            cat_features=cat_indices,
            early_stopping_rounds=40,
            verbose=False,
        )
        val_prob = cb.predict_proba(xva)[:, 1]
        test_prob = cb.predict_proba(xte)[:, 1]
        return val_prob, test_prob

    elif model_type == "xgboost":
        X_aug_xgb = X_aug.copy()
        xva_xgb = xva.copy()
        xte_xgb = xte.copy()
        for c in cat_cols:
            if c in X_aug_xgb.columns:
                X_aug_xgb[c] = X_aug_xgb[c].astype("category").cat.codes
                xva_xgb[c] = xva_xgb[c].astype("category").cat.codes
                xte_xgb[c] = xte_xgb[c].astype("category").cat.codes

        dtrain = xgb.DMatrix(X_aug_xgb, label=y_aug, weight=weights, enable_categorical=True)
        dval = xgb.DMatrix(xva_xgb, label=yva, enable_categorical=True)
        dtest = xgb.DMatrix(xte_xgb, enable_categorical=True)

        params = {
            "tree_method": "hist",
            "device": "cuda" if HAS_GPU else "cpu",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": 0.035,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 5.0,
            "reg_alpha": 0.5,
            "seed": seed,
        }
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=800,
            evals=[(dval, "val")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        return bst.predict(dval), bst.predict(dtest)

    raise ValueError(f"Unsupported model_type: {model_type}")
