"""
Probability calibration routines: Platt Scaling (Logistic Regression) and Isotonic Regression.
"""

from typing import Tuple
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.config import FLOOR, CEIL, EPS


def platt_scaling_calibrate(
    oof_prob: np.ndarray,
    test_prob: np.ndarray,
    y_true: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """
    Cross-fitted univariate Platt scaling (temperature + bias calibration) on logit-transformed probabilities.
    OOF predictions are calibrated out-of-fold via StratifiedKFold to eliminate in-sample leakage.
    Test predictions are averaged across fold calibrators for maximum numerical stability.
    """
    from sklearn.model_selection import StratifiedKFold

    z_oof = logit(np.clip(oof_prob, EPS, 1.0 - EPS)).reshape(-1, 1)
    z_test = logit(np.clip(test_prob, EPS, 1.0 - EPS)).reshape(-1, 1)

    oof_cal = np.zeros(len(y_true), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_models = []
    for trn_idx, val_idx in skf.split(z_oof, y_true):
        fold_cal = LogisticRegression(C=1.0, solver="lbfgs")
        fold_cal.fit(z_oof[trn_idx], y_true[trn_idx])
        oof_cal[val_idx] = fold_cal.predict_proba(z_oof[val_idx])[:, 1]
        fold_models.append(fold_cal)
    oof_cal = np.clip(oof_cal, FLOOR, CEIL)

    test_preds = np.mean([m.predict_proba(z_test)[:, 1] for m in fold_models], axis=0)
    test_cal = np.clip(test_preds, FLOOR, CEIL)
    full_cal = fold_models[0]
    return oof_cal, test_cal, full_cal


def beta_calibrate(
    oof_prob: np.ndarray,
    test_prob: np.ndarray,
    y_true: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    C: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, LogisticRegression]:
    """
    Beta calibration (Kull, Silva Filho, Flach 2017).
    Fits a bivariate logistic regression on [ln(p), -ln(1-p)]:
        logit(q) = a * ln(p) - b * ln(1-p) + c
    Unlike isotonic regression which creates step plateaus that damage ROC-AUC ranking,
    Beta calibration is smooth, strictly monotonic, and models asymmetry in tree probabilities.
    OOF predictions are cross-fitted out-of-fold via StratifiedKFold.
    """
    from sklearn.model_selection import StratifiedKFold

    eps = 1e-6
    p_oof = np.clip(oof_prob, eps, 1.0 - eps)
    p_test = np.clip(test_prob, eps, 1.0 - eps)

    X_oof = np.column_stack([np.log(p_oof), -np.log(1.0 - p_oof)])
    X_test = np.column_stack([np.log(p_test), -np.log(1.0 - p_test)])

    oof_cal = np.zeros(len(y_true), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_models = []
    for trn_idx, val_idx in skf.split(X_oof, y_true):
        fold_cal = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
        fold_cal.fit(X_oof[trn_idx], y_true[trn_idx])
        oof_cal[val_idx] = fold_cal.predict_proba(X_oof[val_idx])[:, 1]
        fold_models.append(fold_cal)

    oof_cal = np.clip(oof_cal, FLOOR, CEIL)
    test_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in fold_models], axis=0)
    test_cal = np.clip(test_preds, FLOOR, CEIL)
    full_cal = fold_models[0]
    return oof_cal, test_cal, full_cal


def isotonic_calibrate(
    oof_prob: np.ndarray,
    test_prob: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, IsotonicRegression]:
    """
    Fits non-parametric isotonic regression to calibrate monotonic probability output.
    """
    iso = IsotonicRegression(out_of_bounds="clip", y_min=FLOOR, y_max=CEIL)
    iso.fit(oof_prob, y_true)

    oof_cal = np.clip(iso.predict(oof_prob), FLOOR, CEIL)
    test_cal = np.clip(iso.predict(test_prob), FLOOR, CEIL)
    return oof_cal, test_cal, iso

