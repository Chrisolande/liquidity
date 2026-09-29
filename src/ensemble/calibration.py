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
    """
    from sklearn.model_selection import StratifiedKFold

    z_oof = logit(np.clip(oof_prob, EPS, 1.0 - EPS)).reshape(-1, 1)
    z_test = logit(np.clip(test_prob, EPS, 1.0 - EPS)).reshape(-1, 1)

    # Cross-fit OOF calibration to ensure zero in-sample evaluation leakage
    oof_cal = np.zeros(len(y_true), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for trn_idx, val_idx in skf.split(z_oof, y_true):
        fold_cal = LogisticRegression(C=1.0, solver="lbfgs")
        fold_cal.fit(z_oof[trn_idx], y_true[trn_idx])
        oof_cal[val_idx] = fold_cal.predict_proba(z_oof[val_idx])[:, 1]
    oof_cal = np.clip(oof_cal, FLOOR, CEIL)

    # Full calibrator for unseen test predictions
    full_cal = LogisticRegression(C=1.0, solver="lbfgs")
    full_cal.fit(z_oof, y_true)
    test_cal = np.clip(full_cal.predict_proba(z_test)[:, 1], FLOOR, CEIL)
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
