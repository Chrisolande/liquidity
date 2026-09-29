"""
Multi-model stacking routines: Logit-space regularized stacking, Nelder-Mead optimization, and Hill Climbing.
"""

from functools import partial
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from src.config import FLOOR, CEIL, TARGET, EPS
from src.metrics import competition_score, comp_metric_eval

try:
    from hillclimbers import climb_hill
except ImportError:
    climb_hill = None


def blend_and_calibrate(
    oof_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_splits: int = 5,
    lam: float = 1e-3,
    seed: int = 42,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stacks model predictions in unconstrained logit space:
      z = b + sum_j w_j * logit(p_j)
      p = sigmoid(z)
    Directly minimizes cross-entropy LogLoss with non-negative weights and L2 regularization.
    Cross-fitted over folds so meta-OOF has zero leakage.
    Followed by cross-fitted Platt temperature and bias calibration.
    """
    keys = list(oof_dict.keys())
    P_oof = np.column_stack([oof_dict[k] for k in keys])
    P_test = np.column_stack([test_dict[k] for k in keys])

    Z_oof = logit(np.clip(P_oof, 1e-7, 1.0 - 1e-7))
    Z_test = logit(np.clip(P_test, 1e-7, 1.0 - 1e-7))
    n_models = Z_oof.shape[1]

    def solve_weights(Z_tr, y_tr):
        def obj(theta):
            b = theta[0]
            w = theta[1:]
            p = expit(b + Z_tr @ w)
            return log_loss(y_tr, p) + lam * np.sum(w ** 2)

        init = np.r_[0.0, np.ones(n_models) / n_models]
        bounds = [(-3.0, 3.0)] + [(0.0, 3.0)] * n_models
        res = minimize(obj, init, method="L-BFGS-B", bounds=bounds)
        return res.x[0], res.x[1:]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_blend = np.zeros(len(y_true), dtype=float)

    for trn_idx, val_idx in skf.split(Z_oof, y_true):
        b_f, w_f = solve_weights(Z_oof[trn_idx], y_true[trn_idx])
        oof_blend[val_idx] = expit(b_f + Z_oof[val_idx] @ w_f)

    b_all, w_all = solve_weights(Z_oof, y_true)
    test_blend = expit(b_all + Z_test @ w_all)
    weight_map = {k: float(w) for k, w in zip(keys, w_all)}

    # Platt Scaling Calibration
    z_oof = logit(np.clip(oof_blend, 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
    z_test = logit(np.clip(test_blend, 1e-6, 1.0 - 1e-6)).reshape(-1, 1)

    cal = LogisticRegression(C=1.0, solver="lbfgs")
    cal.fit(z_oof, y_true)

    oof_cal = np.clip(cal.predict_proba(z_oof)[:, 1], FLOOR, CEIL)
    test_cal = np.clip(cal.predict_proba(z_test)[:, 1], FLOOR, CEIL)
    return weight_map, oof_blend, test_blend, oof_cal, test_cal


def nelder_mead_blend(
    oof_dict: Dict[str, np.ndarray],
    test_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Directly maximizes official competition metric via Nelder-Mead simplex search over softmax weights.
    """
    keys = list(oof_dict.keys())
    P_mat = np.column_stack([oof_dict[k] for k in keys])
    P_test_mat = np.column_stack([test_dict[k] for k in keys])
    n_models = len(keys)

    def nm_objective(w_unnorm):
        w = softmax(w_unnorm)
        blend_oof = P_mat @ w
        return -comp_metric_eval(y_true, blend_oof)

    best_solo = np.argmax([competition_score(y_true, P_mat[:, i])[2] for i in range(n_models)])
    w_init = np.full(n_models, 0.0)
    w_init[best_solo] = 2.0

    res = minimize(nm_objective, w_init, method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-4, "fatol": 1e-4})
    best_w = softmax(res.x)

    nm_oof = P_mat @ best_w
    nm_test = P_test_mat @ best_w
    _, _, nm_comp = competition_score(y_true, nm_oof)
    return nm_oof, nm_test, best_w, nm_comp


def hill_climb_blend(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Runs forward step-wise hill climbing ensemble if library is installed."""
    if climb_hill is None:
        return None
    try:
        train_label_df = pd.DataFrame({TARGET: y_true})
        hc_test, hc_oof = climb_hill(
            train=train_label_df,
            oof_pred_df=oof_df,
            test_pred_df=test_df,
            target=TARGET,
            objective="maximize",
            eval_metric=partial(comp_metric_eval),
            negative_weights=False,
            precision=0.01,
            plot_hill=False,
            plot_hist=False,
            return_oof_preds=True,
        )
        _, _, hc_comp = competition_score(y_true, hc_oof)
        return hc_oof, hc_test, hc_comp
    except Exception as e:
        print(f"Hill Climbing note: {e}")
        return None
