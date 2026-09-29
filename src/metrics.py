"""
Competition metrics, statistical hypothesis testing, and model diagnostics.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from src.config import FLOOR, CEIL, EPS


def competition_score(
    y_true: np.ndarray,
    prob_logloss: np.ndarray,
    prob_rauc: Optional[np.ndarray] = None,
    floor: float = FLOOR,
    ceil: float = CEIL,
) -> Tuple[float, float, float]:
    """
    Computes official AI4EAC Liquidity Stress Competition Metric:
      Metric = 0.40 * AUC + 0.60 * (1.0 - (LogLoss / 0.595))
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_ll = np.clip(np.asarray(prob_logloss, dtype=float), floor, ceil)
    p_rauc = p_ll if prob_rauc is None else np.clip(np.asarray(prob_rauc, dtype=float), floor, ceil)

    ll = float(log_loss(y_arr, p_ll))
    auc = float(roc_auc_score(y_arr, p_rauc))
    norm_ll = ll / 0.595
    comp = float((0.40 * auc) + (0.60 * (1.0 - norm_ll)))
    return ll, auc, comp


def comp_metric_eval(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Convenience wrapper returning solely the final composite competition score."""
    _, _, comp = competition_score(y_true, y_pred)
    return comp


def paired_bootstrap(
    y_true: np.ndarray,
    p_new: np.ndarray,
    p_old: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Performs paired bootstrap hypothesis test comparing two sets of predictions.
    Returns:
      (prob_new_wins, mean_delta_comp, p_value_ll)
    """
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true, dtype=int)
    n = len(y_arr)
    pn = np.clip(p_new, FLOOR, CEIL)
    po = np.clip(p_old, FLOOR, CEIL)

    ll_diffs = np.zeros(n_boot)
    comp_diffs = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        pnb, pob = pn[idx], po[idx]
        ll_new, auc_new, c_new = competition_score(yb, pnb)
        ll_old, auc_old, c_old = competition_score(yb, pob)
        ll_diffs[b] = ll_old - ll_new
        comp_diffs[b] = c_new - c_old

    p_wins = float(np.mean(comp_diffs > 0))
    mean_delta = float(np.mean(comp_diffs))
    p_val_ll = float(np.mean(ll_diffs <= 0))
    return p_wins, mean_delta, p_val_ll


def print_correlation_matrix(oof_dict: Dict[str, np.ndarray]) -> None:
    """Computes pairwise correlation between model OOF predictions and flags redundant pairs."""
    names = list(oof_dict.keys())
    n = len(names)
    if n < 2:
        return
    print("OOF Prediction Correlation Matrix:")
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = float(np.corrcoef(oof_dict[names[i]], oof_dict[names[j]])[0, 1])

    hdr = f"{'Model':<20}" + "".join(f"{name[:10]:>12}" for name in names)
    print(hdr)
    for i in range(n):
        row = f"{names[i]:<20}" + "".join(f"{corr[i, j]:>12.4f}" for j in range(n))
        print(row)

    flagged = False
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > 0.985:
                print(f"High correlation: {names[i]} and {names[j]} ({corr[i, j]:.4f})")
                flagged = True
    if not flagged:
        print("All models are sufficiently diversified (< 0.985).")


def print_segment_loss_table(
    y_true: np.ndarray,
    oof_dict: Dict[str, np.ndarray],
    segments: Dict[str, np.ndarray],
) -> None:
    """Prints log-loss breakdown per segment/cohort to identify where models excel or struggle."""
    print("Segment Log-Loss Breakdown:")
    hdr = f"{'Segment':<20}{'Size':>8}" + "".join(f"{m[:10]:>12}" for m in oof_dict)
    print(hdr)
    for s_name, mask in segments.items():
        if not np.any(mask):
            continue
        y_seg = y_true[mask]
        if len(np.unique(y_seg)) < 2:
            continue
        row = f"{s_name:<20}{int(np.sum(mask)):>8}"
        for m_name, preds in oof_dict.items():
            ll = float(log_loss(y_seg, np.clip(preds[mask], FLOOR, CEIL)))
            row += f"{ll:>12.5f}"
        print(row)
