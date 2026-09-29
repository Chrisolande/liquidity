"""
Categorical target encoding, domain subset extraction, and LightGBM-based feature screening.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from src.config import SEED, CHAMPION_14, MACRO_SOLVENCY_10, MACRO_TRIAGE_10, HAS_GPU


def apply_fold_target_encoding(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    x_te: pd.DataFrame,
    te_cols: List[str],
    smoothing: float = 20.0,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leak-free fold-isolated target encoding:
    - Train fold: Encoded via inner 5-fold cross-validation to prevent self-target overfitting.
    - Val & Test folds: Mapped using smoothed statistics learned strictly on the train fold.
    """
    x_tr = x_tr.copy()
    x_va = x_va.copy()
    x_te = x_te.copy()
    prior = float(y_tr.mean())
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for col in te_cols:
        if col not in x_tr.columns:
            continue
        te_col_name = f"{col}_te"
        x_tr[te_col_name] = prior

        for in_trn_idx, in_val_idx in inner_skf.split(x_tr, y_tr):
            in_tr_slice = x_tr.iloc[in_trn_idx]
            in_y_slice = y_tr[in_trn_idx]
            stats = (
                pd.DataFrame({col: in_tr_slice[col].astype(str), "target": in_y_slice})
                .groupby(col)["target"]
                .agg(["count", "sum"])
            )
            smooth_val = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
            smooth_dict = smooth_val.to_dict()
            mapped_vals = (
                x_tr.iloc[in_val_idx][col]
                .astype(str)
                .map(smooth_dict)
                .fillna(prior)
                .to_numpy(dtype=float)
            )
            x_tr.iloc[in_val_idx, x_tr.columns.get_loc(te_col_name)] = mapped_vals

        tr_stats = (
            pd.DataFrame({col: x_tr[col].astype(str), "target": y_tr})
            .groupby(col)["target"]
            .agg(["count", "sum"])
        )
        smooth_val_full = (tr_stats["sum"] + smoothing * prior) / (tr_stats["count"] + smoothing)
        smooth_full_dict = smooth_val_full.to_dict()
        x_va[te_col_name] = (
            x_va[col].astype(str).map(smooth_full_dict).fillna(prior).to_numpy(dtype=float)
        )
        x_te[te_col_name] = (
            x_te[col].astype(str).map(smooth_full_dict).fillna(prior).to_numpy(dtype=float)
        )

    return x_tr, x_va, x_te


def extract_domain_feature_subsets(
    all_cols: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """Extracts specialized physics, digital channels, momentum, and macro triage sub-views."""
    d1_physics = [c for c in CHAMPION_14 if c in all_cols]
    for extra in ["segment_te", "arpu", "agg_daily_avg_bal_newest_to_oldest_ratio", "agg_vol_shift_m1_m6"]:
        if extra in all_cols and extra not in d1_physics:
            d1_physics.append(extra)
    _digital_keys = ["channel", "digital", "cash", "merchant", "paybill", "bank"]
    _mom_keys = ["velocity", "drift", "activity_rate", "rate_r3", "turnover", "slope", "burn"]

    # Prioritize engineered aggregate / ratio features over raw monthly columns
    raw_prefixes = ("m1_", "m2_", "m3_", "m4_", "m5_", "m6_")
    _digital_candidates = [
        c for c in all_cols
        if any(k in c for k in _digital_keys) and c not in d1_physics and not c.endswith("_te")
    ]
    _digital_candidates.sort(key=lambda c: (1 if c.startswith(raw_prefixes) else 0, c))
    d2_digital = _digital_candidates[:15]

    _mom_candidates = [
        c for c in all_cols
        if any(k in c for k in _mom_keys)
        and c not in d1_physics
        and c not in d2_digital
        and not c.endswith("_te")
    ]
    _mom_candidates.sort(key=lambda c: (1 if c.startswith(raw_prefixes) else 0, c))
    d3_momentum = _mom_candidates[:15]

    d5_triage = [c for c in MACRO_TRIAGE_10 if c in all_cols and c not in d1_physics]

    dom2 = d1_physics + d2_digital
    dom3 = d1_physics + d2_digital + d3_momentum
    dom_triage = d1_physics + d5_triage
    return dom2, dom3, dom_triage


def screen_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
    k_top: int = 300,
    seed: int = 42,
) -> List[str]:
    """Fast feature screening using LightGBM gain importance on numeric columns."""
    num_cols = [
        c
        for c in X_train.columns
        if c not in cat_cols and c not in ["ID", "Target", "liquidity_stress_next_30d"]
    ]
    p_screener = dict(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=63,
        colsample_bytree=0.6,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=100,
        importance_type="gain",
        random_state=seed,
        verbose=-1,
    )
    m = lgb.LGBMClassifier(n_jobs=-1, **p_screener)
    m.fit(X_train[num_cols], y_train)

    gain_s = pd.Series(m.feature_importances_, index=num_cols)
    top_num = gain_s.sort_values(ascending=False).head(k_top).index.tolist()
    guaranteed = [
        c
        for c in X_train.columns
        if c in cat_cols or c in CHAMPION_14 or c in MACRO_SOLVENCY_10 or c in MACRO_TRIAGE_10
    ]
    selected = sorted(set(top_num + guaranteed))
    return selected
