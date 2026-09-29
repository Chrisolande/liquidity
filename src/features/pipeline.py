"""
End-to-end feature engineering pipeline integrating monthly, domain, stress, and Deotte interactions.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import TARGET, ID_COL
from src.features.monthly import (
    month_columns,
    add_monthly_summary_features,
    add_cross_feature_ratios,
)
from src.features.domain import (
    add_entropy_features,
    add_behavioral_shift_features,
    add_longitudinal_stress_features,
    add_liquidity_runway_and_exhaustion_features,
    add_solvency_and_burn_collapse_features,
    add_chris_deotte_features,
    engineer_anti_fn_liquidity_features,
)


def engineer_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    include_solvency: bool = True,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Transforms raw datasets into fully engineered tabular representations with 200+ features.
    Caches to parquet to eliminate redundant re-computation across pipeline stages.
    """
    from pathlib import Path
    cache_dir = Path("checkpoints/feature_cache")
    tr_cache = cache_dir / "train_fe.parquet"
    te_cache = cache_dir / "test_fe.parquet"

    if use_cache and tr_cache.exists() and te_cache.exists():
        try:
            train_fe = pd.read_parquet(tr_cache)
            test_fe = pd.read_parquet(te_cache)
            if len(train_fe) == len(train_raw) and len(test_fe) == len(test_raw):
                print(f"Loaded cached features from {cache_dir} ({train_fe.shape[1]} features)", flush=True)
                target_col = TARGET if TARGET in train_fe.columns else "Target"
                feature_cols = [c for c in train_fe.columns if c not in {target_col, ID_COL}]
                categorical_cols = [c for c in ["gender", "region", "smartphone", "segment", "earning_pattern"] if c in train_fe.columns]
                numeric_cols = [c for c in feature_cols if c not in categorical_cols]
                return train_fe, test_fe, categorical_cols, numeric_cols
        except Exception:
            pass

    combined = pd.concat(
        [
            train_raw.assign(_dataset="train"),
            test_raw.assign(_dataset="test"),
        ],
        ignore_index=True,
    )

    print("Extracting temporal summary features", flush=True)
    monthly_groups = month_columns(combined)
    combined = add_monthly_summary_features(combined, monthly_groups)

    print("Adding cross-feature transaction ratios", flush=True)
    combined = add_cross_feature_ratios(combined)

    print("Computing transaction entropy and diversity", flush=True)
    combined = add_entropy_features(combined)

    print("Computing behavioral shift and volatility metrics", flush=True)
    monthly_groups = month_columns(combined)
    combined = add_behavioral_shift_features(combined, monthly_groups)

    print("Engineering advanced stress features", flush=True)
    combined = add_longitudinal_stress_features(combined)

    print("Engineering liquidity runway and exhaustion indicators", flush=True)
    combined = add_liquidity_runway_and_exhaustion_features(combined)

    if include_solvency:
        print("Computing multi-month solvency and burn collapse metrics", flush=True)
        combined = add_solvency_and_burn_collapse_features(combined)

    print("Computing anti-FN liquidity & runway collapse features", flush=True)
    combined = engineer_anti_fn_liquidity_features(combined)

    print("Computing categorical interactions and frequency encodings", flush=True)
    combined, categorical_cols = add_chris_deotte_features(combined)

    for cat in categorical_cols:
        all_cats_unique = sorted(list(set(combined[cat].astype(str).dropna().unique())))
        cat_dtype = pd.CategoricalDtype(categories=all_cats_unique, ordered=False)
        combined[cat] = combined[cat].astype(str).astype(cat_dtype)

    train_fe = (
        combined.loc[combined["_dataset"] == "train"]
        .drop(columns=["_dataset"])
        .reset_index(drop=True)
    )
    test_fe = (
        combined.loc[combined["_dataset"] == "test"]
        .drop(columns=["_dataset"])
        .reset_index(drop=True)
    )

    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            train_fe.to_parquet(tr_cache, index=False)
            test_fe.to_parquet(te_cache, index=False)
            print(f"Saved feature cache to {cache_dir}", flush=True)
        except Exception:
            pass

    feature_cols = [c for c in train_fe.columns if c not in {TARGET, ID_COL}]
    numeric_cols = train_fe[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    return train_fe, test_fe, categorical_cols, numeric_cols
