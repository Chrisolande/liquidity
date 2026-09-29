"""
Comprehensive feature selection pipeline powered by feature_engine.selection:
1. DropConstantFeatures: filters quasi-constant & zero-variance signals (tol=0.995).
2. DropDuplicateFeatures: strips exact duplicate representations across engineered columns.
3. SmartCorrelatedSelection: clusters collinear features (r > 0.98) and selects the one
   with the highest correlation to the target (corr_with_target).
4. 5-Fold Cross-Validated LightGBM Gain Importance: ranks features and selects top K features.
5. Guaranteed preservation of domain anchors (CHAMPION_14, MACRO_SOLVENCY_10, MACRO_TRIAGE_10, core categoricals).
"""

from typing import List, Tuple, Set, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

try:
    from feature_engine.selection import (
        DropConstantFeatures,
        DropDuplicateFeatures,
        SmartCorrelatedSelection,
    )
except ImportError:
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "feature-engine"])
        from feature_engine.selection import (
            DropConstantFeatures,
            DropDuplicateFeatures,
            SmartCorrelatedSelection,
        )
    except Exception:
        DropConstantFeatures = None
        DropDuplicateFeatures = None
        SmartCorrelatedSelection = None


from src.config import CHAMPION_14, MACRO_SOLVENCY_10, MACRO_TRIAGE_10, SEED, TARGET, ID_COL


def get_protected_features(all_columns: List[str], cat_cols: List[str]) -> List[str]:
    """Returns domain anchors and categorical columns that should never be dropped."""
    protected_set = set(cat_cols + CHAMPION_14 + MACRO_SOLVENCY_10 + MACRO_TRIAGE_10)
    return [c for c in all_columns if c in protected_set]


def filter_with_feature_engine(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    numeric_candidates: List[str],
    corr_threshold: float = 0.98,
    quasi_constant_tol: float = 0.995,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Applies feature_engine transformers on candidate numeric features:
    1. DropConstantFeatures
    2. DropDuplicateFeatures
    3. SmartCorrelatedSelection (corr_with_target)
    """
    if not numeric_candidates:
        return X_train, X_test, []

    df_tr_num = X_train[numeric_candidates].copy()
    df_te_num = X_test[[c for c in numeric_candidates if c in X_test.columns]].copy()

    # Step 1: DropConstantFeatures
    dcf = DropConstantFeatures(tol=quasi_constant_tol, missing_values="ignore")
    df_tr_num = dcf.fit_transform(df_tr_num)
    df_te_num = dcf.transform(df_te_num)
    dropped_const = dcf.features_to_drop_
    print(f"feature_engine.DropConstantFeatures: dropped {len(dropped_const)} features", flush=True)

    # Step 2: DropDuplicateFeatures
    ddf = DropDuplicateFeatures(missing_values="ignore")
    df_tr_num = ddf.fit_transform(df_tr_num)
    df_te_num = ddf.transform(df_te_num)
    dropped_dup = ddf.features_to_drop_
    print(f"feature_engine.DropDuplicateFeatures: dropped {len(dropped_dup)} features", flush=True)

    # Step 3: SmartCorrelatedSelection with correlation to target
    scs = SmartCorrelatedSelection(
        variables=list(df_tr_num.columns),
        method="pearson",
        threshold=corr_threshold,
        selection_method="corr_with_target",
        missing_values="ignore",
    )
    df_tr_num = scs.fit_transform(df_tr_num, y_train)
    df_te_num = scs.transform(df_te_num)
    dropped_corr = scs.features_to_drop_
    print(f"feature_engine.SmartCorrelatedSelection (r>{corr_threshold}): dropped {len(dropped_corr)} features", flush=True)

    retained_numeric = list(df_tr_num.columns)
    return df_tr_num, df_te_num, retained_numeric


def select_features_cv_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: List[str],
    cat_cols: List[str],
    k_top: int = 200,
    n_splits: int = 5,
    seed: int = 42,
) -> List[str]:
    """
    Ranks retained numeric features using 5-fold cross-validated LightGBM gain importance.
    Selects the top k_top features and guarantees retention of domain anchors.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    importances = np.zeros(len(numeric_cols), dtype=float)

    for trn_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            colsample_bytree=0.7,
            subsample=0.8,
            subsample_freq=1,
            importance_type="gain",
            random_state=seed,
            seed=seed,
            bagging_seed=seed + 11,
            feature_fraction_seed=seed + 22,
            extra_seed=seed + 33,
            data_random_seed=seed + 44,
            deterministic=True,
            force_col_wise=True,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X.iloc[trn_idx][numeric_cols], y[trn_idx])
        importances += model.feature_importances_ / n_splits

    rankings = pd.Series(importances, index=numeric_cols).sort_values(ascending=False)
    top_num = rankings.head(k_top).index.tolist()

    protected = get_protected_features(list(X.columns), cat_cols)
    final_selected = sorted(list(set(top_num + protected)))
    return final_selected


def run_feature_engine_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    k_top: int = 200,
    corr_threshold: float = 0.98,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    End-to-end feature selection pipeline using feature_engine + CV LightGBM importance:
    1. Identifies and isolates domain anchors / categorical columns.
    2. Runs DropConstantFeatures, DropDuplicateFeatures, and SmartCorrelatedSelection on numeric features.
    3. Runs 5-fold CV importance ranking to retain top k_top numeric features.
    4. Merges protected domain anchors back into the dataset.
    """
    print(f"--- Feature Selection: Initial count = {X_train.shape[1]} features ---", flush=True)

    protected_cols = get_protected_features(list(X_train.columns), cat_cols)
    numeric_candidates = [
        c
        for c in X_train.columns
        if c not in protected_cols and pd.api.types.is_numeric_dtype(X_train[c])
    ]

    # Step 1-3: feature_engine filtering
    _, _, retained_num = filter_with_feature_engine(
        X_train,
        y_train,
        X_test,
        numeric_candidates,
        corr_threshold=corr_threshold,
    )

    # Step 4: 5-fold CV LightGBM importance selection
    selected_cols = select_features_cv_importance(
        X_train,
        y_train,
        numeric_cols=retained_num,
        cat_cols=cat_cols,
        k_top=k_top,
        n_splits=5,
        seed=seed,
    )

    print(
        f"--- Feature Selection Complete: {len(selected_cols)} features selected (out of {X_train.shape[1]}) ---",
        flush=True,
    )
    return X_train[selected_cols].copy(), X_test[selected_cols].copy(), selected_cols
