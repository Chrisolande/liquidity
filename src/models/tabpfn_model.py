"""
TabPFN foundation prior model runners with RankGauss Quantile transformation.
Zero fallback: executes genuine TabPFN v2 foundation model on GPU with Kaggle secrets authentication.
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from src.config import HAS_GPU, NUM_GPUS
from src.metrics import competition_score


def ensure_tabpfn_auth() -> None:
    """Authenticates TabPFN with PriorLabs API token from Kaggle secrets or environment."""
    if "TABPFN_TOKEN" not in os.environ:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            for key in ["TABPFN_TOKEN", "TABPFN_TOKEN2", "HF_TOKEN"]:
                try:
                    tok = user_secrets.get_secret(key)
                    if tok:
                        os.environ["TABPFN_TOKEN"] = tok
                        print(f"Successfully authenticated TabPFN via Kaggle secret {key}.", flush=True)
                        break
                except Exception:
                    pass
        except Exception:
            pass

    if "TABPFN_TOKEN" in os.environ:
        masked = os.environ["TABPFN_TOKEN"][:8] + "..." if len(os.environ["TABPFN_TOKEN"]) > 10 else "***"
        print(f"TabPFN authentication token detected ({masked}).", flush=True)
    else:
        print("[INFO] No TABPFN_TOKEN detected in environment or Kaggle secrets. TabPFN will run with standard local weights.", flush=True)



def fit_tabpfn_multi_view(
    X_train: pd.DataFrame,
    y_true: np.ndarray,
    X_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    pfn_views: Dict[str, List[str]],
    n_estimators: int = 4,
    batch_size: int = 5000,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Fits genuine foundation TabPFN priors over specialized feature sub-views on GPU.
    No fallback: enforces execution of real TabPFN.
    """
    ensure_tabpfn_auth()

    try:
        from tabpfn import TabPFNClassifier
    except ImportError as e:
        raise ImportError(f"TabPFN package is required but not installed: {e}. Install via 'pip install tabpfn'.")

    n_train = len(X_train)
    n_test = len(X_test)
    n_splits = len(folds)
    oof_dict: Dict[str, np.ndarray] = {}
    test_dict: Dict[str, np.ndarray] = {}

    print(f"Executing Genuine TabPFN Foundation Priors on GPU across {len(pfn_views)} sub-views...", flush=True)

    for pfn_name, pfn_cols in pfn_views.items():
        t_v0 = time.time()
        avail_cols = [c for c in pfn_cols if c in X_train.columns]
        print(f"[{pfn_name}] Features ({len(avail_cols)}): {avail_cols}", flush=True)

        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=42)
        X_tr_norm = qt.fit_transform(X_train[avail_cols].select_dtypes(include="number").fillna(0)).astype(np.float32)
        X_te_norm = qt.transform(X_test[avail_cols].select_dtypes(include="number").fillna(0)).astype(np.float32)

        pfn_oof = np.zeros(n_train, dtype=float)
        pfn_test = np.zeros(n_test, dtype=float)

        for fold, (trn_idx, val_idx) in enumerate(folds):
            t_f = time.time()
            pfn_dev = f"cuda:{fold % NUM_GPUS}" if NUM_GPUS > 1 else "cuda:0"
            clf = TabPFNClassifier(
                device=pfn_dev,
                ignore_pretraining_limits=True,
                n_estimators=n_estimators,
                random_state=42 + fold,
                show_progress_bar=False,
            )
            clf.fit(X_tr_norm[trn_idx], y_true[trn_idx])

            # Predict on validation fold with chunking for memory efficiency
            va_p = np.zeros(len(val_idx), dtype=float)
            x_val = X_tr_norm[val_idx]
            for b_start in range(0, len(x_val), batch_size):
                b_end = min(b_start + batch_size, len(x_val))
                va_p[b_start:b_end] = clf.predict_proba(x_val[b_start:b_end])[:, 1]
            pfn_oof[val_idx] = va_p

            # Predict on test set with chunking
            te_p = np.zeros(n_test, dtype=float)
            for b_start in range(0, n_test, batch_size):
                b_end = min(b_start + batch_size, n_test)
                te_p[b_start:b_end] = clf.predict_proba(X_te_norm[b_start:b_end])[:, 1]
            pfn_test += te_p / n_splits

            fold_ll, fold_auc, fold_comp = competition_score(y_true[val_idx], va_p)
            print(f"  [{pfn_name}] Fold {fold+1}/{n_splits} | LL: {fold_ll:.4f} | AUC: {fold_auc:.4f} | Comp: {fold_comp:.4f} ({time.time() - t_f:.1f}s)", flush=True)

        oof_dict[pfn_name] = pfn_oof
        test_dict[pfn_name] = pfn_test
        ll, auc, comp = competition_score(y_true, pfn_oof)
        print(f"[{pfn_name}] Full OOF | LL: {ll:.5f} | AUC: {auc:.5f} | Comp: {comp:.5f} (Total {time.time() - t_v0:.1f}s)\n", flush=True)

    return oof_dict, test_dict
