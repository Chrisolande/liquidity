"""
Experimental TabPFN Multi-View Evaluation Script.
Standalone benchmark to test TabPFN v2 on expanded feature views:
1. pfn_phys (14 features - legacy baseline)
2. pfn_champ_legacy (20 features - legacy baseline)
3. pfn_solvency_runway (30 features - physics + solvency + cash-burn)
4. pfn_behavior_entropy (30 features - entropy + channel ratios + digital shifts)
5. pfn_sweetspot_50 (50 features - top feature_engine selected subset)

Evaluates on 5-fold Stratified CV with RankGauss QuantileTransformer,
benchmarks individual OOF scores, and tests blending against oof_champ (0.73528).
"""

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from src.config import (
    ID_COL,
    TARGET,
    CHAMPION_14,
    MACRO_SOLVENCY_10,
    MACRO_TRIAGE_10,
    SEED,
    N_SPLITS,
    FLOOR,
    CEIL,
    HAS_GPU,
    NUM_GPUS,
    get_default_dataset_paths,
)
from src.features.pipeline import engineer_features
from src.features.selection import run_feature_engine_selection
from src.models.tabpfn_model import ensure_tabpfn_auth
from src.ensemble.calibration import beta_calibrate
from src.metrics import competition_score

try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False


def evaluate_tabpfn_view(
    view_name: str,
    feature_cols: List[str],
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_te: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_estimators: int = 4,
    batch_size: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Evaluates one feature view with TabPFN on 5 folds."""
    avail_cols = [c for c in feature_cols if c in X_tr.columns]
    print(f"\n[{view_name}] Features: {len(avail_cols)}", flush=True)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", random_state=SEED)
    X_tr_norm = qt.fit_transform(X_tr[avail_cols].select_dtypes(include="number").fillna(0)).astype(np.float32)
    X_te_norm = qt.transform(X_te[avail_cols].select_dtypes(include="number").fillna(0)).astype(np.float32)

    n_train = len(X_tr)
    n_test = len(X_te)
    oof_preds = np.zeros(n_train, dtype=float)
    test_preds = np.zeros(n_test, dtype=float)

    from concurrent.futures import ThreadPoolExecutor

    def fit_single_fold(fold_tuple):
        fold, (trn_idx, val_idx) = fold_tuple
        t_f = time.time()
        gpu_id = (fold - 1) % NUM_GPUS
        pfn_dev = f"cuda:{gpu_id}"

        clf = TabPFNClassifier(
            device=pfn_dev,
            ignore_pretraining_limits=True,
            n_estimators=n_estimators,
            random_state=SEED + fold,
            show_progress_bar=False,
        )
        clf.fit(X_tr_norm[trn_idx], y_tr[trn_idx])

        # Validation prediction with memory-safe chunking
        va_p = np.zeros(len(val_idx), dtype=float)
        x_val = X_tr_norm[val_idx]
        for b_start in range(0, len(x_val), batch_size):
            b_end = min(b_start + batch_size, len(x_val))
            va_p[b_start:b_end] = clf.predict_proba(x_val[b_start:b_end])[:, 1]

        # Test prediction with chunking
        te_p = np.zeros(n_test, dtype=float)
        for b_start in range(0, n_test, batch_size):
            b_end = min(b_start + batch_size, n_test)
            te_p[b_start:b_end] = clf.predict_proba(X_te_norm[b_start:b_end])[:, 1]

        f_ll, f_auc, f_comp = competition_score(y_tr[val_idx], va_p)
        print(f"  [{pfn_dev}] Fold {fold}/{len(folds)} | Comp: {f_comp:.5f} | AUC: {f_auc:.5f} | LL: {f_ll:.5f} ({time.time() - t_f:.1f}s)", flush=True)
        return fold, val_idx, va_p, te_p

    t0 = time.time()
    max_workers = min(NUM_GPUS, len(folds))
    print(f"  --> Running {len(folds)} folds concurrently across {max_workers} GPUs (max_workers={max_workers})...", flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = list(executor.map(fit_single_fold, enumerate(folds, start=1)))

    for fold, val_idx, va_p, te_p in futures:
        oof_preds[val_idx] = va_p
        test_preds += te_p / len(folds)

    ll, auc, comp = competition_score(y_tr, oof_preds)
    print(f"==> {view_name} Full OOF: Comp={comp:.5f} | AUC={auc:.5f} | LL={ll:.5f} ({time.time() - t0:.1f}s)", flush=True)
    return oof_preds, test_preds, comp


def main():
    ensure_tabpfn_auth()

    if not HAS_TABPFN or not HAS_GPU:
        print("Error: TabPFN and CUDA GPU are required for this experiment!", flush=True)
        return

    train_path, test_path = get_default_dataset_paths()
    print("Loading data & extracting features...", flush=True)
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    target_col = TARGET if TARGET in train_raw.columns else "Target"
    y_true = train_raw[target_col].to_numpy(int)

    # 1. Feature Engineering & Selection
    X_train_feat, X_test_feat, cat_cols, _ = engineer_features(train_raw, test_raw)
    drop_meta = [c for c in [ID_COL, target_col] if c in X_train_feat.columns]
    X_tr = X_train_feat.drop(columns=drop_meta)
    X_te = X_test_feat.drop(columns=[c for c in [ID_COL] if c in X_test_feat.columns])

    # Feature Engine 60 selection
    X_tr_sel, X_te_sel, selected_cols = run_feature_engine_selection(
        X_tr, y_true, X_te, cat_cols=cat_cols, k_top=60, corr_threshold=0.98, seed=SEED
    )

    # 2. Build Experimental Views
    # Legacy Views
    phys_14 = [c for c in CHAMPION_14 if c in X_tr.columns]
    champ_legacy = phys_14 + [c for c in X_tr.columns if any(k in c for k in ["channel", "cash", "bank"])][:6]

    # Expanded Views
    solv_runway = list(dict.fromkeys(phys_14 + [c for c in MACRO_SOLVENCY_10 if c in X_tr.columns] + [c for c in X_tr.columns if any(k in c for k in ["runway", "exhaust", "burn", "cushion"])][:10]))
    behavior_entropy = list(dict.fromkeys(phys_14 + [c for c in MACRO_TRIAGE_10 if c in X_tr.columns] + [c for c in X_tr.columns if any(k in c for k in ["entropy", "drift", "velocity", "paybill", "merchant"])][:12]))
    sweetspot_50 = list(selected_cols[:50])

    views = {
        "pfn_phys_legacy": phys_14,
        "pfn_solvency_runway": solv_runway,
        "pfn_sweetspot_50": sweetspot_50,
    }

    print("=" * 80, flush=True)
    print("EXPERIMENTAL TABPFN MULTI-VIEW BENCHMARK (DUAL T4 GPU CONCURRENT)", flush=True)
    print(f"Views configured: {len(views)}", flush=True)
    for k, v in views.items():
        print(f"  • {k:<22}: {len(v)} features")
    print("=" * 80, flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_tr, y_true))

    results_oof: Dict[str, np.ndarray] = {}
    results_test: Dict[str, np.ndarray] = {}
    scores: Dict[str, float] = {}

    for name, cols in views.items():
        oof, test_p, comp = evaluate_tabpfn_view(
            view_name=name,
            feature_cols=cols,
            X_tr=X_tr,
            y_tr=y_true,
            X_te=X_te,
            folds=folds,
            n_estimators=4,
        )
        results_oof[name] = oof
        results_test[name] = test_p
        scores[name] = comp

        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80, flush=True)
    print("BENCHMARK SUMMARY ACROSS TABPFN VIEWS", flush=True)
    print("=" * 80, flush=True)
    for name, s in sorted(scores.items(), key=lambda x: -x[1]):
        ll, auc, comp = competition_score(y_true, results_oof[name])
        print(f"  {name:<22}: Comp={comp:.5f} | AUC={auc:.5f} | LL={ll:.5f}")

    # Test blending with anchor
    anchor_p = "checkpoints/oof_champ_train.npy"
    if os.path.exists(anchor_p):
        oof_anchor = np.load(anchor_p)
        a_ll, a_auc, a_comp = competition_score(y_true, oof_anchor)
        print(f"\nCurrent Champion Anchor: Comp={a_comp:.5f} | AUC={a_auc:.5f} | LL={a_ll:.5f}", flush=True)

        for name, oof in results_oof.items():
            # Test simple 90/10 blend
            blend_90_10 = 0.90 * oof_anchor + 0.10 * oof
            b_ll, b_auc, b_comp = competition_score(y_true, blend_90_10)
            diff = b_comp - a_comp
            print(f"  Blend Anchor + 10% {name:<22}: Comp={b_comp:.5f} ({diff:+.5f})")

    # Save experiment outputs
    os.makedirs("checkpoints/experiments", exist_ok=True)
    out_p = "checkpoints/experiments/tabpfn_view_sweep.npz"
    np.savez_compressed(out_p, **{f"oof_{k}": v for k, v in results_oof.items()})
    print(f"\nSaved benchmark checkpoint to {out_p}", flush=True)


if __name__ == "__main__":
    main()
