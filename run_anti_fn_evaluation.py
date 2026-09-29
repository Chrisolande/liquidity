"""
Run Anti-False-Negative Liquidity Feature Augmentation, 10-Fold CV Training, and Comparative Error Analysis.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.domain import engineer_anti_fn_liquidity_features
from src.baseline import run_baseline
from src.metrics import competition_score


def main():
    print("=" * 80)
    print("STEP 1: AUGMENTING CACHED FEATURES WITH ANTI-FN LIQUIDITY SIGNALS")
    print("=" * 80, flush=True)

    cache_dir = Path("checkpoints/feature_cache")
    tr_path = cache_dir / "train_fe.parquet"
    te_path = cache_dir / "test_fe.parquet"

    if tr_path.exists() and te_path.exists():
        print(f"Loading existing cache from {cache_dir}...", flush=True)
        train_fe = pd.read_parquet(tr_path)
        test_fe = pd.read_parquet(te_path)
        print(f"Initial cache shape: Train={train_fe.shape}, Test={test_fe.shape}", flush=True)

        # Apply Anti-FN features
        train_fe = engineer_anti_fn_liquidity_features(train_fe)
        test_fe = engineer_anti_fn_liquidity_features(test_fe)
        print(f"Augmented cache shape: Train={train_fe.shape}, Test={test_fe.shape}", flush=True)

        # Save back to cache
        train_fe.to_parquet(tr_path, index=False)
        test_fe.to_parquet(te_path, index=False)
        print(f"Saved augmented features back to {cache_dir}", flush=True)
    else:
        print("Feature cache not found; baseline pipeline will compute from scratch.", flush=True)

    print("\n" + "=" * 80)
    print("STEP 2: RUNNING 10-FOLD CV STAGE 1 WITH ANTI-FN FEATURES & SPW RESTORATION")
    print("=" * 80, flush=True)

    t0 = time.time()
    res = run_baseline(
        seeds=(42,),
        n_splits=10,
        k_top_features=140,
        output_dir="checkpoints",
        sub_dir="submissions",
    )
    t_elapsed = time.time() - t0
    print(f"\nTraining pipeline completed in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min)", flush=True)

    print("\n" + "=" * 80)
    print("STEP 3: RUNNING DEEP ERROR ANALYSIS ON NEW OUT-OF-FOLD PREDICTIONS")
    print("=" * 80, flush=True)

    y_true = np.load("checkpoints/y_true.npy")
    new_oof = np.load("checkpoints/oof_champ_train.npy")
    n = len(y_true)

    new_ll, new_auc, new_comp = competition_score(y_true, new_oof)
    print(f"New Champion OOF: Comp={new_comp:.5f} | AUC={new_auc:.5f} | LL={new_ll:.5f}", flush=True)

    eps_clip = 1e-15
    p_clipped = np.clip(new_oof, eps_clip, 1.0 - eps_clip)
    individual_ll = -(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))

    total_loss_mass = float(individual_ll.sum())
    top_1pct_idx = np.argsort(individual_ll)[-int(n * 0.01):]
    top_5pct_idx = np.argsort(individual_ll)[-int(n * 0.05):]

    loss_share_1pct = float(individual_ll[top_1pct_idx].sum() / total_loss_mass * 100)
    loss_share_5pct = float(individual_ll[top_5pct_idx].sum() / total_loss_mass * 100)

    # Class breakdown
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    pos_loss_mass = float(individual_ll[pos_mask].sum())
    neg_loss_mass = float(individual_ll[neg_mask].sum())
    mean_pos_ll = float(individual_ll[pos_mask].mean())
    mean_neg_ll = float(individual_ll[neg_mask].mean())

    # FN: True positive predicted p < 0.15
    fn_mask = pos_mask & (new_oof < 0.15)
    fn_count = int(fn_mask.sum())
    fn_loss_mass = float(individual_ll[fn_mask].sum())
    fn_loss_share = float(fn_loss_mass / total_loss_mass * 100)

    # FP: True negative predicted p > 0.50
    fp_mask = neg_mask & (new_oof > 0.50)
    fp_count = int(fp_mask.sum())
    fp_loss_mass = float(individual_ll[fp_mask].sum())
    fp_loss_share = float(fp_loss_mass / total_loss_mass * 100)

    new_metrics = {
        "comp": float(new_comp),
        "auc": float(new_auc),
        "logloss": float(new_ll),
        "total_loss_mass": total_loss_mass,
        "loss_share_1pct": loss_share_1pct,
        "loss_share_5pct": loss_share_5pct,
        "pos_loss_mass": pos_loss_mass,
        "neg_loss_mass": neg_loss_mass,
        "mean_pos_ll": mean_pos_ll,
        "mean_neg_ll": mean_neg_ll,
        "fn_count": fn_count,
        "fn_loss_mass": fn_loss_mass,
        "fn_loss_share": fn_loss_share,
        "fp_count": fp_count,
        "fp_loss_mass": fp_loss_mass,
        "fp_loss_share": fp_loss_share,
    }

    # Load old investigation metrics if available
    old_file = "checkpoints/deep_error_investigation.json"
    old_metrics = None
    if os.path.exists(old_file):
        try:
            with open(old_file, "r") as f:
                old_data = json.load(f)
                old_metrics = old_data.get("loss_profile", {})
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("COMPARATIVE FORENSIC REPORT: BEFORE vs AFTER ANTI-FN INTERVENTIONS")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Baseline (Before)':<20} | {'Anti-FN (After)':<20} | {'Delta':<15}")
    print("-" * 95)
    print(f"{'Composite Score':<35} | {0.73409:<20.5f} | {new_comp:<20.5f} | {new_comp - 0.73409:+<15.5f}")
    print(f"{'ROC AUC':<35} | {0.91968:<20.5f} | {new_auc:<20.5f} | {new_auc - 0.91968:+<15.5f}")
    print(f"{'Mean LogLoss':<35} | {0.23184:<20.5f} | {new_ll:<20.5f} | {new_ll - 0.23184:+<15.5f}")
    print(f"{'Total Loss Mass':<35} | {9273.41:<20.1f} | {total_loss_mass:<20.1f} | {total_loss_mass - 9273.41:+<15.1f}")
    print(f"{'Top 5% Worst Loss Share (%)':<35} | {47.09:<20.2f} | {loss_share_5pct:<20.2f} | {loss_share_5pct - 47.09:+<15.2f}%")
    print(f"{'False Negative Count (p < 0.15)':<35} | {986:<20d} | {fn_count:<20d} | {fn_count - 986:+<15d}")
    print(f"{'False Negative Loss Share (%)':<35} | {29.49:<20.2f} | {fn_loss_share:<20.2f} | {fn_loss_share - 29.49:+<15.2f}%")
    print(f"{'Mean LogLoss on Positive Class':<35} | {0.9582:<20.4f} | {mean_pos_ll:<20.4f} | {mean_pos_ll - 0.9582:+<15.4f}")
    print("=" * 95)

    output_path = "checkpoints/deep_error_investigation_anti_fn.json"
    with open(output_path, "w") as f:
        json.dump(new_metrics, f, indent=2)
    print(f"Saved complete comparative error analysis to {output_path}", flush=True)


if __name__ == "__main__":
    main()
