"""
Deep LogLoss Error Forensic & High-Impact Intervention Lab (Comprehensive Analysis Edition).
Expansions included:
1. Exact Pointwise LogLoss Contribution Breakdown (identifying top penalty outliers).
2. Segment x Earning Pattern Cross-Cohort Risk & Base-Rate Drift Matrix.
3. Feature Contrast: Missing Stress Signatures in False Negatives vs. True Negatives.
4. Non-Linear Logit Tail Penalty Simulation (asymmetric margin penalty & clipping floor).
5. Leak-Free Out-of-Fold Segment-Conditioned Calibration (Nelder-Mead cross-fitted).
6. scale_pos_weight Sweep with Strict Odds Inversion on GPU.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

from src.config import EPS, SEED
from src.features.selection import run_feature_engine_selection
from src.metrics import competition_score


def run_deep_investigation():
    print("=" * 90)
    print("DEEP LOGLOSS FORENSIC & INTERVENTION LAB (FULL MULTI-LEVEL ANALYSIS)")
    print("=" * 90, flush=True)

    y_true = np.load("checkpoints/y_true.npy")
    oof_preds = np.load("checkpoints/oof_champ_train.npy")
    n = len(y_true)

    base_ll, base_auc, base_comp = competition_score(y_true, oof_preds)
    print(
        f"\n[Baseline Benchmark] Comp: {base_comp:.5f} | AUC: {base_auc:.5f} | LL: {base_ll:.5f}\n",
        flush=True,
    )

    # -------------------------------------------------------------
    # 1. POINTWISE LOGLOSS DECOMPOSITION & LOSS ASYMMETRY
    # -------------------------------------------------------------
    eps_clip = 1e-15
    p_clipped = np.clip(oof_preds, eps_clip, 1.0 - eps_clip)
    individual_ll = -(
        y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped)
    )

    total_loss_mass = individual_ll.sum()
    top_1pct_idx = np.argsort(individual_ll)[-int(n * 0.01) :]
    top_5pct_idx = np.argsort(individual_ll)[-int(n * 0.05) :]

    loss_share_1pct = individual_ll[top_1pct_idx].sum() / total_loss_mass * 100
    loss_share_5pct = individual_ll[top_5pct_idx].sum() / total_loss_mass * 100

    print("-" * 70)
    print("SECTION 1: POINTWISE LOGLOSS ASYMMETRY & TAIL CONCENTRATION")
    print("-" * 70)
    print(f"Total LogLoss Mass: {total_loss_mass:.2f}")
    print(
        f"Top 1% worst accounts (n={len(top_1pct_idx)}) drive: {loss_share_1pct:.2f}% of entire LogLoss"
    )
    print(
        f"Top 5% worst accounts (n={len(top_5pct_idx)}) drive: {loss_share_5pct:.2f}% of entire LogLoss"
    )

    fn_mask = (y_true == 1) & (oof_preds < 0.15)
    tn_mask = (y_true == 0) & (oof_preds < 0.15)
    fp_mask = (y_true == 0) & (oof_preds > 0.50)
    tp_mask = (y_true == 1) & (oof_preds > 0.50)

    fn_loss_share = individual_ll[fn_mask].sum() / total_loss_mass * 100
    fp_loss_share = individual_ll[fp_mask].sum() / total_loss_mass * 100

    print(f"\nCohort Penalty Distribution:")
    print(
        f"  Catastrophic False Negatives (y=1, p < 0.15): {fn_mask.sum():5d} accounts | Mean LL: {individual_ll[fn_mask].mean():.4f} | Loss Share: {fn_loss_share:.2f}%"
    )
    print(
        f"  Catastrophic False Positives (y=0, p > 0.50): {fp_mask.sum():5d} accounts | Mean LL: {individual_ll[fp_mask].mean():.4f} | Loss Share: {fp_loss_share:.2f}%"
    )
    print(
        f"  Clean True Negatives         (y=0, p < 0.15): {tn_mask.sum():5d} accounts | Mean LL: {individual_ll[tn_mask].mean():.4f}"
    )
    print(
        f"  Confirmed True Positives     (y=1, p > 0.50): {tp_mask.sum():5d} accounts | Mean LL: {individual_ll[tp_mask].mean():.4f}"
    )

    # -------------------------------------------------------------
    # 2. CROSS-COHORT ERROR AUDIT (SEGMENT x EARNING PATTERN)
    # -------------------------------------------------------------
    train_fe_path = "checkpoints/feature_cache/train_fe.parquet"
    if os.path.exists(train_fe_path):
        df_full = pd.read_parquet(train_fe_path)
        cohort_cols = [
            c
            for c in ["segment", "earning_pattern", "region"]
            if c in df_full.columns
        ]

        if "segment" in cohort_cols and "earning_pattern" in cohort_cols:
            print("\n" + "-" * 70)
            print(
                "SECTION 2: SEGMENT x EARNING PATTERN COHORT LOGLOSS BREAKDOWN"
            )
            print("-" * 70)
            df_full["_y"] = y_true
            df_full["_p"] = oof_preds
            df_full["_ll"] = individual_ll

            cohort_summary = (
                df_full.groupby(["segment", "earning_pattern"])
                .agg(
                    count=("_y", "count"),
                    base_rate=("_y", "mean"),
                    mean_pred=("_p", "mean"),
                    mean_ll=("_ll", "mean"),
                    total_ll=("_ll", "sum"),
                )
                .reset_index()
            )

            cohort_summary["loss_contrib_pct"] = (
                cohort_summary["total_ll"] / total_loss_mass * 100
            )
            cohort_summary["calibration_gap"] = (
                cohort_summary["mean_pred"] - cohort_summary["base_rate"]
            )
            cohort_summary = cohort_summary.sort_values(
                by="total_ll", ascending=False
            )

            print(
                f"{'Segment':<8s} | {'Earning Pattern':<16s} | {'N':<6s} | {'BaseRate':<8s} | {'PredMean':<8s} | {'MeanLL':<7s} | {'LossShare':<9s} | {'CalibGap':<8s}"
            )
            print("-" * 88)
            for _, row in cohort_summary.head(10).iterrows():
                print(
                    f"{str(row['segment']):<8s} | {str(row['earning_pattern']):<16s} | {int(row['count']):<6d} | "
                    f"{row['base_rate']:<8.3f} | {row['mean_pred']:<8.3f} | {row['mean_ll']:<7.4f} | "
                    f"{row['loss_contrib_pct']:<8.2f}% | {row['calibration_gap']:+8.4f}"
                )

    # -------------------------------------------------------------
    # 3. STATISTICAL FEATURE CONTRAST: WHAT SIGNALS ARE MISSING IN FN?
    # -------------------------------------------------------------
    diffs_sorted = []
    if os.path.exists(train_fe_path):
        print("\n" + "-" * 70)
        print("SECTION 3: STATISTICAL CONTRAST (FN vs. TN DISCRIMINATION)")
        print("-" * 70)

        numeric_cols = (
            df_full.select_dtypes(include=[np.number]).columns.tolist()
        )
        drop_cols = [
            "Target",
            "ID",
            "_dataset",
            "_y",
            "_p",
            "_ll",
            "fold",
            "cohort",
        ]
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]

        diffs = []
        for col in numeric_cols:
            fn_vals = df_full.loc[fn_mask, col].dropna()
            tn_vals = df_full.loc[tn_mask, col].dropna()
            if len(fn_vals) > 50 and len(tn_vals) > 50:
                fn_median = float(fn_vals.median())
                tn_median = float(tn_vals.median())
                iqr = float(
                    df_full[col].quantile(0.75) - df_full[col].quantile(0.25)
                )

                # Run Kolmogorov-Smirnov test to detect non-linear distribution separation
                ks_stat, _ = ks_2samp(
                    fn_vals.values[:3000], tn_vals.values[:3000]
                )

                if iqr > 1e-6:
                    norm_diff = (fn_median - tn_median) / iqr
                    diffs.append(
                        (col, norm_diff, ks_stat, fn_median, tn_median)
                    )

        diffs_sorted = sorted(diffs, key=lambda x: x[2], reverse=True)
        print(
            f"{'Feature Name':<45s} | {'KS Stat':<7s} | {'NormDiff':<9s} | {'FN Median':<10s} | {'TN Median':<10s}"
        )
        print("-" * 92)
        for col, diff, ks, fn_m, tn_m in diffs_sorted[:15]:
            print(
                f"{col:<45s} | {ks:<7.3f} | {diff:+9.3f} | {fn_m:<10.3f} | {tn_m:<10.3f}"
            )

    # -------------------------------------------------------------
    # 4. INTERVENTION 1: TAIL PROBABILITY FLOOR & MARGIN TUNING
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4: SIMULATION - ASYMMETRIC LOGIT FLOOR / MARGIN TUNING")
    print("=" * 70)

    # Evaluate whether setting a floor on low predictions prevents infinite LogLoss penalties
    floors_to_test = [1e-5, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    print("Sweeping probability floors across predictions:")
    for fl in floors_to_test:
        adj_p = np.clip(oof_preds, fl, 1.0 - fl)
        sim_ll, sim_auc, sim_comp = competition_score(y_true, adj_p)
        print(
            f"  Floor: {fl:<6.4f} -> Comp: {sim_comp:.5f} | AUC: {sim_auc:.5f} | LL: {sim_ll:.5f} (Delta: {sim_comp - base_comp:+.5f})"
        )

    # -------------------------------------------------------------
    # 5. INTERVENTION 2: LEAK-FREE SEGMENT-CONDITIONED CALIBRATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 5: EXPERIMENT - OUT-OF-FOLD SEGMENT CALIBRATION")
    print("=" * 70)

    calibrated_preds = oof_preds.copy()
    if os.path.exists(train_fe_path) and "segment" in df_full.columns:
        seg_series = df_full["segment"].astype(str)
        skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        for trn_cal_idx, val_cal_idx in skf_cal.split(oof_preds, y_true):
            for seg in seg_series.unique():
                trn_seg = (seg_series.iloc[trn_cal_idx] == seg).values
                val_seg = (seg_series.iloc[val_cal_idx] == seg).values

                if trn_seg.sum() < 50 or val_seg.sum() == 0:
                    continue

                y_fit = y_true[trn_cal_idx][trn_seg]
                p_fit = np.clip(
                    oof_preds[trn_cal_idx][trn_seg], 1e-6, 1.0 - 1e-6
                )

                def seg_loss(params):
                    temp, shift = params
                    logits = np.log(p_fit / (1.0 - p_fit))
                    adj = 1.0 / (
                        1.0 + np.exp(-(logits / max(temp, 0.05) + shift))
                    )
                    return -np.mean(
                        y_fit * np.log(adj + 1e-12)
                        + (1 - y_fit) * np.log(1 - adj + 1e-12)
                    )

                res = minimize(
                    seg_loss,
                    x0=[1.0, 0.0],
                    method="Nelder-Mead",
                    options={"maxiter": 200},
                )
                t_opt, s_opt = res.x

                val_p = np.clip(
                    oof_preds[val_cal_idx][val_seg], 1e-6, 1.0 - 1e-6
                )
                val_logits = np.log(val_p / (1.0 - val_p))
                calibrated_preds[val_cal_idx[val_seg]] = 1.0 / (
                    1.0 + np.exp(-(val_logits / max(t_opt, 0.05) + s_opt))
                )

        c_ll, c_auc, c_comp = competition_score(y_true, calibrated_preds)
        print(
            f"--> Out-of-Fold Segment Calibrated: Comp={c_comp:.5f} | AUC={c_auc:.5f} | LL={c_ll:.5f} (Delta: {c_comp - base_comp:+.5f})"
        )

    # -------------------------------------------------------------
    # 6. INTERVENTION 3: SCALE_POS_WEIGHT SWEEP (CATBOOST GPU)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 6: EXPERIMENT - scale_pos_weight SWEEP (CATBOOST GPU)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(skf.split(df_full, y_true))

    drop_cols = [
        c
        for c in [
            "liquidity_stress_next_30d",
            "Target",
            "ID",
            "_dataset",
            "_y",
            "_p",
            "_ll",
            "fold",
            "cohort",
        ]
        if c in df_full.columns
    ]
    X_raw = df_full.drop(columns=drop_cols, errors="ignore")
    cat_cols = ["segment", "region", "gender", "earning_pattern", "smartphone"]
    cat_cols_present = [c for c in cat_cols if c in X_raw.columns]

    # Expand feature budget to 140 to prevent starvation during test
    X_sel, _, selected_cols = run_feature_engine_selection(
        X_raw,
        y_true,
        X_raw.head(100),
        cat_cols=cat_cols_present,
        k_top=140,
        corr_threshold=0.98,
        seed=SEED,
    )
    active_cats = [c for c in cat_cols_present if c in selected_cols]
    for c in active_cats:
        X_sel[c] = X_sel[c].astype("category")
    cat_idx = [X_sel.columns.get_loc(c) for c in active_cats]

    weights_to_test = [1.0, 1.15, 1.30, 1.50]
    spw_results = []

    for spw in weights_to_test:
        cb_oof = np.zeros(n)
        for trn_idx, val_idx in splits:
            x_tr, y_tr = X_sel.iloc[trn_idx], y_true[trn_idx]
            x_va, y_va = X_sel.iloc[val_idx], y_true[val_idx]

            cb = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                iterations=700,
                learning_rate=0.045,
                depth=6,
                l2_leaf_reg=12.0,
                random_strength=1.2,
                bagging_temperature=0.3,
                scale_pos_weight=spw,
                border_count=128,
                task_type="GPU",
                random_seed=SEED,
                verbose=False,
            )
            cb.fit(
                x_tr,
                y_tr,
                eval_set=(x_va, y_va),
                cat_features=cat_idx,
                early_stopping_rounds=40,
                verbose=False,
            )

            raw_p = np.clip(cb.predict_proba(x_va)[:, 1], 1e-7, 1.0 - 1e-7)
            if spw != 1.0:
                odds = raw_p / (1.0 - raw_p)
                unweighted_p = odds / (odds + spw)
                cb_oof[val_idx] = unweighted_p
            else:
                cb_oof[val_idx] = raw_p

        s_ll, s_auc, s_comp = competition_score(y_true, cb_oof)
        spw_results.append((spw, s_comp, s_auc, s_ll))
        print(
            f"  scale_pos_weight={spw:<4.2f} -> Comp: {s_comp:.5f} | AUC: {s_auc:.5f} | LL: {s_ll:.5f} (Delta vs Bench: {s_comp - base_comp:+.5f})"
        )

    best_spw = max(spw_results, key=lambda x: x[1])

    # -------------------------------------------------------------
    # 7. EXPORT INVESTIGATION METRICS
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FORENSIC AUDIT SUMMARY")
    print(
        f"  Total Loss Concentration: Top 5% samples generate {loss_share_5pct:.2f}% of LogLoss."
    )
    print(
        f"  Best scale_pos_weight: {best_spw[0]:.2f} (Comp: {best_spw[1]:.5f}, Delta: {best_spw[1]-base_comp:+.5f})"
    )
    print("=" * 90, flush=True)

    summary_payload = {
        "loss_profile": {
            "total_loss_mass": float(total_loss_mass),
            "top_1pct_loss_share": float(loss_share_1pct),
            "top_5pct_loss_share": float(loss_share_5pct),
            "fn_count": int(fn_mask.sum()),
            "fn_loss_share": float(fn_loss_share),
            "fp_count": int(fp_mask.sum()),
            "fp_loss_share": float(fp_loss_share),
        },
        "top_features_discrepancy": [
            {
                "feature": col,
                "ks_stat": float(ks),
                "norm_diff": float(diff),
                "fn_median": float(fn_m),
                "tn_median": float(tn_m),
            }
            for col, diff, ks, fn_m, tn_m in diffs_sorted[:20]
        ],
        "spw_sweep": [
            {"spw": s, "comp": c, "auc": a, "logloss": l}
            for s, c, a, l in spw_results
        ],
        "best_intervention": {
            "scale_pos_weight": best_spw[0],
            "comp": best_spw[1],
            "auc": best_spw[2],
            "logloss": best_spw[3],
        },
    }

    with open("checkpoints/deep_error_investigation.json", "w") as f:
        json.dump(summary_payload, f, indent=2)


if __name__ == "__main__":
    run_deep_investigation()