"""
Stage 1 LogLoss Error Analysis & Probability Calibration Diagnostic.
Identifies outlier concentration, cohort-specific logloss breakdown,
optimal clipping thresholds, and probability transforms to reduce LogLoss
and boost the competition composite score.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize_scalar

from src.metrics import competition_score


def run_error_analysis():
    print("=" * 75)
    print("STAGE 1 ERROR ANALYSIS & LOGLOSS REDUCTION DIAGNOSTIC")
    print("=" * 75)

    y_true = np.load("checkpoints/y_true.npy")
    oof_preds = np.load("checkpoints/oof_champ_train.npy")
    n = len(y_true)

    base_ll, base_auc, base_comp = competition_score(y_true, oof_preds)
    print(f"\nBaseline Anchor Performance:")
    print(f"  Composite Score : {base_comp:.5f}")
    print(f"  ROC-AUC         : {base_auc:.5f}")
    print(f"  LogLoss         : {base_ll:.5f}")
    print(f"  Base Rate (Mean): {y_true.mean():.4f}")
    print(f"  Mean Pred Prob  : {oof_preds.mean():.4f}")

    # 1. Pointwise Loss Calculation
    eps = 1e-15
    p_clipped = np.clip(oof_preds, eps, 1 - eps)
    pointwise_ll = -(y_true * np.log(p_clipped) + (1 - y_true) * np.log(1 - p_clipped))

    # 2. Outlier Loss Concentration
    sorted_losses = np.sort(pointwise_ll)[::-1]
    total_loss = np.sum(pointwise_ll)

    top_1pct_count = int(0.01 * n)
    top_2pct_count = int(0.02 * n)
    top_5pct_count = int(0.05 * n)

    pct_1 = np.sum(sorted_losses[:top_1pct_count]) / total_loss * 100
    pct_2 = np.sum(sorted_losses[:top_2pct_count]) / total_loss * 100
    pct_5 = np.sum(sorted_losses[:top_5pct_count]) / total_loss * 100

    print("\n--- Outlier Loss Concentration ---")
    print(f"  Top 1% worst errors (400 samples) : {pct_1:.2f}% of total LogLoss!")
    print(f"  Top 2% worst errors (800 samples) : {pct_2:.2f}% of total LogLoss!")
    print(f"  Top 5% worst errors (2000 samples): {pct_5:.2f}% of total LogLoss!")

    # 3. Asymmetric Directional Breakdown: False Negatives vs False Positives
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    pos_loss = np.mean(pointwise_ll[pos_mask])
    neg_loss = np.mean(pointwise_ll[neg_mask])
    total_pos_contribution = np.sum(pointwise_ll[pos_mask]) / total_loss * 100
    total_neg_contribution = np.sum(pointwise_ll[neg_mask]) / total_loss * 100

    print("\n--- Directional Loss Breakdown ---")
    print(f"  Positive Cases (y=1, {pos_mask.sum()} rows) : Mean Loss = {pos_loss:.4f} ({total_pos_contribution:.1f}% of total loss)")
    print(f"  Negative Cases (y=0, {neg_mask.sum()} rows) : Mean Loss = {neg_loss:.4f} ({total_neg_contribution:.1f}% of total loss)")

    # 4. Cohort-Specific LogLoss Analysis
    train_fe_path = "checkpoints/feature_cache/train_fe.parquet"
    cohort_results = {}
    if os.path.exists(train_fe_path):
        df_meta = pd.read_parquet(train_fe_path, columns=[c for c in ["segment", "region", "gender", "earning_pattern", "financial_earning_indicator", "m1_daily_avg_bal", "smartphone"] if c in pd.read_parquet(train_fe_path).columns])
        df_meta["pointwise_ll"] = pointwise_ll
        df_meta["Target"] = y_true

        print("\n--- Cohort LogLoss Breakdown ---")
        if "segment" in df_meta.columns:
            seg_summary = df_meta.groupby("segment")["pointwise_ll"].agg(["count", "mean", "std"]).rename(columns={"mean": "mean_logloss"})
            seg_summary["pos_rate"] = df_meta.groupby("segment")["Target"].mean()
            print("\nBy Customer Segment:")
            print(seg_summary.sort_values(by="mean_logloss", ascending=False).to_string())
            cohort_results["segment"] = seg_summary.to_dict(orient="index")

        earn_col = "earning_pattern" if "earning_pattern" in df_meta.columns else ("financial_earning_indicator" if "financial_earning_indicator" in df_meta.columns else None)
        if earn_col:
            earn_summary = df_meta.groupby(earn_col)["pointwise_ll"].agg(["count", "mean"]).rename(columns={"mean": "mean_logloss"})
            print(f"\nBy {earn_col}:")
            print(earn_summary.sort_values(by="mean_logloss", ascending=False).to_string())
            cohort_results["earning"] = earn_summary.to_dict(orient="index")

        if "m1_daily_avg_bal" in df_meta.columns:
            df_meta["bal_quartile"] = pd.qcut(df_meta["m1_daily_avg_bal"].rank(method="first"), q=4, labels=["Q1 (Lowest Bal)", "Q2", "Q3", "Q4 (Highest Bal)"])
            bal_summary = df_meta.groupby("bal_quartile")["pointwise_ll"].agg(["count", "mean"]).rename(columns={"mean": "mean_logloss"})
            bal_summary["pos_rate"] = df_meta.groupby("bal_quartile")["Target"].mean()
            print("\nBy Baseline Balance Quartile:")
            print(bal_summary.to_string())
            cohort_results["balance_quartile"] = bal_summary.to_dict(orient="index")

    # 5. Tail Clipping Optimization
    print("\n--- Tail Clipping Threshold Grid Search ---")
    best_clip_comp = base_comp
    best_clip = (0.0020, 0.9995)
    best_clip_ll = base_ll

    floors = [0.0005, 0.0010, 0.0020, 0.0050, 0.0100, 0.0150, 0.0200, 0.0300]
    ceils = [0.9500, 0.9700, 0.9800, 0.9900, 0.9950, 0.9995]

    for fl in floors:
        for ce in ceils:
            clipped = np.clip(oof_preds, fl, ce)
            cl_ll, cl_auc, cl_comp = competition_score(y_true, clipped)
            if cl_comp > best_clip_comp:
                best_clip_comp = cl_comp
                best_clip = (fl, ce)
                best_clip_ll = cl_ll

    print(f"  Current Clipping (floor=0.0020, ceil=0.9995) : Comp={base_comp:.5f} | LL={base_ll:.5f}")
    print(f"  Optimal Clipping (floor={best_clip[0]:.4f}, ceil={best_clip[1]:.4f}): Comp={best_clip_comp:.5f} | LL={best_clip_ll:.5f} (Gain: {best_clip_comp - base_comp:+.5f})")

    # 6. Temperature / Logit Scaling Optimization
    print("\n--- Temperature & Power Probability Scaling ---")
    # Logit transformation
    eps_l = 1e-6
    p_safe = np.clip(oof_preds, eps_l, 1.0 - eps_l)
    logits = np.log(p_safe / (1.0 - p_safe))

    def temp_loss(temp):
        scaled_p = 1.0 / (1.0 + np.exp(-logits / temp))
        ll, _, comp = competition_score(y_true, scaled_p)
        return -comp  # minimize negative comp

    res_t = minimize_scalar(temp_loss, bounds=(0.5, 2.0), method="bounded")
    best_temp = res_t.x
    scaled_p = 1.0 / (1.0 + np.exp(-logits / best_temp))
    t_ll, t_auc, t_comp = competition_score(y_true, scaled_p)
    print(f"  Optimal Temperature (T={best_temp:.3f}): Comp={t_comp:.5f} | LL={t_ll:.5f} | AUC={t_auc:.5f} (Gain: {t_comp - base_comp:+.5f})")

    # Power scaling p^gamma
    def power_loss(gamma):
        p_pow = p_safe ** gamma
        p_pow /= (p_pow + (1.0 - p_safe) ** gamma)
        ll, _, comp = competition_score(y_true, p_pow)
        return -comp

    res_g = minimize_scalar(power_loss, bounds=(0.5, 2.0), method="bounded")
    best_gamma = res_g.x
    pow_p = p_safe ** best_gamma
    pow_p /= (pow_p + (1.0 - p_safe) ** best_gamma)
    g_ll, g_auc, g_comp = competition_score(y_true, pow_p)
    print(f"  Optimal Power (gamma={best_gamma:.3f}) : Comp={g_comp:.5f} | LL={g_ll:.5f} | AUC={g_auc:.5f} (Gain: {g_comp - base_comp:+.5f})")

    # 7. Summary & Actionable Recommendations
    summary = {
        "base_metrics": {"comp": base_comp, "auc": base_auc, "logloss": base_ll},
        "outlier_concentration": {"top_1pct": pct_1, "top_2pct": pct_2, "top_5pct": pct_5},
        "directional_loss": {"pos_loss": pos_loss, "neg_loss": neg_loss, "pos_contribution": total_pos_contribution, "neg_contribution": total_neg_contribution},
        "optimal_clipping": {"floor": best_clip[0], "ceil": best_clip[1], "comp": best_clip_comp, "ll": best_clip_ll},
        "optimal_temperature": {"temp": best_temp, "comp": t_comp, "ll": t_ll},
        "optimal_power": {"gamma": best_gamma, "comp": g_comp, "ll": g_ll},
    }

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/error_analysis_stage1.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("ACTIONABLE LOGLOSS REDUCTION STRATEGIES IDENTIFIED:")
    if best_clip_comp > base_comp + 0.0001:
        print(f"1. Floor/Ceil tuning: Raising floor to {best_clip[0]:.4f} cuts extreme FN penalties, gaining {best_clip_comp - base_comp:+.5f} Comp.")
    if t_comp > base_comp + 0.0001:
        print(f"2. Temperature scaling (T={best_temp:.3f}) cools overconfident boundary predictions, gaining {t_comp - base_comp:+.5f} Comp.")
    print("=" * 75)


if __name__ == "__main__":
    run_error_analysis()
