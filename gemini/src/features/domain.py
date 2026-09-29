"""
Domain feature engineering: entropy, behavioral shift, longitudinal stress, runway & exhaustion, Deotte playbook.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import EPS
from src.features.monthly import safe_linear_slope


def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes channel transaction entropy & diversity across 6 months."""
    feature_data: Dict[str, np.ndarray] = {}
    channels = [
        "withdraw_total_value",
        "deposit_total_value",
        "received_total_value",
        "mm_send_total_value",
        "paybill_total_value",
        "merchantpay_total_value",
        "transfer_from_bank_total_value",
    ]
    for m in range(1, 7):
        m_cols = [f"m{m}_{ch}" for ch in channels]
        m_matrix = np.maximum(df[m_cols].to_numpy(dtype=float), 0.0)
        row_sums = m_matrix.sum(axis=1, keepdims=True)
        probs = np.divide(
            m_matrix,
            row_sums + EPS,
            out=np.zeros_like(m_matrix),
            where=(row_sums > 0),
        )
        safe_log = np.where(probs > 0, np.log(probs + EPS), 0.0)
        entropy = -np.sum(probs * safe_log, axis=1)
        feature_data[f"m{m}_channel_entropy"] = entropy
        feature_data[f"m{m}_active_channels"] = (m_matrix > 0).sum(axis=1)

    temp = pd.DataFrame(feature_data, index=df.index)
    feature_data["agg_channel_entropy_mean"] = temp[[f"m{m}_channel_entropy" for m in range(1, 7)]].mean(axis=1)
    feature_data["agg_channel_entropy_std"] = temp[[f"m{m}_channel_entropy" for m in range(1, 7)]].std(axis=1)
    feature_data["agg_channel_entropy_trend"] = (
        temp["m1_channel_entropy"] - temp["m6_channel_entropy"]
    )
    feature_data["agg_active_channels_mean"] = temp[[f"m{m}_active_channels" for m in range(1, 7)]].mean(axis=1)
    feature_data["agg_active_channels_min"] = temp[[f"m{m}_active_channels" for m in range(1, 7)]].min(axis=1)
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_behavioral_shift_features(
    df: pd.DataFrame, monthly_groups: Dict[str, List[str]]
) -> pd.DataFrame:
    """Computes behavioral shifts and velocity changes between recent and old periods."""
    feature_data: Dict[str, np.ndarray] = {}
    m1_vol = df[[f"m1_{k}" for k in ["paybill_volume", "merchantpay_volume", "mm_send_volume", "withdraw_volume"]]].sum(axis=1)
    m2_vol = df[[f"m2_{k}" for k in ["paybill_volume", "merchantpay_volume", "mm_send_volume", "withdraw_volume"]]].sum(axis=1)
    m6_vol = df[[f"m6_{k}" for k in ["paybill_volume", "merchantpay_volume", "mm_send_volume", "withdraw_volume"]]].sum(axis=1)

    feature_data["agg_vol_shift_m1_m2"] = (m1_vol - m2_vol) / (m2_vol + EPS)
    feature_data["agg_vol_shift_m1_m6"] = (m1_vol - m6_vol) / (m6_vol + EPS)

    recent_balance = df["m1_daily_avg_bal"].to_numpy(dtype=float)
    recent_inflow = df["m1_inflow_total"].to_numpy(dtype=float)
    recent_outflow = df["m1_outflow_total"].to_numpy(dtype=float)
    paycheck_proxy = df[[f"m{i}_received_total_value" for i in range(1, 7)]].to_numpy(dtype=float).max(axis=1)
    outflow_pressure = df[[f"m{i}_withdraw_total_value" for i in range(1, 7)]].to_numpy(dtype=float).mean(axis=1)

    feature_data["agg_paycheck_to_pressure_ratio"] = paycheck_proxy / (outflow_pressure + EPS)
    feature_data["agg_recent_balance_buffer"] = recent_balance / (recent_outflow + EPS)
    feature_data["agg_recent_net_to_balance"] = df["m1_net_cashflow"].to_numpy(dtype=float) / (recent_balance + EPS)
    feature_data["agg_recent_inflow_minus_outflow"] = recent_inflow - recent_outflow
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_longitudinal_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    """Longitudinal stress, balance drawdown, and cashflow velocity features."""
    feature_data: Dict[str, np.ndarray] = {}
    balance = df[[f"m{i}_daily_avg_bal" for i in range(1, 7)]].to_numpy(dtype=float)
    inflow = df[[f"m{i}_inflow_total" for i in range(1, 7)]].to_numpy(dtype=float)
    outflow = df[[f"m{i}_outflow_total" for i in range(1, 7)]].to_numpy(dtype=float)
    net = df[[f"m{i}_net_cashflow" for i in range(1, 7)]].to_numpy(dtype=float)
    withdraw_ratio = df[[f"m{i}_withdraw_to_balance_ratio" for i in range(1, 7)]].to_numpy(dtype=float)
    activity = df[[f"m{i}_activity_intensity" for i in range(1, 7)]].to_numpy(dtype=float)

    recent_balance = balance[:, :3]
    old_balance = balance[:, 3:]
    recent_inflow = inflow[:, :3]
    old_inflow = inflow[:, 3:]
    recent_outflow = outflow[:, :3]
    old_outflow = outflow[:, 3:]
    recent_net = net[:, :3]
    old_net = net[:, 3:]
    recent_withdraw = withdraw_ratio[:, :3]
    old_withdraw = withdraw_ratio[:, 3:]
    recent_activity = activity[:, :3]
    old_activity = activity[:, 3:]

    latest_balance = balance[:, 0]
    oldest_balance = balance[:, -1]
    peak_balance = balance.max(axis=1)
    floor_balance = balance.min(axis=1)

    feature_data["stress_balance_drawdown_pct"] = (latest_balance - peak_balance) / (np.abs(peak_balance) + EPS)
    feature_data["stress_balance_floor_to_peak_ratio"] = floor_balance / (np.abs(peak_balance) + EPS)
    feature_data["stress_latest_minus_oldest_balance"] = latest_balance - oldest_balance
    feature_data["stress_balance_recent3_to_old3"] = recent_balance.mean(axis=1) / (np.abs(old_balance.mean(axis=1)) + EPS)
    feature_data["stress_recent3_balance_min"] = recent_balance.min(axis=1)
    feature_data["stress_recent3_balance_mean"] = recent_balance.mean(axis=1)
    feature_data["stress_recent3_balance_std"] = recent_balance.std(axis=1)
    feature_data["stress_old3_balance_mean"] = old_balance.mean(axis=1)

    feature_data["stress_recent3_cashflow_sum"] = recent_net.sum(axis=1)
    feature_data["stress_recent3_cashflow_mean"] = recent_net.mean(axis=1)
    feature_data["stress_old3_cashflow_sum"] = old_net.sum(axis=1)
    neg_recent = (recent_net < 0).astype(int)
    feature_data["stress_recent3_negative_cashflow_count"] = neg_recent.sum(axis=1)
    feature_data["stress_recent3_negative_cashflow_streak"] = (
        neg_recent[:, 0] * (1 + neg_recent[:, 1] * (1 + neg_recent[:, 2]))
    )

    feature_data["stress_recent3_inflow_to_outflow"] = recent_inflow.sum(axis=1) / (recent_outflow.sum(axis=1) + EPS)
    feature_data["stress_recent3_balance_to_outflow"] = recent_balance.mean(axis=1) / (recent_outflow.mean(axis=1) + EPS)
    feature_data["stress_recent3_pressure_share"] = (recent_withdraw > 1).mean(axis=1)

    feature_data["stress_pressure_acceleration"] = recent_withdraw.mean(axis=1) - old_withdraw.mean(axis=1)
    feature_data["stress_activity_drop_pct"] = (
        old_activity.mean(axis=1) - recent_activity.mean(axis=1)
    ) / (old_activity.mean(axis=1) + EPS)
    feature_data["stress_inflow_drop_pct"] = (
        old_inflow.mean(axis=1) - recent_inflow.mean(axis=1)
    ) / (old_inflow.mean(axis=1) + EPS)
    feature_data["stress_outflow_rise_pct"] = (
        recent_outflow.mean(axis=1) - old_outflow.mean(axis=1)
    ) / (old_outflow.mean(axis=1) + EPS)

    feature_data["stress_latest_balance_vs_arpu"] = df["m1_daily_avg_bal"].to_numpy(dtype=float) / (
        df["arpu"].to_numpy(dtype=float) + EPS
    )
    feature_data["stress_latest_outflow_vs_arpu"] = df["m1_outflow_total"].to_numpy(dtype=float) / (
        df["arpu"].to_numpy(dtype=float) + EPS
    )
    feature_data["stress_latest_inflow_vs_arpu"] = df["m1_inflow_total"].to_numpy(dtype=float) / (
        df["arpu"].to_numpy(dtype=float) + EPS
    )

    feature_data["stress_recent_bill_stop_flag"] = (
        (df["m1_paybill_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_paybill_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["stress_recent_merchant_stop_flag"] = (
        (df["m1_merchantpay_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_merchantpay_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["stress_recent_deposit_stop_flag"] = (
        (df["m1_deposit_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_deposit_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["stress_recent_received_stop_flag"] = (
        (df["m1_received_volume"].to_numpy(dtype=float) == 0)
        & (df[[f"m{i}_received_volume" for i in [4, 5, 6]]].sum(axis=1).to_numpy(dtype=float) > 0)
    ).astype(int)
    feature_data["stress_paycheck_pressure_combo"] = df["agg_paycheck_to_pressure_ratio"].to_numpy(dtype=float) * (
        1.0 / (df["agg_recent_balance_buffer"].to_numpy(dtype=float) + EPS)
    )
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


# Backward compatibility alias
add_v3_features = add_longitudinal_stress_features


def add_liquidity_runway_and_exhaustion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes burn rate, runway, exhaustion, and shrinkage features."""
    feature_data: Dict[str, np.ndarray] = {}
    month_idx = range(1, 7)
    
    for i in month_idx:
        bal = df[f"m{i}_daily_avg_bal"].to_numpy(dtype=float)
        inflow = df[f"m{i}_inflow_total"].to_numpy(dtype=float)
        outflow = df[f"m{i}_outflow_total"].to_numpy(dtype=float)
        withdraw = df[f"m{i}_withdraw_total_value"].to_numpy(dtype=float)
        received = df[f"m{i}_received_total_value"].to_numpy(dtype=float)
        bank = df[f"m{i}_transfer_from_bank_total_value"].to_numpy(dtype=float)
        
        feature_data[f"m{i}_inflow_exhaustion_rate"] = outflow / (inflow + bal + EPS)
        feature_data[f"m{i}_liquidity_runway"] = np.clip(bal / (np.maximum(0.0, outflow - inflow) + EPS), 0.0, 12.0)
        feature_data[f"m{i}_withdraw_drain_rate"] = withdraw / (inflow + bal + EPS)
        feature_data[f"m{i}_p2p_dependency"] = received / (inflow + EPS)
        feature_data[f"m{i}_emergency_bank_rate"] = bank / (inflow + EPS)

    temp = pd.DataFrame(feature_data, index=df.index)
    exhaust_cols = [f"m{i}_inflow_exhaustion_rate" for i in month_idx]
    runway_cols = [f"m{i}_liquidity_runway" for i in month_idx]
    
    feature_data["agg_runway_min"] = temp[runway_cols].min(axis=1)
    feature_data["agg_runway_mean"] = temp[runway_cols].mean(axis=1)
    feature_data["agg_runway_slope"] = safe_linear_slope(temp[runway_cols].to_numpy(dtype=float))
    
    feature_data["agg_exhaustion_max"] = temp[exhaust_cols].max(axis=1)
    feature_data["agg_exhaustion_mean"] = temp[exhaust_cols].mean(axis=1)
    feature_data["agg_exhaustion_slope"] = safe_linear_slope(temp[exhaust_cols].to_numpy(dtype=float))
    feature_data["agg_exhaustion_acceleration"] = (
        (temp["m1_inflow_exhaustion_rate"] - temp["m2_inflow_exhaustion_rate"])
        - (temp["m2_inflow_exhaustion_rate"] - temp["m3_inflow_exhaustion_rate"])
    )
    
    feature_data["agg_bal_collapse_1_max"] = df["m1_daily_avg_bal"].to_numpy(dtype=float) / (
        df[[f"m{i}_daily_avg_bal" for i in range(2, 7)]].max(axis=1).to_numpy(dtype=float) + EPS
    )
    feature_data["agg_bal_collapse_1_2"] = df["m1_daily_avg_bal"].to_numpy(dtype=float) / (
        df["m2_daily_avg_bal"].to_numpy(dtype=float) + EPS
    )
    feature_data["agg_burn_rate_1_2"] = (
        df["m2_daily_avg_bal"].to_numpy(dtype=float) - df["m1_daily_avg_bal"].to_numpy(dtype=float)
    ) / (df["m2_daily_avg_bal"].to_numpy(dtype=float) + EPS)
    feature_data["agg_burn_rate_1_3"] = (
        df["m3_daily_avg_bal"].to_numpy(dtype=float) - df["m1_daily_avg_bal"].to_numpy(dtype=float)
    ) / (df["m3_daily_avg_bal"].to_numpy(dtype=float) + EPS)
    feature_data["agg_burn_acceleration"] = (
        (df["m2_daily_avg_bal"].to_numpy(dtype=float) - df["m1_daily_avg_bal"].to_numpy(dtype=float))
        - (df["m3_daily_avg_bal"].to_numpy(dtype=float) - df["m2_daily_avg_bal"].to_numpy(dtype=float))
    )
    
    feature_data["agg_agent_shrinkage"] = (
        df["m1_deposit_agents"].to_numpy(dtype=float) + df["m1_withdraw_agents"].to_numpy(dtype=float)
    ) / (df["m6_deposit_agents"].to_numpy(dtype=float) + df["m6_withdraw_agents"].to_numpy(dtype=float) + EPS)
    feature_data["agg_sender_shrinkage"] = df["m1_received_senders"].to_numpy(dtype=float) / (
        df["m6_received_senders"].to_numpy(dtype=float) + EPS
    )
    feature_data["agg_recip_shrinkage"] = df["m1_mm_send_recipients"].to_numpy(dtype=float) / (
        df["m6_mm_send_recipients"].to_numpy(dtype=float) + EPS
    )
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_solvency_and_burn_collapse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates multi-month cashflow solvency and burn collapse metrics."""
    feature_data: Dict[str, np.ndarray] = {}

    for m in range(1, 7):
        in_m = (
            df[f"m{m}_received_total_value"].to_numpy(dtype=float)
            + df[f"m{m}_deposit_total_value"].to_numpy(dtype=float)
            + df[f"m{m}_transfer_from_bank_total_value"].to_numpy(dtype=float)
        )
        out_m = (
            df[f"m{m}_withdraw_total_value"].to_numpy(dtype=float)
            + df[f"m{m}_mm_send_total_value"].to_numpy(dtype=float)
            + df[f"m{m}_paybill_total_value"].to_numpy(dtype=float)
            + df[f"m{m}_merchantpay_total_value"].to_numpy(dtype=float)
        )
        bal_m = df[f"m{m}_daily_avg_bal"].to_numpy(dtype=float)

        feature_data[f"solv_inflow_m{m}"] = in_m
        feature_data[f"solv_outflow_m{m}"] = out_m
        feature_data[f"solv_net_m{m}"] = in_m - out_m
        feature_data[f"solv_cushion_m{m}"] = bal_m + in_m - out_m
        feature_data[f"solv_obligation_m{m}"] = (
            df[f"m{m}_paybill_total_value"].to_numpy(dtype=float)
            + df[f"m{m}_withdraw_total_value"].to_numpy(dtype=float)
        )

    # Inflow and Deposit Collapse
    in_rec = feature_data["solv_inflow_m1"] + feature_data["solv_inflow_m2"]
    in_old = feature_data["solv_inflow_m5"] + feature_data["solv_inflow_m6"]
    dep_rec = df["m1_deposit_total_value"].to_numpy(dtype=float) + df["m2_deposit_total_value"].to_numpy(dtype=float)
    dep_old = df["m5_deposit_total_value"].to_numpy(dtype=float) + df["m6_deposit_total_value"].to_numpy(dtype=float)

    feature_data["solv_in_collapse_ratio"] = in_rec / (in_old + EPS)
    feature_data["solv_in_drop_diff"] = in_old - in_rec
    feature_data["solv_in_drop_pct"] = (in_old - in_rec) / (np.abs(in_old) + EPS)
    feature_data["solv_dep_collapse_ratio"] = dep_rec / (dep_old + EPS)
    feature_data["solv_dep_drop_diff"] = dep_old - dep_rec

    # Balance and Cushion Drawdown
    bal_rec = (
        df["m1_daily_avg_bal"].to_numpy(dtype=float)
        + df["m2_daily_avg_bal"].to_numpy(dtype=float)
        + df["m3_daily_avg_bal"].to_numpy(dtype=float)
    ) / 3.0
    bal_old = (
        df["m4_daily_avg_bal"].to_numpy(dtype=float)
        + df["m5_daily_avg_bal"].to_numpy(dtype=float)
        + df["m6_daily_avg_bal"].to_numpy(dtype=float)
    ) / 3.0
    cush_rec = (
        feature_data["solv_cushion_m1"]
        + feature_data["solv_cushion_m2"]
        + feature_data["solv_cushion_m3"]
    ) / 3.0
    cush_old = (
        feature_data["solv_cushion_m4"]
        + feature_data["solv_cushion_m5"]
        + feature_data["solv_cushion_m6"]
    ) / 3.0

    feature_data["solv_bal_collapse_ratio"] = bal_rec / (bal_old + EPS)
    feature_data["solv_bal_collapse_diff"] = bal_old - bal_rec
    feature_data["solv_cushion_collapse_ratio"] = cush_rec / (np.abs(cush_old) + EPS)
    feature_data["solv_cushion_drop_pct"] = (cush_old - cush_rec) / (np.abs(cush_old) + EPS)
    feature_data["solv_cushion_to_arpu"] = feature_data["solv_cushion_m1"] / (df["arpu"].to_numpy(dtype=float) + EPS)

    # Cumulative Burn & Solvency Runway
    cum_burn_3m = (
        feature_data["solv_net_m1"]
        + feature_data["solv_net_m2"]
        + feature_data["solv_net_m3"]
    )
    feature_data["solv_cum_burn_3m"] = cum_burn_3m
    feature_data["solv_cum_burn_to_bal"] = cum_burn_3m / (df["m1_daily_avg_bal"].to_numpy(dtype=float) + EPS)
    feature_data["solv_runway_from_burn"] = df["m1_daily_avg_bal"].to_numpy(dtype=float) / (
        np.maximum(0.0, -cum_burn_3m / 3.0) + EPS
    )

    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_chris_deotte_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Combinatorial interaction categoricals, frequency encoding, and transductive rankings."""
    feature_data: Dict[str, np.ndarray] = {}

    earn_col = "earning_pattern" if "earning_pattern" in df.columns else "financial_earning_indicator"
    phone_col = "smartphone" if "smartphone" in df.columns else None

    df["seg_earn"] = df["segment"].astype(str) + "_" + df[earn_col].astype(str)
    df["reg_seg"] = df["region"].astype(str) + "_" + df["segment"].astype(str)
    df["reg_earn"] = df["region"].astype(str) + "_" + df[earn_col].astype(str)
    df["gen_earn"] = df["gender"].astype(str) + "_" + df[earn_col].astype(str)
    df["gen_seg"] = df["gender"].astype(str) + "_" + df["segment"].astype(str)

    combo_cats = ["seg_earn", "reg_seg", "reg_earn", "gen_earn", "gen_seg"]
    base_cats = ["segment", earn_col, "region", "gender"]

    if phone_col and phone_col in df.columns:
        df["phone_seg"] = df[phone_col].astype(str) + "_" + df["segment"].astype(str)
        df["phone_earn"] = df[phone_col].astype(str) + "_" + df[earn_col].astype(str)
        combo_cats.extend(["phone_seg", "phone_earn"])
        base_cats.append(phone_col)

    if "age" in df.columns:
        age_binned = pd.qcut(df["age"].fillna(df["age"].median()), q=5, labels=False, duplicates="drop").astype(str)
        df["age_seg"] = age_binned + "_" + df["segment"].astype(str)
        combo_cats.append("age_seg")

    all_cats = base_cats + combo_cats

    for col in all_cats:
        counts = df[col].value_counts(normalize=True)
        feature_data[f"{col}_freq"] = df[col].map(counts).to_numpy(dtype=float)

    seg_bal = df.groupby("segment")["m1_daily_avg_bal"].transform("median")
    feature_data["bal_to_seg_median"] = df["m1_daily_avg_bal"] / (seg_bal + EPS)

    earn_inflow = df.groupby(earn_col)["m1_inflow_total"].transform("median")
    feature_data["inflow_to_earn_median"] = df["m1_inflow_total"] / (earn_inflow + EPS)

    seg_arpu = df.groupby("segment")["arpu"].transform("mean")
    feature_data["arpu_to_seg_mean"] = df["arpu"] / (seg_arpu + EPS)

    for rcol in ["m1_daily_avg_bal", "m1_inflow_total", "agg_runway_min", "agg_exhaustion_max"]:
        if rcol in df.columns:
            feature_data[f"{rcol}_rank_pct"] = df[rcol].rank(pct=True).to_numpy(dtype=float)
        elif rcol in feature_data:
            feature_data[f"{rcol}_rank_pct"] = pd.Series(feature_data[rcol]).rank(pct=True).to_numpy(dtype=float)

    df_out = pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)
    return df_out, all_cats
