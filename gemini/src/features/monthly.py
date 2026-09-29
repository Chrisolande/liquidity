"""
Monthly temporal aggregation, summary metrics, and cross-channel ratio features.
"""

import re
from typing import Dict, List
import numpy as np
import pandas as pd

from src.config import EPS


def month_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Groups monthly columns m1..m6 by channel/signal suffix."""
    pattern = re.compile(r"^m([1-6])_(.+)$")
    groups: Dict[str, List[str]] = {}
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        suffix = match.group(2)
        groups.setdefault(suffix, []).append(col)
    for suffix, cols in groups.items():
        groups[suffix] = sorted(cols, key=lambda c: int(pattern.match(c).group(1)), reverse=True)
    return groups


def safe_linear_slope(values: np.ndarray) -> np.ndarray:
    """Computes linear trend slope across time points safely without crashing on NaNs."""
    n_points = values.shape[1]
    x = np.arange(n_points, dtype=float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    y_mean = np.nanmean(values, axis=1, keepdims=True)
    y_mean = np.nan_to_num(y_mean, nan=0.0)
    y_filled = np.where(np.isnan(values), y_mean, values)
    y_centered = y_filled - y_mean
    return np.dot(y_centered, x_centered) / (denom + EPS)


def add_monthly_summary_features(
    df: pd.DataFrame, monthly_groups: Dict[str, List[str]]
) -> pd.DataFrame:
    """Computes statistical summaries (mean, std, CV, slope, recent vs old) across months."""
    feature_data: Dict[str, np.ndarray] = {}
    for suffix, cols in monthly_groups.items():
        values = df[cols].to_numpy(dtype=float)
        oldest = values[:, 0]
        newest = values[:, -1]
        mean = np.nanmean(values, axis=1)
        std = np.nanstd(values, axis=1)
        min_ = np.nanmin(values, axis=1)
        max_ = np.nanmax(values, axis=1)
        median = np.nanmedian(values, axis=1)
        slope = safe_linear_slope(values)
        recent_3 = np.nanmean(values[:, -3:], axis=1)
        old_3 = np.nanmean(values[:, :3], axis=1)
        non_zero = (values > 0).sum(axis=1)
        last_positive_month = np.where(
            (values > 0).any(axis=1),
            values.shape[1] - np.argmax(values[:, ::-1] > 0, axis=1),
            0,
        )
        zero_share = (values == 0).mean(axis=1)

        prefix = f"agg_{suffix}"
        feature_data[f"{prefix}_mean"] = mean
        feature_data[f"{prefix}_std"] = std
        feature_data[f"{prefix}_cv"] = std / (np.abs(mean) + EPS)
        feature_data[f"{prefix}_min"] = min_
        feature_data[f"{prefix}_max"] = max_
        feature_data[f"{prefix}_median"] = median
        feature_data[f"{prefix}_range"] = max_ - min_
        feature_data[f"{prefix}_newest"] = newest
        feature_data[f"{prefix}_oldest"] = oldest
        feature_data[f"{prefix}_newest_to_oldest_ratio"] = newest / (oldest + EPS)
        feature_data[f"{prefix}_recent3_to_old3_ratio"] = recent_3 / (old_3 + EPS)
        feature_data[f"{prefix}_recent3_minus_old3"] = recent_3 - old_3
        feature_data[f"{prefix}_slope"] = slope
        feature_data[f"{prefix}_non_zero_months"] = non_zero
        feature_data[f"{prefix}_zero_share"] = zero_share
        feature_data[f"{prefix}_last_positive_month_index"] = last_positive_month
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)


def add_cross_feature_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Computes inflow/outflow, activity intensity, and liquidity dynamics."""
    feature_data: Dict[str, np.ndarray] = {}
    month_idx = range(1, 7)
    for i in month_idx:
        inflow = (
            df[f"m{i}_received_total_value"]
            + df[f"m{i}_deposit_total_value"]
            + df[f"m{i}_transfer_from_bank_total_value"]
        )
        outflow = (
            df[f"m{i}_withdraw_total_value"]
            + df[f"m{i}_mm_send_total_value"]
            + df[f"m{i}_paybill_total_value"]
            + df[f"m{i}_merchantpay_total_value"]
        )
        feature_data[f"m{i}_inflow_total"] = inflow
        feature_data[f"m{i}_outflow_total"] = outflow
        feature_data[f"m{i}_net_cashflow"] = inflow - outflow
        feature_data[f"m{i}_inflow_to_outflow_ratio"] = inflow / (outflow + EPS)
        feature_data[f"m{i}_withdraw_to_balance_ratio"] = df[f"m{i}_withdraw_total_value"] / (
            df[f"m{i}_daily_avg_bal"] + EPS
        )
        feature_data[f"m{i}_deposit_to_withdraw_ratio"] = df[f"m{i}_deposit_total_value"] / (
            df[f"m{i}_withdraw_total_value"] + EPS
        )
        feature_data[f"m{i}_received_to_send_ratio"] = df[f"m{i}_received_total_value"] / (
            df[f"m{i}_mm_send_total_value"] + EPS
        )
        feature_data[f"m{i}_merchantpay_to_paybill_ratio"] = df[f"m{i}_merchantpay_total_value"] / (
            df[f"m{i}_paybill_total_value"] + EPS
        )
        feature_data[f"m{i}_avg_withdraw_amt"] = df[f"m{i}_withdraw_total_value"] / (df[f"m{i}_withdraw_volume"] + EPS)
        feature_data[f"m{i}_avg_deposit_amt"] = df[f"m{i}_deposit_total_value"] / (df[f"m{i}_deposit_volume"] + EPS)
        feature_data[f"m{i}_avg_received_amt"] = df[f"m{i}_received_total_value"] / (df[f"m{i}_received_volume"] + EPS)
        feature_data[f"m{i}_avg_send_amt"] = df[f"m{i}_mm_send_total_value"] / (df[f"m{i}_mm_send_volume"] + EPS)
        feature_data[f"m{i}_activity_intensity"] = (
            df[f"m{i}_paybill_volume"]
            + df[f"m{i}_merchantpay_volume"]
            + df[f"m{i}_transfer_from_bank_volume"]
            + df[f"m{i}_mm_send_volume"]
            + df[f"m{i}_received_volume"]
            + df[f"m{i}_deposit_volume"]
            + df[f"m{i}_withdraw_volume"]
        )

    temp = pd.DataFrame(feature_data, index=df.index)
    feature_data["agg_total_inflow_6m"] = temp[[f"m{i}_inflow_total" for i in month_idx]].sum(axis=1)
    feature_data["agg_total_outflow_6m"] = temp[[f"m{i}_outflow_total" for i in month_idx]].sum(axis=1)
    feature_data["agg_total_net_cashflow_6m"] = temp[[f"m{i}_net_cashflow" for i in month_idx]].sum(axis=1)
    feature_data["agg_inflow_to_outflow_6m"] = feature_data["agg_total_inflow_6m"] / (
        feature_data["agg_total_outflow_6m"] + EPS
    )
    feature_data["agg_recent3_inflow"] = temp[[f"m{i}_inflow_total" for i in [1, 2, 3]]].sum(axis=1)
    feature_data["agg_recent3_outflow"] = temp[[f"m{i}_outflow_total" for i in [1, 2, 3]]].sum(axis=1)
    feature_data["agg_recent3_net_cashflow"] = temp[[f"m{i}_net_cashflow" for i in [1, 2, 3]]].sum(axis=1)
    feature_data["agg_old3_inflow"] = temp[[f"m{i}_inflow_total" for i in [4, 5, 6]]].sum(axis=1)
    feature_data["agg_old3_outflow"] = temp[[f"m{i}_outflow_total" for i in [4, 5, 6]]].sum(axis=1)
    feature_data["agg_old3_net_cashflow"] = temp[[f"m{i}_net_cashflow" for i in [4, 5, 6]]].sum(axis=1)

    feature_data["agg_inflow_trend_ratio"] = feature_data["agg_recent3_inflow"] / (
        feature_data["agg_old3_inflow"] + EPS
    )
    feature_data["agg_outflow_trend_ratio"] = feature_data["agg_recent3_outflow"] / (
        feature_data["agg_old3_outflow"] + EPS
    )
    feature_data["agg_net_cashflow_delta"] = (
        feature_data["agg_recent3_net_cashflow"] - feature_data["agg_old3_net_cashflow"]
    )
    return pd.concat([df, pd.DataFrame(feature_data, index=df.index)], axis=1)
