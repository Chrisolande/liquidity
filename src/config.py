"""
Global configuration, constants, and environment utilities for AI4EAC Liquidity Stress.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

# Competition Invariants
SEED: int = 42
N_SPLITS: int = 10
EPS: float = 1e-6
TARGET: str = "liquidity_stress_next_30d"
ID_COL: str = "ID"
FLOOR: float = 0.0020
CEIL: float = 0.9995

# Budget Presets
BUDGET: str = os.environ.get("LSEW_BUDGET", "fast").lower()

_PRESETS: Dict[str, dict] = {
    "smoke": dict(
        n_splits=3,
        gbdt_seeds=(42,),
        cb_iters=60,
        xgb_iters=60,
        lgb_iters=60,
        screen_features=30,
        use_mlp=False,
        use_tabpfn=False,
    ),
    "fast": dict(
        n_splits=5,
        gbdt_seeds=(42, 100),
        cb_iters=350,
        xgb_iters=350,
        lgb_iters=400,
        screen_features=80,
        use_mlp=True,
        mlp_epochs=12,
        use_tabpfn=False,
    ),
    "balanced": dict(
        n_splits=10,
        gbdt_seeds=(42, 100, 2024),
        cb_iters=700,
        xgb_iters=700,
        lgb_iters=800,
        screen_features=120,
        use_mlp=True,
        mlp_epochs=25,
        use_tabpfn=True,
        tabpfn_views=("physics", "champion"),
    ),
    "max": dict(
        n_splits=10,
        gbdt_seeds=(42, 100, 2024, 777),
        cb_iters=1200,
        xgb_iters=1200,
        lgb_iters=1400,
        screen_features=0,
        use_mlp=True,
        mlp_epochs=40,
        use_tabpfn=True,
        tabpfn_views=("physics", "champion", "gain30"),
    ),
}

# Domain Feature Subsets (Exact column names produced by features/pipeline.py)
CHAMPION_14: List[str] = [
    "agg_inflow_trend_ratio",
    "agg_daily_avg_bal_recent3_to_old3_ratio",
    "agg_withdraw_total_value_recent3_to_old3_ratio",
    "agg_inflow_to_outflow_6m",
    "agg_paybill_total_value_recent3_to_old3_ratio",
    "agg_recent3_net_cashflow",
    "agg_recent3_inflow",
    "agg_withdraw_highest_amount_recent3_to_old3_ratio",
    "x_90_d_activity_rate",
    "agg_withdraw_volume_recent3_to_old3_ratio",
    "agg_merchantpay_total_value_recent3_to_old3_ratio",
    "agg_daily_avg_bal_cv",
    "agg_daily_avg_bal_recent3_minus_old3",
    "agg_net_cashflow_delta",
]

MACRO_SOLVENCY_10: List[str] = [
    "agg_recent3_net_cashflow",
    "agg_recent3_inflow",
    "agg_inflow_trend_ratio",
    "agg_inflow_to_outflow_6m",
    "agg_daily_avg_bal_recent3_to_old3_ratio",
    "agg_daily_avg_bal_recent3_minus_old3",
    "agg_withdraw_total_value_recent3_to_old3_ratio",
    "agg_paybill_total_value_recent3_to_old3_ratio",
    "x_90_d_activity_rate",
    "agg_daily_avg_bal_cv",
]

MACRO_TRIAGE_10: List[str] = [
    "agg_inflow_trend_ratio",
    "agg_daily_avg_bal_recent3_to_old3_ratio",
    "agg_withdraw_total_value_recent3_to_old3_ratio",
    "agg_inflow_to_outflow_6m",
    "agg_withdraw_highest_amount_recent3_to_old3_ratio",
    "agg_withdraw_volume_recent3_to_old3_ratio",
    "agg_merchantpay_total_value_recent3_to_old3_ratio",
    "agg_daily_avg_bal_cv",
    "agg_recent3_net_cashflow",
    "x_90_d_activity_rate",
]



def get_hardware_info() -> Dict[str, object]:
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"
    num_gpus = torch.cuda.device_count() if has_gpu else 0
    return {
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "num_gpus": num_gpus,
    }


_HW = get_hardware_info()
HAS_GPU: bool = _HW["has_gpu"]
GPU_NAME: str = _HW["gpu_name"]
NUM_GPUS: int = _HW["num_gpus"]


def get_budget_preset(budget: str = None) -> dict:
    """Returns runtime parameters for a given execution budget preset."""
    b = (budget or BUDGET).lower()
    return _PRESETS.get(b, _PRESETS["fast"])


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# Initialize global seed on import
seed_everything(SEED)


def get_default_dataset_paths() -> Tuple[str, str]:
    base_dirs = [
        Path("/kaggle/input/datasets/chrisolande2/zindi-competition"),
        Path("/kaggle/input/datasets/chrisolande/zindi-competition"),
        Path("/kaggle/input/chrisolande/zindi-competition"),
        Path("/kaggle/input/ai4eac-liquidity-stress"),
        Path("chrisolande/zindi-competition"),
        Path("."),
        Path("data"),
    ]
    for b in base_dirs:
        tr = b / "Train.csv"
        te = b / "Test.csv"
        if tr.exists() and te.exists():
            return str(tr), str(te)

    # Dynamic fallback scan under /kaggle/input
    if Path("/kaggle/input").exists():
        tr_candidates = [p for p in Path("/kaggle/input").rglob("Train.csv") if p.is_file() and p.stat().st_size > 1000]
        te_candidates = [p for p in Path("/kaggle/input").rglob("Test.csv") if p.is_file() and p.stat().st_size > 1000]
        if tr_candidates and te_candidates:
            return str(tr_candidates[0]), str(te_candidates[0])

    default_dir = Path("/kaggle/input/datasets/chrisolande2/zindi-competition")
    return str(default_dir / "Train.csv"), str(default_dir / "Test.csv")

