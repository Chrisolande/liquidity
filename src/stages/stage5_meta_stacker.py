"""
Stage 5: Multi-Stage Meta-Stacker, Hill Climbing, and Final Submission Generator.
Integrates all model streams into the winning composite ensemble.
"""

import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from src.config import (
    ID_COL,
    TARGET,
    FLOOR,
    CEIL,
    get_default_dataset_paths,
)
from src.ensemble.stacking import blend_and_calibrate, nelder_mead_blend, hill_climb_blend
from src.ensemble.calibration import platt_scaling_calibrate
from src.metrics import competition_score, paired_bootstrap, print_correlation_matrix


def run_stage5(
    ckpt_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    test_path: str = None,
) -> float:
    """Executes Stage 5 Meta-Stacker and final verification."""
    print("Stage 5: Meta-Stacker and Hill Climbing Optimizer")

    os.makedirs(sub_dir, exist_ok=True)
    if not test_path:
        _, test_path = get_default_dataset_paths()
    test_df = pd.read_csv(test_path)

    y_path = os.path.join(ckpt_dir, "y_true.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing '{y_path}'. Run earlier stages first to populate checkpoints.")

    y_true = np.load(y_path)
    n_train = len(y_true)
    n_test = len(test_df)

    oof_dict: Dict[str, np.ndarray] = {}
    test_dict: Dict[str, np.ndarray] = {}

    s1_oof_path = os.path.join(ckpt_dir, "oof_champ_train.npy")
    s1_sub_path = os.path.join(sub_dir, "submission_best_0.73731.csv")
    if os.path.exists(s1_oof_path) and os.path.exists(s1_sub_path):
        oof_dict["champ_s1"] = np.load(s1_oof_path)
        test_dict["champ_s1"] = pd.read_csv(s1_sub_path)["Target"].values

    s2_path = os.path.join(ckpt_dir, "gbdt_zoo_4seed.npz")
    if os.path.exists(s2_path):
        data = np.load(s2_path)
        oof_dict["zoo_s2"] = data["oof_s4_tree_zoo"]
        test_dict["zoo_s2"] = data["test_s4_tree_zoo"]

    s3_path = os.path.join(ckpt_dir, "tabpfn.npz")
    if os.path.exists(s3_path):
        data = np.load(s3_path)
        oof_dict["tabpfn_s3"] = data["oof_tabpfn"]
        test_dict["tabpfn_s3"] = data["test_tabpfn"]

    mlp_path = os.path.join(ckpt_dir, "tab_mlp_fast.npz")
    if os.path.exists(mlp_path):
        data = np.load(mlp_path)
        oof_dict["tab_mlp_s4"] = data["oof_mlp"]
        test_dict["tab_mlp_s4"] = data["test_mlp"]

    distill_path = os.path.join(ckpt_dir, "distill_student_10fold.npz")
    if os.path.exists(distill_path):
        data = np.load(distill_path)
        oof_dict["student_lgb_s4"] = data["oof_student_lgb"]
        test_dict["student_lgb_s4"] = data["test_student_lgb"]
        oof_dict["student_xgb_s4"] = data["oof_student_xgb"]
        test_dict["student_xgb_s4"] = data["test_student_xgb"]

    ms_path = os.path.join(ckpt_dir, "multistrata.npz")
    if os.path.exists(ms_path):
        data = np.load(ms_path)
        oof_dict["multistrata_s4"] = data["oof_raw"] if "oof_raw" in data else data["oof_multistrata"]
        test_dict["multistrata_s4"] = data["test_raw"] if "test_raw" in data else data["test_multistrata"]

    if len(oof_dict) < 2:
        print(f"Found {len(oof_dict)} streams in '{ckpt_dir}'. Creating self-contained ensemble.")

    print_correlation_matrix(oof_dict)

    weight_map, oof_blend, test_blend, cal_oof, cal_test = blend_and_calibrate(
        oof_dict=oof_dict,
        test_dict=test_dict,
        y_true=y_true,
        n_splits=5,
    )

    print("Learned Model Weights:")
    for k, w in weight_map.items():
        print(f"  {k:<20}: {w:.4f}")

    oof_df = pd.DataFrame(oof_dict)
    test_pred_df = pd.DataFrame(test_dict)
    hc_res = hill_climb_blend(oof_df, test_pred_df, y_true)

    if hc_res is not None and hc_res[2] > competition_score(y_true, cal_oof)[2]:
        print(f"Hill Climbing score: {hc_res[2]:.5f} > {competition_score(y_true, cal_oof)[2]:.5f}")
        final_oof = hc_res[0]
        final_test = hc_res[1]
    else:
        final_oof = cal_oof
        final_test = cal_test

    final_ll, final_auc, final_comp = competition_score(y_true, final_oof)
    print(f"Stage 5 final meta-stacker score: {final_comp:.5f} (AUC: {final_auc:.5f}, LL: {final_ll:.5f})")

    out_csv = os.path.join(sub_dir, "submission_step13_4seed_0.73737.csv")
    sub_df = pd.DataFrame({ID_COL: test_df[ID_COL], "Target": np.clip(final_test, FLOOR, CEIL)})
    sub_df.to_csv(out_csv, index=False)
    np.save(os.path.join(ckpt_dir, f"oof_step13_{final_comp:.5f}.npy"), final_oof)
    print(f"Saved final submission to {out_csv}")
    return final_comp


if __name__ == "__main__":
    run_stage5()
