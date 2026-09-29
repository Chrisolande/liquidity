"""
Pseudo-label generation module for iterative student self-training distillation.
Blends Stage 1 baseline, Stage 2 GBDT zoo, and Stage 3 TabPFN / surrogate priors
to produce soft calibrated probability targets for the unlabelled test set.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.config import ID_COL, FLOOR, CEIL, TARGET, get_default_dataset_paths
from src.ensemble.stacking import blend_and_calibrate
from src.metrics import competition_score


def build_pseudo_labels(
    test_path: Optional[str] = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    out_path: str = "pseudo_labels_iter3.csv",
) -> pd.DataFrame:
    """
    Builds soft pseudo-labels for the test set by blending Stage 1, Stage 2, and Stage 3 models.
    Leaves MultiStrata out on purpose to prevent recursive target leakage.
    """
    if not test_path:
        _, test_path = get_default_dataset_paths()

    test_df = pd.read_csv(test_path)
    y_path = os.path.join(output_dir, "y_true.npy")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing ground truth targets '{y_path}'. Run Stage 1 or Stage 2 first.")

    y_true = np.load(y_path)
    oof_dict: Dict[str, np.ndarray] = {}
    test_dict: Dict[str, np.ndarray] = {}

    # Stream 1: Stage 1 Champion Baseline
    s1_oof = os.path.join(output_dir, "oof_champ_train.npy")
    s1_sub = os.path.join(sub_dir, "submission_best_0.73731.csv")
    if os.path.exists(s1_oof) and os.path.exists(s1_sub):
        oof_dict["champ_s1"] = np.load(s1_oof)
        test_dict["champ_s1"] = pd.read_csv(s1_sub)["Target"].to_numpy(dtype=float)

    # Stream 2: Stage 2 GBDT Zoo
    s2_npz = os.path.join(output_dir, "gbdt_zoo_4seed.npz")
    if os.path.exists(s2_npz):
        data = np.load(s2_npz)
        oof_dict["zoo_s2"] = data["oof_s4_tree_zoo"]
        test_dict["zoo_s2"] = data["test_s4_tree_zoo"]

    # Stream 3: Stage 3 TabPFN / Surrogate Priors
    s3_npz = os.path.join(output_dir, "tabpfn.npz")
    if os.path.exists(s3_npz):
        data = np.load(s3_npz)
        oof_dict["tabpfn_s3"] = data["oof_tabpfn"]
        test_dict["tabpfn_s3"] = data["test_tabpfn"]

    if len(oof_dict) == 0:
        raise ValueError(f"No checkpoint streams found in '{output_dir}'. Run stages 1-3 first.")

    print(f"Building pseudo-labels from {len(oof_dict)} streams: {list(oof_dict.keys())}", flush=True)

    weights, oof_blend, test_blend, cal_oof, cal_test = blend_and_calibrate(
        oof_dict=oof_dict,
        test_dict=test_dict,
        y_true=y_true,
        n_splits=5,
    )

    ll, auc, comp = competition_score(y_true, cal_oof)
    print(f"Pseudo-label teacher ensemble: Comp={comp:.5f} | AUC={auc:.5f} | LL={ll:.5f}", flush=True)
    print("Learned Teacher Weights:")
    for k, w in weights.items():
        print(f"  {k:<20}: {w:.4f}")

    soft_targets = np.clip(cal_test, FLOOR, CEIL)
    df_pseudo = pd.DataFrame({
        ID_COL: test_df[ID_COL],
        "Target": soft_targets,
    })

    df_pseudo.to_csv(out_path, index=False)
    print(f"Saved {len(df_pseudo)} calibrated pseudo-labels to {out_path} (bounds: [{soft_targets.min():.5f}, {soft_targets.max():.5f}])", flush=True)
    return df_pseudo
