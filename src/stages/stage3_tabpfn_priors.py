"""
Stage 3: Dual TabPFN Foundation Priors pipeline.
Generates checkpoints/tabpfn.npz and checkpoints/tabpfn_priors.npz.
"""

import os
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CHAMPION_14,
    TARGET,
    ID_COL,
    get_default_dataset_paths,
)
from src.features.pipeline import engineer_features
from src.models.tabpfn_model import fit_tabpfn_multi_view
from src.ensemble.calibration import platt_scaling_calibrate
from src.metrics import competition_score


def run_stage3(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    n_splits: int = 10,
) -> float:
    """Executes Stage 3 TabPFN Foundation Priors."""
    print("Stage 3: Dual TabPFN Foundation Priors")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    if not train_path or not test_path:
        train_path, test_path = get_default_dataset_paths()

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    X_train_feat, X_test_feat, _, _ = engineer_features(train_raw, test_raw)
    target_col = TARGET if TARGET in X_train_feat.columns else "Target"
    y_true = X_train_feat[target_col].to_numpy(int)

    drop_meta = [c for c in [ID_COL, target_col] if c in X_train_feat.columns]
    X_tr = X_train_feat.drop(columns=drop_meta)
    X_te = X_test_feat.drop(columns=[c for c in [ID_COL] if c in X_test_feat.columns])

    pfn_views = {
        "pfn_phys": [c for c in CHAMPION_14 if c in X_tr.columns],
        "pfn_champ": [c for c in CHAMPION_14 if c in X_tr.columns]
        + [c for c in X_tr.columns if any(k in c for k in ["channel", "cash", "bank"])][:6],
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = list(skf.split(X_tr, y_true))

    oof_dict, test_dict = fit_tabpfn_multi_view(
        X_train=X_tr,
        y_true=y_true,
        X_test=X_te,
        folds=folds,
        pfn_views=pfn_views,
    )

    blended_oof = (oof_dict["pfn_phys"] + oof_dict["pfn_champ"]) / 2.0
    blended_test = (test_dict["pfn_phys"] + test_dict["pfn_champ"]) / 2.0
    cal_oof, cal_test, _ = platt_scaling_calibrate(blended_oof, blended_test, y_true)

    ll, auc, final_comp = competition_score(y_true, cal_oof)
    print(f"Stage 3 TabPFN score: {final_comp:.5f} (AUC: {auc:.5f}, LL: {ll:.5f})")

    out_pfn = os.path.join(output_dir, "tabpfn.npz")
    out_priors = os.path.join(output_dir, "tabpfn_priors.npz")
    np.savez_compressed(
        out_pfn,
        oof_tabpfn=cal_oof,
        test_tabpfn=cal_test,
        oof_pfn_phys=oof_dict["pfn_phys"],
        oof_pfn_champ=oof_dict["pfn_champ"],
    )
    np.savez_compressed(out_priors, oof_tabpfn=cal_oof, test_tabpfn=cal_test)

    sub_path = os.path.join(sub_dir, "submission_stage3_tabpfn.csv")
    pd.DataFrame({ID_COL: test_raw[ID_COL], "Target": cal_test}).to_csv(sub_path, index=False)
    print(f"Saved {out_pfn}, {out_priors}, and {sub_path}")
    return final_comp


if __name__ == "__main__":
    run_stage3()
