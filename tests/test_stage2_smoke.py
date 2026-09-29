import os
import numpy as np
import pandas as pd
import pytest
from src.stages.stage2_gbdt_zoo import run_stage2


def test_stage2_execution_smoke(tmp_path, monkeypatch):
    n_tr, n_te = 40, 20
    np.random.seed(42)

    # 1. Create mock engineered feature matrices
    feature_names = [f"feat_{i}" for i in range(15)]
    cat_cols = ["cat_a"]

    X_train_mock = pd.DataFrame({
        "ID": [f"TR_{i}" for i in range(n_tr)],
        "Target": np.random.binomial(1, 0.2, size=n_tr),
        **{f: np.random.randn(n_tr) for f in feature_names},
        "cat_a": np.random.choice(["A", "B"], size=n_tr),
    })

    X_test_mock = pd.DataFrame({
        "ID": [f"TE_{i}" for i in range(n_te)],
        **{f: np.random.randn(n_te) for f in feature_names},
        "cat_a": np.random.choice(["A", "B"], size=n_te),
    })

    def mock_engineer_features(tr_raw, te_raw, include_solvency=True, use_cache=True):
        return X_train_mock.copy(), X_test_mock.copy(), cat_cols, feature_names

    monkeypatch.setattr("src.stages.stage2_gbdt_zoo.engineer_features", mock_engineer_features)

    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    X_train_mock[["ID", "Target"]].to_csv(tr_path, index=False)
    X_test_mock[["ID"]].to_csv(te_path, index=False)

    ckpt_dir = tmp_path / "checkpoints"
    sub_dir = tmp_path / "submissions"

    comp_score = run_stage2(
        train_path=str(tr_path),
        test_path=str(te_path),
        output_dir=str(ckpt_dir),
        sub_dir=str(sub_dir),
        seeds=(42,),
        n_splits=2,
        k_top_features=8,
    )
    assert comp_score > 0.0
    assert os.path.exists(ckpt_dir / "gbdt_zoo_4seed.npz")
    assert os.path.exists(sub_dir / "submission_s2_gbdt_zoo.csv")

    # Verify raw predictions are saved
    data = np.load(ckpt_dir / "gbdt_zoo_4seed.npz")
    assert "oof_s4_tree_zoo" in data
    assert "test_s4_tree_zoo" in data
    assert "oof_cb_d7_dom26" in data
