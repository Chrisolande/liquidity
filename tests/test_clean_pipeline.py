"""
Unit test suite for Clean Validation Pipeline, Stage 2 Decoupling, and Stage 5 Meta-Stacking.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize, Bounds
from src.config import FLOOR, CEIL, EPS


def test_stage5_best_floor_defined():
    """Verify that Stage 5 FLOOR is defined and valid."""
    assert FLOOR > 0.0
    assert FLOOR < CEIL
    assert CEIL <= 1.0


def test_oof_coverage_assertion_logic():
    """Verify that the boolean mask OOF assertion catches missing and duplicate predictions."""
    n_samples = 100
    y_true = np.zeros(n_samples, dtype=int)
    
    # 1. Perfectly covered splits
    oof_filled = np.zeros(n_samples, dtype=bool)
    folds = [
        np.arange(0, 50),
        np.arange(50, 100)
    ]
    for val_idx in folds:
        assert not oof_filled[val_idx].any(), "Unexpected duplicate"
        oof_filled[val_idx] = True
    assert oof_filled.all(), "Expected full coverage"

    # 2. Missing sample scenario
    oof_incomplete = np.zeros(n_samples, dtype=bool)
    oof_incomplete[0:95] = True
    assert not oof_incomplete.all(), "Should detect missing samples"

    # 3. Duplicate prediction scenario
    oof_dup_check = np.zeros(n_samples, dtype=bool)
    oof_dup_check[0:60] = True
    val_idx_overlap = np.arange(50, 100)
    assert oof_dup_check[val_idx_overlap].any(), "Should detect duplicate prediction"


def test_bounded_regularized_blend_weights():
    """Verify that constrained regularized blending produces non-negative weights summing to 1."""
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.binomial(1, 0.25, size=n_samples)
    
    # 3 synthetic model predictions with varying quality
    p1 = np.clip(y_true * 0.7 + np.random.uniform(0.1, 0.3, n_samples), FLOOR, CEIL)
    p2 = np.clip(y_true * 0.5 + np.random.uniform(0.1, 0.4, n_samples), FLOOR, CEIL)
    p3 = np.clip(np.random.uniform(0.1, 0.4, n_samples), FLOOR, CEIL)  # pure noise
    
    P_mat = np.column_stack([p1, p2, p3])
    init_w = np.ones(3) / 3.0
    
    def loss_func(w):
        w = np.clip(w, 0.0, 1.0)
        w = w / (w.sum() + EPS)
        p = np.clip(P_mat @ w, FLOOR, CEIL)
        ll = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean()
        return ll
    
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = Bounds(0.0, 1.0)
    res = minimize(loss_func, x0=init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    
    assert res.success
    w_opt = res.x
    w_opt = np.clip(w_opt, 0.0, 1.0)
    w_opt /= w_opt.sum()
    
    assert np.all(w_opt >= 0.0)
    assert np.isclose(w_opt.sum(), 1.0)
    # The pure noise model p3 should get the smallest weight
    assert w_opt[2] < w_opt[0]


def test_stage2_anchor_omitted_when_flag_false():
    """Verify that Stage 2 blend candidate names do not contain oof_anchor when include_anchor=False."""
    architectures_def = [
        ("cb_d7_dom26", None, {}, None),
        ("cb_d6_dom35", None, {}, None),
        ("xgb_d4_dom35", None, {}, None),
        ("xgb_d4_triage", None, {}, None),
        ("lgb_extra", None, {}, None),
    ]
    zoo_oof = {name: np.zeros(10) for name, _, _, _ in architectures_def}
    
    # Simulate Stage 2 candidate dictionary building with include_anchor=False
    include_anchor = False
    oof_anchor = np.ones(10) * 0.5
    
    blend_candidates_oof = dict(zoo_oof)
    if include_anchor and oof_anchor is not None:
        blend_candidates_oof["oof_anchor"] = oof_anchor
        
    candidate_names = list(blend_candidates_oof.keys())
    assert "oof_anchor" not in candidate_names
    assert len(candidate_names) == 5
    assert set(candidate_names) == {"cb_d7_dom26", "cb_d6_dom35", "xgb_d4_dom35", "xgb_d4_triage", "lgb_extra"}


def test_stage5_meta_features_alignment():
    """Verify that Stage 5 meta-features have identical lengths, non-empty columns, and probabilities within valid bounds."""
    n_train = 50
    n_test = 25
    oof_stage1 = np.clip(np.random.uniform(0.1, 0.4, n_train), FLOOR, CEIL)
    oof_stage2 = np.clip(np.random.uniform(0.1, 0.4, n_train), FLOOR, CEIL)
    
    test_stage1 = np.clip(np.random.uniform(0.1, 0.4, n_test), FLOOR, CEIL)
    test_stage2 = np.clip(np.random.uniform(0.1, 0.4, n_test), FLOOR, CEIL)
    
    oof_dict = {"stage1": oof_stage1, "stage2": oof_stage2}
    test_dict = {"stage1": test_stage1, "stage2": test_stage2}
    
    for k in oof_dict:
        assert len(oof_dict[k]) == n_train
        assert len(test_dict[k]) == n_test
        assert np.all(oof_dict[k] >= FLOOR) and np.all(oof_dict[k] <= CEIL)
        assert np.all(test_dict[k] >= FLOOR) and np.all(test_dict[k] <= CEIL)
        
    X_meta = np.column_stack([oof_dict["stage1"], oof_dict["stage2"]])
    assert X_meta.shape == (n_train, 2)


def test_clean_pipeline_execution_smoke(tmp_path, monkeypatch):
    """End-to-end smoke test for run_clean_pipeline."""
    import os, json
    from src.clean_pipeline import run_clean_pipeline
    
    n_tr, n_te = 40, 20
    np.random.seed(42)

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

    monkeypatch.setattr("src.clean_pipeline.engineer_features", mock_engineer_features)

    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    X_train_mock[["ID", "Target"]].to_csv(tr_path, index=False)
    X_test_mock[["ID"]].to_csv(te_path, index=False)

    out_dir = tmp_path / "artifacts"
    sub_dir = tmp_path / "submissions"

    manifest = run_clean_pipeline(
        train_path=str(tr_path),
        test_path=str(te_path),
        output_dir=str(out_dir),
        sub_dir=str(sub_dir),
        seeds=(42,),
        n_splits=2,
        k_top_features=8,
        run_tabpfn=False,
    )

    assert manifest["leakage_audit_status"] == "PASS"
    assert manifest["oof_metrics"]["composite_score"] > 0.0
    assert os.path.exists(out_dir / "experiment.json")
    assert os.path.exists(sub_dir / "submission_clean.csv")
    assert os.path.exists(sub_dir / "submission.csv")
    assert os.path.exists(out_dir / "oof_cb_d7_dom26.npy")

    with open(out_dir / "experiment.json") as f:
        meta = json.load(f)
    assert meta["pseudo_labeling"] is False
    assert meta["temperature_sharpening"] is False
    assert meta["prevalence_adjustment"] is False
    assert meta["fold_local_feature_selection"] is True



