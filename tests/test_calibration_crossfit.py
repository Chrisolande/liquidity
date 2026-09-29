import numpy as np
import pytest
from src.ensemble.stacking import blend_and_calibrate
from src.metrics import competition_score


def test_blend_and_calibrate_is_cross_fitted():
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.binomial(1, 0.15, size=n_samples)

    # Synthetic predictions from two models with slight noise
    m1 = np.clip(y_true * 0.7 + np.random.normal(0, 0.1, size=n_samples), 0.01, 0.99)
    m2 = np.clip(y_true * 0.6 + np.random.normal(0, 0.15, size=n_samples), 0.01, 0.99)

    oof_dict = {"m1": m1, "m2": m2}
    test_dict = {"m1": m1[:50], "m2": m2[:50]}

    weights, oof_b, test_b, oof_cal, test_cal = blend_and_calibrate(
        oof_dict, test_dict, y_true, n_splits=5, seed=42
    )

    # 1. Output shapes must match inputs
    assert oof_cal.shape == (n_samples,)
    assert test_cal.shape == (50,)

    from src.config import FLOOR, CEIL
    assert np.all(oof_cal >= FLOOR) and np.all(oof_cal <= CEIL)
    assert np.all(test_cal >= FLOOR) and np.all(test_cal <= CEIL)

    # 3. Verify it is not an in-sample fit:
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    z_oof = logit(np.clip(oof_b, 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
    in_sample_cal = LogisticRegression(C=1.0, solver="lbfgs").fit(z_oof, y_true)
    in_sample_preds = in_sample_cal.predict_proba(z_oof)[:, 1]

    # Out-of-fold cross-fitted predictions should not exactly match global in-sample fit
    assert not np.allclose(oof_cal, in_sample_preds, atol=1e-5)
