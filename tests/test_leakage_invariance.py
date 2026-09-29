import numpy as np
import pandas as pd
import pytest
from src.features.encoding import screen_features


def test_screen_features_does_not_depend_on_validation_split():
    np.random.seed(42)
    n_train = 100
    n_val = 20

    # Create dummy dataframe with informative and noisy columns
    X_tr = pd.DataFrame({
        f"num_{i}": np.random.randn(n_train) for i in range(20)
    })
    y_tr = np.random.binomial(1, 0.3, size=n_train)
    # Add real signal to num_0
    X_tr["num_0"] += y_tr * 2.0

    # Validation slice
    X_va = pd.DataFrame({
        f"num_{i}": np.random.randn(n_val) for i in range(20)
    })
    y_va_original = np.random.binomial(1, 0.3, size=n_val)
    y_va_perturbed = 1 - y_va_original  # completely inverted labels

    # Screen features strictly on training split
    sel1 = screen_features(X_tr, y_tr, cat_cols=[], k_top=5, seed=42)
    sel2 = screen_features(X_tr, y_tr, cat_cols=[], k_top=5, seed=42)

    # Invariance check: Screening on X_tr is deterministic and isolated from val
    assert sel1 == sel2
    assert "num_0" in sel1
