import inspect
import pytest
import src.stages.stage4_diversity as s4


def test_stage4_does_not_train_on_sharpened_test_data():
    source = inspect.getsource(s4.run_stage4)

    # 1. Must not sharpen test probabilities with T=0.85
    assert "temperature=0.85" not in source, "Found temperature=0.85 sharpening in stage4!"

    # 2. Must not inject unlabelled test set into training folds
    assert "np.vstack([X_tr_arr[trn_idx], X_te_arr])" not in source, "Found test pseudo-label stack in stage4!"
    assert "np.r_[y_true[trn_idx], p_sharp]" not in source, "Found sharpened pseudo-target stack in stage4!"
