import inspect
import pytest
import src.stages.stage5_meta_stacker as s5


def test_stage5_does_not_contain_hill_climb_or_prevalence_hack():
    source = inspect.getsource(s5.run_stage5)

    # 1. Must not contain greedy hill_climb_blend call
    assert "hill_climb_blend(" not in source, "Found greedy hill_climb_blend in stage5!"

    # 2. Must not contain hardcoded 0.15340 prevalence hack
    assert "0.15340" not in source, "Found hardcoded 0.15340 target prevalence in stage5!"
    assert "best_test * adj_factor" not in source, "Found multiplicative test scaling in stage5!"

    # 3. Must use cross-fitted calibration at the final step
    assert "platt_scaling_calibrate" in source, "Missing platt_scaling_calibrate in stage5!"
