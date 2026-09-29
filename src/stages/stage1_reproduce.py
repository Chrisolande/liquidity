"""
Stage 1: Baseline Reproduction Pipeline.
Executes baseline cross-validation (CatBoost + HistGradientBoosting + Isotonic Calibration),
producing checkpoints/oof_champ_train.npy and submissions/submission_best_0.73731.csv.
"""

from src.baseline import run_baseline


def run_stage1(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
) -> float:
    """Executes Stage 1 baseline reproduction."""
    comp, _, _ = run_baseline(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        sub_dir=sub_dir,
    )
    return comp


if __name__ == "__main__":
    run_stage1()
