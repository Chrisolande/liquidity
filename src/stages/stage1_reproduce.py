"""
Stage 1: Complete Baseline & Metric-Direct Stepwise Hill Climbing Pipeline.
Executes:
1. Baseline multi-seed 3-way GBDT ensemble with feature_engine selection.
2. Metric-direct stepwise Hill Climbing (CatBoost Depth 7 + LightGBM + ExtraTrees).
Produces updated checkpoints/oof_champ_train.npy (0.73522+) and submissions/submission_stage1_hillclimb.csv.
"""

from src.baseline import run_baseline
from src.stages.stage1_hillclimb import run_stage1_hillclimb


def run_stage1(
    train_path: str = None,
    test_path: str = None,
    output_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    k_top_features: int = 60,
    n_splits: int = 10,
    seeds: tuple = (42, 2026),
) -> float:
    """Executes Stage 1 end-to-end: Baseline Anchor + Stepwise HillClimber."""
    print("Stage 1: Baseline and hill climbing pipeline", flush=True)

    # Step 1A: Generate Baseline Anchor
    print(f"Stage 1A: Baseline {len(seeds)}-seed ensemble ({n_splits}-fold CV)", flush=True)
    b_comp, _, _ = run_baseline(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        sub_dir=sub_dir,
        k_top_features=k_top_features,
        n_splits=n_splits,
        seeds=seeds,
    )
    print(f"Baseline anchor completed: Comp={b_comp:.5f}", flush=True)

    # Step 1B: Stepwise Metric-Direct Hill Climbing
    print("Stage 1B: Metric-direct stepwise hill climbing", flush=True)
    final_comp, _, _ = run_stage1_hillclimb(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        sub_dir=sub_dir,
        k_top_features=k_top_features,
        n_splits=n_splits,
        seeds=seeds,
    )
    print(f"Stage 1 final score: Comp={final_comp:.5f}\n", flush=True)
    return final_comp


if __name__ == "__main__":
    run_stage1()

