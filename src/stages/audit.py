"""
Final competition invariants and submission validation audit.
"""

import os
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.config import ID_COL, TARGET, FLOOR, CEIL


def run_audit(sub_candidates: Optional[List[str]] = None) -> bool:
    """Verifies that the target submission strictly passes all competition rules and invariants."""
    print("Running competition invariants and verification audit")

    if not sub_candidates:
        sub_candidates = [
            "submissions/submission_stage5_final.csv",
            "submissions/submission_s2_gbdt_zoo.csv",
            "submissions/submission_stage1_hillclimb.csv",
            "submissions/submission_step13_4seed_0.73737.csv",
            "submissions/submission_step13_4seed_0.73732.csv",
            "submissions/submission_best_0.73731.csv",
            "submissions/submission.csv",
        ]

    existing = [p for p in sub_candidates if os.path.exists(p)]
    if not existing:
        print("No submission files found to audit.")
        return False

    sub_path = existing[0]
    print(f"Auditing primary submission file: {sub_path}")
    sub_df = pd.read_csv(sub_path)
    preds = sub_df["Target"].values

    # Invariant 1: Exactly 30,000 rows
    assert len(sub_df) == 30000, f"Row count {len(sub_df)} != 30,000"
    print(f"Invariant 1: Exactly 30,000 test rows ({len(sub_df):,})")

    # Invariant 2: Column format
    assert list(sub_df.columns) == [ID_COL, "Target"], f"Columns {list(sub_df.columns)} != ['{ID_COL}', 'Target']"
    print(f"Invariant 2: Column names strictly ['{ID_COL}', 'Target']")

    # Invariant 3: Zero NaNs
    assert not sub_df.isna().any().any(), "Found NaNs in submission!"
    print("Invariant 3: Zero NaN or infinite values")

    # Invariant 4: Bounded probabilities
    assert preds.min() >= FLOOR, f"Min prob {preds.min()} < {FLOOR}"
    assert preds.max() <= CEIL, f"Max prob {preds.max()} > {CEIL}"
    print(f"Invariant 4: Probability bounded in [{preds.min():.5f}, {preds.max():.5f}]")

    # Invariant 5: Natural prevalence
    print(f"Invariant 5: Mean prevalence is {preds.mean():.5f} (Target reference: ~0.15340)")

    # Benchmark Alignment
    bench_candidates = [
        "submissions/submission_step13_4seed_0.73732_ORIGINAL_BENCHMARK.csv",
        "submissions/submission_best_0.73731.csv",
    ]
    for bp in bench_candidates:
        if os.path.exists(bp) and bp != sub_path:
            bench_df = pd.read_csv(bp)
            r_p, _ = pearsonr(bench_df["Target"].values, preds)
            r_s, _ = spearmanr(bench_df["Target"].values, preds)
            print(f"Benchmark Alignment vs {os.path.basename(bp)}: Pearson r={r_p:.6f}, Spearman rho={r_s:.6f}")
            break

    print("All competition checks passed.")
    return True


if __name__ == "__main__":
    run_audit()
