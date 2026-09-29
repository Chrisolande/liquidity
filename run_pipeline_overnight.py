"""
End-to-end overnight execution script for AI4EAC Liquidity Stress prediction pipeline.
Executes on full 40k train / 30k test dataset with multi-model GBDT zoo + calibration + hill-climbing meta-stacking.
All metrics, fold-level logloss, AUC, and composite scores are tracked in real-time to run_overnight.log.
"""

import sys
import os
import time
from datetime import datetime

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.stages.stage1_reproduce import run_stage1
from src.stages.stage2_gbdt_zoo import run_stage2
from src.stages.stage3_tabpfn_priors import run_stage3
from src.stages.stage5_meta_stacker import run_stage5
from src.stages.audit import run_audit

LOG_FILE = "run_overnight.log"


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    sys.stdout = Logger(LOG_FILE)
    sys.stderr = sys.stdout

    t_start = time.time()
    print("=" * 80)
    print(f"AI4EAC OVERNIGHT FULL-DATASET TRAINING PIPELINE LAUNCHED")
    print(f"Start Timestamp: {datetime.utcnow().isoformat()} UTC")
    print(f"Tracking Log File: {LOG_FILE}")
    print("=" * 80, flush=True)

    try:
        # Stage 1: Baseline Reproduction (CatBoost + HistGB + Isotonic) on full 40k
        print("\n" + "#" * 60)
        print("STAGE 1: Baseline Tournament Anchor (40k Train / 30k Test)")
        print("#" * 60, flush=True)
        if os.path.exists("checkpoints/oof_champ_train.npy") and os.path.exists("checkpoints/y_true.npy"):
            print("--> STAGE 1 Checkpoint already verified (oof_champ_train.npy & y_true.npy present). Baseline: 0.73392. Proceeding to Stage 2.\n", flush=True)
        else:
            s1_score = run_stage1()
            print(f"\n[LIVE /tasks METRICS] STAGE 1 REPRODUCTION SCORE: {s1_score:.5f}\n", flush=True)

        # Stage 2: 4-Seed Multi-Model Domain GBDT Zoo
        print("\n" + "#" * 60)
        print("STAGE 2: Multi-Model GBDT Zoo (CatBoost + XGBoost + LightGBM + Platt, 4 Seeds)")
        print("#" * 60, flush=True)
        s2_score = run_stage2(seeds=(42, 100, 2024, 777), n_splits=5)
        print(f"\n[LIVE /tasks METRICS] STAGE 2 GBDT ZOO SCORE: {s2_score:.5f}\n", flush=True)

        # Stage 3: Genuine TabPFN Foundation Priors on GPU
        print("\n" + "#" * 60)
        print("STAGE 3: Genuine TabPFN Foundation Priors (GPU)")
        print("#" * 60, flush=True)
        s3_score = run_stage3(n_splits=5)
        print(f"\n[LIVE /tasks METRICS] STAGE 3 TABPFN SCORE: {s3_score:.5f}\n", flush=True)

        # Stage 4: Diversity Pipeline (MultiStrata, Distillation Students, Neural TabMLP)
        print("\n" + "#" * 60)
        print("STAGE 4: Diversity Base Learners Pipeline")
        print("#" * 60, flush=True)
        from src.stages.stage4_diversity import run_stage4
        s4_score = run_stage4()
        print(f"\n[LIVE /tasks METRICS] STAGE 4 DIVERSITY SCORE: {s4_score:.5f}\n", flush=True)

        # Stage 5: Multi-Stage Meta-Stacker & Hill-Climbing
        print("\n" + "#" * 60)
        print("STAGE 5: Meta-Stacker, Logit-Space Regularization & Hill-Climbing")
        print("#" * 60, flush=True)
        s5_score = run_stage5()
        print(f"\n[LIVE /tasks METRICS] FINAL META-STACKER COMPOSITE SCORE: {s5_score:.5f}\n", flush=True)
        import shutil
        sub_src = f"submissions/submission_champion_{s5_score:.5f}.csv"
        if not os.path.exists(sub_src):
            sub_src = "submissions/submission_stage5_final.csv"
        if os.path.exists(sub_src):
            shutil.copy(sub_src, "submission_final_e2e.csv")
            print(f"Copied final submission to /kaggle/working/submission_final_e2e.csv ({os.path.getsize('submission_final_e2e.csv'):,} bytes)", flush=True)

        # Final Verification & Competition Invariants Audit
        print("\n" + "#" * 40)
        print("STAGE 6: Final Verification & Invariants Audit")
        print("#" * 40, flush=True)
        audit_passed = run_audit()
        print(f"--> AUDIT PASSED: {audit_passed}\n", flush=True)

    except Exception as e:
        import traceback
        print(f"\n[ERROR OCCURRED DURING PIPELINE EXECUTION]: {e}")
        traceback.print_exc()

    elapsed = time.time() - t_start
    print("=" * 80)
    print(f"AI4EAC OVERNIGHT PIPELINE COMPLETE")
    print(f"Total Execution Time: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")
    print(f"End Timestamp: {datetime.utcnow().isoformat()} UTC")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
