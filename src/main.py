"""
Main command-line orchestrator for AI4EAC Liquidity Stress pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import ACTIVE_SEEDS, ENABLE_STAGE3
from src.stages.stage1_reproduce import run_stage1
from src.stages.stage2_gbdt_zoo import run_stage2
from src.stages.stage3_tabpfn_priors import run_stage3
from src.stages.stage4_diversity import run_stage4
from src.stages.stage5_meta_stacker import run_stage5
from src.stages.audit import run_audit


def main():
    parser = argparse.ArgumentParser(description="AI4EAC Liquidity Stress Modular Prediction Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "all", "audit"],
        help="Pipeline stage to execute (1: Reproduce, 2: GBDT Zoo, 3: TabPFN, 4: Diversity, 5: Meta-Stacker, all: Full pipeline, audit: Verification)",
    )
    parser.add_argument("--train", type=str, default=None, help="Path to raw Train.csv")
    parser.add_argument("--test", type=str, default=None, help="Path to raw Test.csv")
    parser.add_argument(
        "--enable-stage3",
        action="store_true",
        default=ENABLE_STAGE3,
        help="Enable TabPFN Foundation training in Stage 3 (default: True)",
    )
    args = parser.parse_args()

    stage = args.stage.lower()

    if stage in ["1", "all"]:
        run_stage1(train_path=args.train, test_path=args.test, seeds=ACTIVE_SEEDS)
    if stage in ["2", "all"]:
        run_stage2(train_path=args.train, test_path=args.test, seeds=ACTIVE_SEEDS)
    if stage == "3":
        run_stage3(train_path=args.train, test_path=args.test, enabled=True)
    elif stage == "all":
        run_stage3(train_path=args.train, test_path=args.test, enabled=args.enable_stage3)
    if stage in ["4", "all"]:
        run_stage4(train_path=args.train, test_path=args.test, use_tabpfn=args.enable_stage3)
    if stage in ["5", "all"]:
        run_stage5(test_path=args.test, use_tabpfn=args.enable_stage3)
    if stage in ["audit", "all"]:
        run_audit()


if __name__ == "__main__":
    main()
