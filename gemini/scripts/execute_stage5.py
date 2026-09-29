#!/usr/bin/env python3
import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath("."))
from kaggle_runner import execute_code, pull_file

def pull_stage_files(prefix_filter=None):
    os.makedirs("pulled", exist_ok=True)
    list_code = """
import glob, os, json
all_files = glob.glob('checkpoints/**/*', recursive=True) + glob.glob('submissions/**/*', recursive=True)
files = [f for f in all_files if os.path.isfile(f) and f.endswith(('.csv', '.npy', '.npz', '.json'))]
print('REMOTE_FILES_JSON:' + json.dumps(files))
"""
    out = execute_code(list_code, print_stream=False)
    for line in out.splitlines():
        if line.startswith("REMOTE_FILES_JSON:"):
            remote_files = json.loads(line.replace("REMOTE_FILES_JSON:", ""))
            for rf in sorted(remote_files):
                fname = os.path.basename(rf)
                if prefix_filter and not any(p in fname for p in prefix_filter):
                    continue
                local_target = os.path.join("pulled", fname)
                print(f"Pulling {rf} -> {local_target}...", flush=True)
                try:
                    pull_file(rf, local_target)
                except Exception as e:
                    print(f"  [ERROR pulling {rf}]: {e}", flush=True)

def main():
    print("=" * 80)
    print("EXECUTING STAGE 5: META-STACKER, HILL CLIMBING & TAIL SMOOTHING")
    print("=" * 80, flush=True)

    stage5_code = """
import sys, time
to_remove = [k for k in list(sys.modules.keys()) if k.startswith('src') or k.startswith('experiments')]
for k in to_remove:
    del sys.modules[k]

from src.stages.stage5_meta_stacker import run_stage5
t0 = time.time()
score = run_stage5()
elapsed = time.time() - t0
print(f"STAGE 5 COMPLETE in {elapsed/60:.2f} min ({elapsed:.1f}s) | Final Composite Score: {score:.5f}", flush=True)
"""
    execute_code(stage5_code, timeout=1800.0)

    print("\n" + "=" * 80)
    print("RUNNING FINAL COMPETITION INVARIANTS AUDIT")
    print("=" * 80, flush=True)
    audit_code = """
import sys
to_remove = [k for k in list(sys.modules.keys()) if k.startswith('src')]
for k in to_remove:
    del sys.modules[k]

from src.stages.audit import run_audit
passed = run_audit()
print(f"AUDIT STATUS: {'PASSED' if passed else 'FAILED'}", flush=True)
"""
    execute_code(audit_code, timeout=300.0)

    print("\n" + "=" * 80)
    print("PULLING STAGE 5 DELIVERABLES TO LOCAL pulled/")
    print("=" * 80, flush=True)

    stage5_targets = [
        "stage5",
        "champion",
        "final",
        "submission",
        "oof_stage5",
    ]
    pull_stage_files(prefix_filter=stage5_targets)
    print("\nSTAGE 5 EXECUTION AND PULL COMPLETE!", flush=True)

if __name__ == "__main__":
    main()
