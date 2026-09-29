#!/usr/bin/env python3
"""
Remote runner to execute Stage 4 Diversity, Stage 5 Meta-Stacker, Audit,
and retrieve all deliverables to local pulled/.
"""
import os
import sys
import time
import subprocess

from kaggle_runner import execute_code, push_via_exec, pull_file

def main():
    print("=== PREPARING STAGE 4 & 5 REMOTE EXECUTION ===", flush=True)

    # 1. Push updated code to Kaggle
    files_to_push = [
        "src/models/multistrata.py",
        "src/stages/stage4_diversity.py",
        "src/stages/stage5_meta_stacker.py",
        "src/stages/audit.py",
    ]
    for f in files_to_push:
        if os.path.exists(f):
            print(f"Pushing {f}...", flush=True)
            push_via_exec(f, f)

    # 2. Check TabPFN checkpoint & assemble standard tabpfn.npz if needed
    prep_pfn_code = """
import os, glob
import numpy as np

sweep_p = "checkpoints/experiments/tabpfn_view_sweep.npz"
tabpfn_p = "checkpoints/tabpfn.npz"

if os.path.exists(sweep_p):
    sweep = np.load(sweep_p)
    print("Available keys in sweep:", sweep.files)
    save_dict = {}
    for k in sweep.files:
        save_dict[k] = sweep[k]
    
    # Identify best OOF for standard keys
    if "oof_pfn_phys_legacy" in sweep:
        save_dict["oof_pfn_phys"] = sweep["oof_pfn_phys_legacy"]
    if "oof_pfn_sweetspot_50" in sweep:
        save_dict["oof_pfn_champ"] = sweep["oof_pfn_sweetspot_50"]
        save_dict["oof_tabpfn"] = sweep["oof_pfn_sweetspot_50"]
    elif "oof_pfn_phys_legacy" in sweep:
        save_dict["oof_tabpfn"] = sweep["oof_pfn_phys_legacy"]
        
    np.savez_compressed(tabpfn_p, **save_dict)
    np.savez_compressed("checkpoints/tabpfn_priors.npz", **save_dict)
    print("Assembled", tabpfn_p, "and checkpoints/tabpfn_priors.npz with keys:", list(save_dict.keys()))
else:
    print("Note: sweep_p not found yet at", sweep_p)
"""
    print("\nAssembling TabPFN checkpoints on Kaggle...", flush=True)
    execute_code(prep_pfn_code)

    # 3. Run Stage 4 Diversity Pipeline
    print("\n=== RUNNING STAGE 4: DIVERSITY PIPELINE ===", flush=True)
    stage4_code = """
import sys, time
to_remove = [k for k in sys.modules if k.startswith('src')]
for k in to_remove: del sys.modules[k]

from src.stages.stage4_diversity import run_stage4
t0 = time.time()
score = run_stage4()
print(f"STAGE 4 COMPLETE in {(time.time()-t0)/60:.2f} min | Composite Score: {score:.5f}", flush=True)
"""
    execute_code(stage4_code)

    # 4. Run Stage 5 Meta-Stacker & Tail Smoothing
    print("\n=== RUNNING STAGE 5: META-STACKER & TAIL SMOOTHING ===", flush=True)
    stage5_code = """
import sys, time
to_remove = [k for k in sys.modules if k.startswith('src')]
for k in to_remove: del sys.modules[k]

from src.stages.stage5_meta_stacker import run_stage5
t0 = time.time()
score = run_stage5()
print(f"STAGE 5 COMPLETE in {(time.time()-t0)/60:.2f} min | Final Composite Score: {score:.5f}", flush=True)
"""
    execute_code(stage5_code)

    # 5. Run Competition Audit
    print("\n=== RUNNING FINAL COMPETITION AUDIT ===", flush=True)
    audit_code = """
import sys
to_remove = [k for k in sys.modules if k.startswith('src')]
for k in to_remove: del sys.modules[k]

from src.stages.audit import run_audit
passed = run_audit()
print(f"AUDIT STATUS: {'PASSED' if passed else 'FAILED'}", flush=True)
"""
    execute_code(audit_code)

    # 6. List and Pull Deliverables
    print("\n=== PULLING DELIVERABLES TO LOCAL pulled/ ===", flush=True)
    os.makedirs("pulled", exist_ok=True)
    list_code = """
import glob, json
files = glob.glob('checkpoints/**/*', recursive=True) + glob.glob('submissions/**/*', recursive=True)
print('REMOTE_FILES_JSON:' + json.dumps(files))
"""
    out = execute_code(list_code, print_stream=False)
    for line in out.splitlines():
        if line.startswith("REMOTE_FILES_JSON:"):
            import json
            remote_files = json.loads(line.replace("REMOTE_FILES_JSON:", ""))
            for rf in remote_files:
                if os.path.isfile(rf) or not rf.endswith('/'):
                    fname = os.path.basename(rf)
                    if fname.endswith(('.csv', '.npy', '.npz', '.json')):
                        local_target = os.path.join("pulled", fname)
                        print(f"Pulling {rf} -> {local_target}...")
                        try:
                            pull_file(rf, local_target)
                        except Exception as e:
                            print(f"Failed pulling {rf}: {e}")

    print("\n=== ALL STAGES AND AUDIT COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
