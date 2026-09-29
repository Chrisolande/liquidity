"""
Stage 5: Multi-Stage Meta-Stacker, Logit-Space L2 Regularization, Hill Climbing, and Final Submission Generator.
Integrates all model streams into the winning composite ensemble.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

# Ensure repo root is in sys.path for direct script execution by reviewers
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import (
    ID_COL,
    TARGET,
    FLOOR,
    CEIL,
    ENABLE_STAGE3,
    get_default_dataset_paths,
)
from src.ensemble.calibration import platt_scaling_calibrate
from src.metrics import competition_score, print_correlation_matrix


def run_stage5(
    ckpt_dir: str = "checkpoints",
    sub_dir: str = "submissions",
    test_path: str = None,
    use_tabpfn: bool = None,
) -> float:
    """Executes upgraded Stage 5 Logit-Space L2 Meta-Stacker, Hill Climbing, and final verification."""
    if use_tabpfn is None:
        use_tabpfn = ENABLE_STAGE3
    print("=" * 80)
    print("STAGE 5: UPGRADED 10-FOLD LOGIT-SPACE L2 META-STACKER & HILL CLIMBER")
    print("=" * 80, flush=True)

    os.makedirs(sub_dir, exist_ok=True)
    test_ids = None
    if test_path and os.path.exists(test_path):
        test_ids = pd.read_csv(test_path)[ID_COL]
    else:
        for p_cand in [
            "Test.csv",
            "test.csv",
            "/kaggle/input/datasets/chrisolande/zindi-competition/Test.csv",
            "chrisolande/zindi-competition/Test.csv",
            os.path.join(sub_dir, "submission_s2_gbdt_zoo.csv"),
            os.path.join("pulled", "submission_s2_gbdt_zoo.csv"),
            os.path.join(sub_dir, "submission_multistrata.csv"),
            os.path.join("pulled", "submission_multistrata.csv"),
            "submission_30k.csv",
        ]:
            if os.path.exists(p_cand):
                try:
                    test_ids = pd.read_csv(p_cand)[ID_COL]
                    break
                except Exception:
                    continue
    if test_ids is None:
        raise FileNotFoundError("Could not find Test.csv or existing submission to extract test IDs.")
    n_test = len(test_ids)

    # Load ground truth y_true
    y_true = None
    y_candidates = [
        os.path.join(ckpt_dir, "y_true.npy"),
        os.path.join("pulled", "y_true.npy"),
    ]
    for y_cand in y_candidates:
        if os.path.exists(y_cand):
            y_true = np.load(y_cand)
            break
    if y_true is None:
        for p_cand in [
            "Train.csv",
            "train.csv",
            "/kaggle/input/datasets/chrisolande/zindi-competition/Train.csv",
            "chrisolande/zindi-competition/Train.csv",
        ]:
            if os.path.exists(p_cand):
                try:
                    y_true = pd.read_csv(p_cand)[TARGET].values
                    break
                except Exception:
                    continue
    if y_true is None:
        raise FileNotFoundError("Could not find y_true.npy or Train.csv to extract ground truth targets.")
    n_train = len(y_true)

    oof_dict: Dict[str, np.ndarray] = {}
    test_dict: Dict[str, np.ndarray] = {}

    # 1. Champion Anchor
    s1_oof_candidates = [
        os.path.join(ckpt_dir, "oof_champ_train.npy"),
        os.path.join("pulled", "oof_champ_train.npy"),
    ]
    s1_oof_path = next((p for p in s1_oof_candidates if os.path.exists(p)), None)
    s1_sub_candidates = [
        os.path.join(sub_dir, "submission_stage1_hillclimb.csv"),
        os.path.join("pulled", "submission_stage1_hillclimb.csv"),
        os.path.join(sub_dir, "submission_baseline.csv"),
        os.path.join("pulled", "submission_baseline.csv"),
    ]
    s1_sub_path = next((p for p in s1_sub_candidates if os.path.exists(p)), None)
    if s1_oof_path and s1_sub_path:
        oof_dict["champ_anchor"] = np.load(s1_oof_path)
        test_dict["champ_anchor"] = pd.read_csv(s1_sub_path)["Target"].values
        assert len(oof_dict["champ_anchor"]) == n_train, f"champ_anchor OOF length ({len(oof_dict['champ_anchor'])}) != n_train ({n_train})"
        assert len(test_dict["champ_anchor"]) == n_test, f"champ_anchor test length ({len(test_dict['champ_anchor'])}) != n_test ({n_test})"
        print(f"Loaded Champion Anchor: {s1_oof_path}")

    # 2. Step 13 Benchmark Anchor (if available)
    s13_oof_candidates = [
        os.path.join(ckpt_dir, "oof_step13_0.73439.npy"),
        os.path.join("pulled", "oof_step13_0.73439.npy"),
    ]
    s13_sub_candidates = [
        os.path.join(sub_dir, "submission_step13_4seed_0.73737.csv"),
        os.path.join("pulled", "submission_step13_4seed_0.73737.csv"),
        os.path.join(sub_dir, "submission_best_0.73731.csv"),
        os.path.join("pulled", "submission_best_0.73731.csv"),
    ]
    s13_oof_p = next((p for p in s13_oof_candidates if os.path.exists(p)), None)
    s13_sub_p = next((p for p in s13_sub_candidates if os.path.exists(p)), None)
    if s13_oof_p and s13_sub_p:
        oof_dict["step13"] = np.load(s13_oof_p)
        test_dict["step13"] = pd.read_csv(s13_sub_p)["Target"].values
        print(f"Loaded Step 13 Benchmark: {s13_oof_p}")

    # 3. 4-Seed GBDT Zoo Streams
    zoo_candidates = [
        os.path.join(ckpt_dir, "gbdt_zoo_4seed.npz"),
        os.path.join("pulled", "gbdt_zoo_4seed.npz"),
        os.path.join(ckpt_dir, "gbdt_balanced_dom1_p2_seeds4.npz"),
        os.path.join("pulled", "gbdt_balanced_dom1_p2_seeds4.npz"),
    ]
    zoo_p = next((p for p in zoo_candidates if os.path.exists(p)), None)
    if zoo_p:
        zoo = np.load(zoo_p)
        for k in ["cb_d7_dom26", "cb_d6_dom35", "xgb_d4_dom35", "xgb_d4_triage", "lgb_extra"]:
            if f"oof_{k}" in zoo and f"test_{k}" in zoo:
                oof_dict[k] = zoo[f"oof_{k}"]
                test_dict[k] = zoo[f"test_{k}"]

    # Dedicated Independent Stage 2 Domain Ensemble
    s2_domain_oof_candidates = [
        os.path.join(ckpt_dir, "oof_stage2_domain.npy"),
        os.path.join("pulled", "oof_stage2_domain.npy"),
    ]
    s2_domain_test_candidates = [
        os.path.join(ckpt_dir, "test_stage2_domain.npy"),
        os.path.join("pulled", "test_stage2_domain.npy"),
    ]
    s2_oof_p = next((p for p in s2_domain_oof_candidates if os.path.exists(p)), None)
    s2_test_p = next((p for p in s2_domain_test_candidates if os.path.exists(p)), None)
    if s2_oof_p and s2_test_p:
        oof_dict["s2_domain_ensemble"] = np.load(s2_oof_p)
        test_dict["s2_domain_ensemble"] = np.load(s2_test_p)
        print(f"Loaded Independent Stage 2 Domain Ensemble: {s2_oof_p}")

    # 4. MultiStrata Iterative Stratification Stream
    ms_candidates = [
        os.path.join(ckpt_dir, "multistrata.npz"),
        os.path.join("pulled", "multistrata.npz"),
    ]
    ms_p = next((p for p in ms_candidates if os.path.exists(p)), None)
    if ms_p:
        ms = np.load(ms_p)
        k_oof = next((x for x in ["oof_multistrata", "oof_raw"] if x in ms), None)
        k_te = next((x for x in ["test_multistrata", "test_raw"] if x in ms), None)
        if k_oof and k_te:
            oof_dict["multistrata"] = ms[k_oof]
            test_dict["multistrata"] = ms[k_te]

    # 5. TabPFN Priors
    if use_tabpfn:
        pfn_candidates = [
            os.path.join(ckpt_dir, "tabpfn.npz"),
            os.path.join("pulled", "tabpfn.npz"),
            os.path.join(ckpt_dir, "tabpfn_priors.npz"),
            os.path.join("pulled", "tabpfn_priors.npz"),
        ]
        pfn_p = next((p for p in pfn_candidates if os.path.exists(p)), None)
        if pfn_p:
            pfn = np.load(pfn_p)
            for k in ["tabpfn", "pfn_phys", "pfn_champ"]:
                oof_k = next((x for x in [f"oof_{k}", k] if x in pfn), None)
                test_k = next((x for x in [f"test_{k}", "test_tabpfn"] if x in pfn), None)
                if oof_k and test_k:
                    oof_dict[f"pfn_{k}"] = pfn[oof_k]
                    test_dict[f"pfn_{k}"] = pfn[test_k]
    else:
        print("[Stage 5] Skipping TabPFN priors inclusion (use_tabpfn=False).")

    # 6. Distillation Students
    dist_candidates = [
        os.path.join(ckpt_dir, "distill_student_10fold.npz"),
        os.path.join("pulled", "distill_student_10fold.npz"),
    ]
    dist_p = next((p for p in dist_candidates if os.path.exists(p)), None)
    if dist_p:
        dist = np.load(dist_p)
        k_cb = next((x for x in ["oof_cb_student", "oof_student_cb"] if x in dist), None)
        k_xgb = next((x for x in ["oof_xgb_student", "oof_student_xgb"] if x in dist), None)
        k_lgb = next((x for x in ["oof_lgb_student", "oof_student_lgb"] if x in dist), None)
        if k_cb:
            t_cb = next((x for x in ["test_cb_student", "test_student_cb"] if x in dist), None)
            oof_dict["student_cb"] = dist[k_cb]
            test_dict["student_cb"] = dist[t_cb] if t_cb else np.full(n_test, float(dist[k_cb].mean()))
        elif k_lgb:
            t_lgb = next((x for x in ["test_lgb_student", "test_student_lgb"] if x in dist), None)
            oof_dict["student_lgb"] = dist[k_lgb]
            test_dict["student_lgb"] = dist[t_lgb] if t_lgb else np.full(n_test, float(dist[k_lgb].mean()))
        if k_xgb:
            t_xgb = next((x for x in ["test_xgb_student", "test_student_xgb"] if x in dist), None)
            oof_dict["student_xgb"] = dist[k_xgb]
            test_dict["student_xgb"] = dist[t_xgb] if t_xgb else np.full(n_test, float(dist[k_xgb].mean()))

    print(f"\nTotal candidate streams assembled: {len(oof_dict)}")
    valid_streams = {}
    valid_test = {}
    for k in oof_dict:
        k_ll, k_auc, k_comp = competition_score(y_true, oof_dict[k])
        print(f"  Stream {k:<18}: Comp={k_comp:.5f} | AUC={k_auc:.5f} | LL={k_ll:.5f}")
        # Safeguard: filter out broken or severely uncalibrated streams (e.g. exploding MLPs)
        if k_auc >= 0.90 and k_ll <= 0.27:
            valid_streams[k] = oof_dict[k]
            valid_test[k] = test_dict[k]

    print(f"Retained {len(valid_streams)} quality streams for Meta-Stacker.")

    # Method 1: Correlation Pruning Filter (Drop redundant collinear streams with r > 0.996)
    print("\nPruning collinear streams (threshold r > 0.996)", flush=True)
    solo_scores = {k: competition_score(y_true, valid_streams[k])[2] for k in valid_streams}
    # Prioritize champ_anchor, then order by solo composite score
    priority_order = (
        ["champ_anchor"] if "champ_anchor" in valid_streams else []
    ) + sorted([k for k in valid_streams if k != "champ_anchor"], key=lambda m: solo_scores[m], reverse=True)

    pruned_streams = {}
    pruned_test = {}
    dropped_info = []
    max_corr = 0.996

    for m in priority_order:
        m_oof = valid_streams[m]
        drop = False
        drop_reason = None
        for s in pruned_streams:
            r = np.corrcoef(m_oof, pruned_streams[s])[0, 1]
            if r > max_corr:
                drop = True
                drop_reason = f"r={r:.4f} with {s} (Comp={solo_scores[s]:.5f})"
                break
        if drop:
            dropped_info.append((m, solo_scores[m], drop_reason))
        else:
            pruned_streams[m] = valid_streams[m]
            pruned_test[m] = valid_test[m]

    for m, sc, r in dropped_info:
        print(f"  Pruned stream: {m:<16} (Comp={sc:.5f}) - {r}", flush=True)

    print(f"\nRetained {len(pruned_streams)} high-diversity streams for Meta-Stacker:", flush=True)
    for s in pruned_streams:
        print(f"  Kept stream: {s:<16} (Comp={solo_scores[s]:.5f})", flush=True)

    valid_streams = pruned_streams
    valid_test = pruned_test
    print_correlation_matrix(valid_streams)

    # Convert to logit space
    model_names = list(valid_streams.keys())
    X_oof_logits = np.column_stack([logit(np.clip(valid_streams[m], 1e-4, 1.0 - 1e-4)) for m in model_names])
    X_test_logits = np.column_stack([logit(np.clip(valid_test[m], 1e-4, 1.0 - 1e-4)) for m in model_names])

    # 10-Fold Cross-Validated L2 Logit-Space Stacker with Analytical Gradient
    print("\nOptimizing logit stacker L2 regularization (10-fold CV)", flush=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    best_comp = 0.0
    best_oof = None
    best_test = None
    best_alpha = 0.0

    alphas = [1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3]
    for alpha in alphas:
        oof_pred = np.zeros(n_train)
        test_pred = np.zeros(n_test)

        for tr_idx, val_idx in skf.split(X_oof_logits, y_true):
            X_tr, y_tr = X_oof_logits[tr_idx], y_true[tr_idx]
            X_va = X_oof_logits[val_idx]

            def loss_fn(params):
                w = params[:-1]
                b = params[-1]
                l = np.dot(X_tr, w) + b
                p = expit(l)
                p_c = np.clip(p, 1e-7, 1.0 - 1e-7)
                ce = -np.mean(y_tr * np.log(p_c) + (1.0 - y_tr) * np.log(1.0 - p_c))
                reg = 0.5 * alpha * np.sum(w**2)
                diff = (p - y_tr) / len(y_tr)
                grad_w = np.dot(X_tr.T, diff) + alpha * w
                grad_b = np.sum(diff)
                return ce + reg, np.append(grad_w, grad_b)

            w0 = np.zeros(len(model_names) + 1)
            if "champ_anchor" in model_names:
                w0[model_names.index("champ_anchor")] = 0.4
            if "multistrata" in model_names:
                w0[model_names.index("multistrata")] = 0.4
            if "step13" in model_names:
                w0[model_names.index("step13")] = 0.2

            res = minimize(loss_fn, w0, method="L-BFGS-B", jac=True, options={"maxiter": 250})
            w_opt = res.x[:-1]
            b_opt = res.x[-1]

            oof_pred[val_idx] = expit(np.dot(X_va, w_opt) + b_opt)
            test_pred += expit(np.dot(X_test_logits, w_opt) + b_opt) / 10.0

        p_eval = np.clip(oof_pred, 0.0020, 0.9980)
        ll, auc, comp = competition_score(y_true, p_eval)
        print(f"  Alpha {alpha:8.6f}: LL={ll:.5f} | AUC={auc:.5f} | Comp={comp:.5f}", flush=True)
        if comp > best_comp:
            best_comp = comp
            best_oof = p_eval
            best_test = test_pred
            best_alpha = alpha

    print(f"\nOptimal Alpha={best_alpha:.6f} with Logit Stacker OOF Score = {best_comp:.5f}")

    # Final Single-Stage Cross-Fitted Calibration
    print("Applying single-stage cross-fitted calibration", flush=True)
    final_oof, final_test, _ = platt_scaling_calibrate(
        oof_prob=best_oof,
        test_prob=best_test,
        y_true=y_true,
        n_splits=10,
        seed=42,
    )

    final_ll, final_auc, final_comp = competition_score(y_true, final_oof)
    print(f"Stage 5 Final Meta-Stacker: Comp={final_comp:.5f} | AUC={final_auc:.5f} | LL={final_ll:.5f} | Floor={FLOOR:.4f} | Alpha={best_alpha:.6f}", flush=True)

    out_csv = os.path.join(sub_dir, "submission_stage5_final.csv")
    sub_df = pd.DataFrame({ID_COL: test_ids, "Target": final_test})
    sub_df.to_csv(out_csv, index=False)

    champ_out_csv = os.path.join(sub_dir, f"submission_champion_{final_comp:.5f}.csv")
    sub_df.to_csv(champ_out_csv, index=False)

    np.save(os.path.join(ckpt_dir, f"oof_stage5_{final_comp:.5f}.npy"), final_oof)
    print(f"Saved final submission to {out_csv} and {champ_out_csv}")
    return final_comp


if __name__ == "__main__":
    run_stage5()
