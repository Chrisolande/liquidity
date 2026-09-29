# Design Document: Clean Validation & Stabilized Competition Submission Pipeline

**Date**: 2026-09-29  
**Branch**: `002-clean-submission-pipeline` (or `main`)  
**Status**: Approved  
**Author**: Antigravity Assistant & Competition Team  

---

## 1. Executive Summary & Objective

With approximately 4 hours remaining before the competition submission deadline, this project stabilizes the modeling repository (`Chrisolande/liquidity`) by:
1. Eliminating validation leakage (ensuring feature screening and target encodings are strictly fold-local).
2. Disabling unconstrained out-of-fold overfitting (multi-round hill climbing against OOF targets).
3. Stripping out test-set prevalence hacking and arbitrary temperature sharpening ($T=1.0$).
4. Disabling test-set pseudo-labeling.
5. Retaining genuine dual-view TabPFN foundation priors (`pfn_phys` and `pfn_champ`) and strictly eliminating surrogate fallback paths (e.g. CatBoost fallback).
6. Fixing known runtime defects (specifically the `best_floor` reference error in `src/stages/stage5_meta_stacker.py`).
7. Preserving raw out-of-fold and test predictions for all individual models before blending.
8. Generating an audit run manifest (`experiment.json`) and clean `submission.csv` via a streamlined single-entry pipeline callable from `pipeline_reproduction.ipynb`.

---

## 2. Architecture & Data Flow

```text
Raw Train & Test CSVs
        │
        ▼
Target-Independent Feature Engineering
(Statistical aggregates, ratios, differences, interaction terms)
        │
        ▼
10-Fold Stratified Split (Seed 42)
        │
   ┌────┴────────────────────────────────────────────────────────┐
   │ Outer Fold Loop (Folds 1 to 10)                             │
   │                                                             │
   │ 1. Slice Fold Partitions:                                   │
   │    x_tr_raw, y_tr, x_va_raw, y_va, x_te_raw                 │
   │                                                             │
   │ 2. Fold-Local Feature Screening:                            │
   │    selected_cols = screen_features(                         │
   │        x_tr_raw, y_tr, cat_cols=cat_cols, k_top=60, seed=42│
   │    )                                                        │
   │    * Computed EXCLUSIVELY on training partition (x_tr, y_tr)│
   │    * Zero validation or test data seen by selector          │
   │                                                             │
   │ 3. Slice Active Columns & Domain Views:                     │
   │    x_tr = x_tr_raw[selected_cols]                           │
   │    x_va = x_va_raw[selected_cols]                           │
   │    x_te = x_te_raw[selected_cols]                           │
   │    dom2_cols, dom3_cols, dom_triage_cols =                  │
   │        extract_domain_feature_subsets(selected_cols)        │
   │                                                             │
   │ 4. Fold-Local Target Encoding (if applicable):              │
   │    x_tr, x_va, x_te = apply_fold_target_encoding(           │
   │        x_tr, y_tr, x_va, x_te, ...                          │
   │    )                                                        │
   │                                                             │
   │ 5. Model Execution on Fold Data:                            │
   │    - CatBoost (GPU, Depth 7, dom2_cols)                     │
   │    - XGBoost (GPU, Depth 4, dom3_cols)                      │
   │    - LightGBM (CPU/GPU, ExtraTrees, selected_cols)          │
   │    - TabPFN (GPU, pfn_phys & pfn_champ, batch_size=5000)    │
   │                                                             │
   │ 6. Out-of-Fold Assignment:                                  │
   │    oof_dict[m][val_idx] = val_preds                         │
   │    test_dict[m] += test_preds / n_splits                    │
   └─────────────────────────────────────────────────────────────┘
        │
        ▼
Validation Checks:
- Verify 100% OOF coverage: assert oof_filled.all()
- Save raw unblended arrays: oof_*.npy, test_*.npy
        │
        ▼
Bounded Regularized Blend (No Hill Climbing, No Sharpening)
- Solve for w_i >= 0, sum(w_i) = 1 minimizing LogLoss
        │
        ▼
Final Deliverables:
- submissions/submission.csv
- artifacts/experiment.json (Run Manifest & LEAKAGE CHECK: PASS)
```

---

## 3. Component Specifications

### 3.1. Clean Pipeline Module (`src/clean_pipeline.py`)
- Coordinates the entire workflow without loading legacy Stage 1 / Baseline anchors.
- Exposes `run_clean_pipeline(seed=42, n_splits=10, run_tabpfn=True, output_dir="artifacts", sub_dir="submissions")`.

### 3.2. Model Zoo & Configurations
1. **CatBoost (`cb_d7_dom26`)**:
   - `depth=7, learning_rate=0.038, l2_leaf_reg=20.0, iterations=650, early_stopping_rounds=40`.
2. **XGBoost (`xgb_d4_dom35`)**:
   - `max_depth=4, learning_rate=0.035, min_child_weight=5.0, subsample=0.85, colsample_bytree=0.75, reg_lambda=2.0, gamma=1.5, n_estimators=600`.
3. **LightGBM (`lgb_extra`)**:
   - `num_leaves=45, learning_rate=0.030, min_child_samples=60, colsample_bytree=0.60, subsample=0.75, extra_trees=True, n_estimators=600`.
4. **TabPFN Foundation Priors**:
   - Views:
     - `pfn_phys`: 14 features (`CHAMPION_14`).
     - `pfn_champ`: 20 features (`CHAMPION_14` + top 6 digital/channel/bank features).
   - Parameters: `n_estimators=4`, `ignore_pretraining_limits=True`, `batch_size=5000`.
   - **Zero Surrogate Fallback**: Strictly remove the CatBoost fallback. If TabPFN or GPU is unavailable, raise an explicit error or toggle off via configuration flag.

### 3.3. Ensembling & Post-Processing
- **Constrained SLSQP / NNLS Weight Optimization**:
  - Direct minimization of LogLoss with bounds $w_i \in [0, 1]$ and equality constraint $\sum w_i = 1$.
  - Evaluated exactly once on OOF predictions.
- **Strictly Disabled**:
  - Zero temperature sharpening ($T = 1.0$).
  - Zero prevalence adjustment multipliers.
  - Zero pseudo-labeling / student distillation.
  - Zero iterative hill-climbing over the validation set.

### 3.4. Defect Corrections
- In `src/stages/stage5_meta_stacker.py`, line 332: Replace undefined `best_floor` reference with `FLOOR`.
- In `src/models/tabpfn_model.py` and pipeline scripts: Eliminate CatBoost fallback branches in TabPFN runner.

### 3.5. In-Model Determinism Enforcement
- Global random seeding via `seed_everything(seed)`:
  - Python `random.seed(seed)`
  - `os.environ["PYTHONHASHSEED"] = str(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
- LightGBM:
  - `random_state=seed, seed=seed, bagging_seed=seed+11, feature_fraction_seed=seed+22, extra_seed=seed+33, data_random_seed=seed+44`
  - `deterministic=True, force_col_wise=True`
- CatBoost:
  - `random_seed=seed`, deterministic categorical processing
- XGBoost:
  - `random_state=seed`
- TabPFN:
  - `random_state=seed + fold`
- Cross-Validation:
  - `StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)`

---

## 4. Verification Protocol

1. **Compilation Check**:
   `python -m compileall src`
2. **OOF Completeness Assertion**:
   Ensure `len(oof) == len(y_true)` and `oof_filled.all()`.
3. **Submission Integrity**:
   Verify `submission.csv` contains 30,000 rows, matching `ID`s, non-null values, and probabilities within $[0.0020, 0.9995]$.
4. **Audit Manifest**:
   Record Git commit, seed, fold count, feature counts per fold, models used, and `LEAKAGE CHECK: PASS`.
5. **Determinism Verification**:
   Verify identical prediction outputs across duplicate runs with seed 42.
