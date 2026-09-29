# Design Document: Clean Validation, Stage 5 Preservation & Auditable Submission Pipeline

**Date**: 2026-09-29  
**Branch**: `002-clean-submission-pipeline`  
**Status**: Approved  
**Author**: Antigravity Assistant & Competition Engineering Team  

---

## 1. Context & Architectural Principles

The repository contains an advanced, high-performing solution currently ranked approximately **#2 on the competition leaderboard**. 

The goal of this work is **not** to simplify the model into a primitive baseline or delete experimental work. Rather, the objective is to:
1. **Preserve the current improved architecture**: Stages 1 through 5, GBDT zoo, distillation students, diversity models, and the Stage 5 grand-teacher / meta-stacker remain intact.
2. **Establish an auditable, leak-free clean validation and submission path**: A standalone execution entry point guaranteeing strict fold-local feature selection, fold-local target encoding, zero test pseudo-labels, zero temperature sharpening ($T=1.0$), zero prevalence adjustment, and zero unconstrained OOF hill climbing.
3. **Explicitly distinguish clean validation from experimental competition optimization**: Clearly document and demarcate experimental components.
4. **Fix known runtime defects**: Resolve the undefined `best_floor` reference in `src/stages/stage5_meta_stacker.py`.
5. **Ensure reproducibility**: Enforce in-model determinism, save raw per-model prediction arrays, and generate an audit `experiment.json` manifest with an explicit `LEAKAGE CHECK: PASS`.

---

## 2. Architecture & Data Flow

### 2.1. Clean Validation & Submission Path

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
- Save raw unblended arrays: artifacts/oof_*.npy, test_*.npy
        │
        ▼
Bounded Regularized Blend (No Hill Climbing, No Sharpening)
- Solve for w_i >= 0, sum(w_i) = 1 minimizing LogLoss
        │
        ▼
Final Deliverables:
- submissions/submission_clean.csv
- artifacts/experiment.json (Run Manifest & LEAKAGE CHECK: PASS)
```

### 2.2. Preserved Experimental Tournament Path

```text
Stage 1 (Baseline Anchor + Hill Climbing)
   │
   ▼
Stage 2 (Domain GBDT Zoo)
   │
   ▼
Stage 3 (Dual TabPFN Foundation Priors)
   │
   ▼
Stage 4 (Diversity: MultiStrata, Distillation Students, TabMLP)
   │
   ▼
Stage 5 (Grand-Teacher Logit-Space L2 Meta-Stacker & Adaptive Optimizer)
   │
   ▼
submissions/submission_stage5_final.csv (#2 Leaderboard Tournament Model)
```

---

## 3. Component Audits & Surgical Modifications

### 3.1. Stage 5 Meta-Stacker (`src/stages/stage5_meta_stacker.py`)
- **Status**: **PRESERVED & AUDITED**
- **Defect Fix**: On line 332, replace undefined `best_floor` reference with `FLOOR`.
- **Validation**: Ensures meta-stacker inputs are genuine OOF prediction vectors from previous stages, and cross-fitted calibration is applied.
- **Classification**: Tagged as *Experimental Meta-Stacking Path* because it optimizes alpha/weights across the combined ensemble pool.

### 3.2. Clean Pipeline Runner (`src/clean_pipeline.py`)
- **Status**: **NEW MODULE**
- Dedicated entry point providing the leak-free, scientifically defensible evaluation and submission path.
- Strictly encapsulates fold-local screening, fold-local target encoding, model training, raw prediction export, single-pass regularized ensembling, and run-manifest generation.

### 3.3. Foundation TabPFN (`src/models/tabpfn_model.py`)
- **Status**: **AUDITED & CLEANED**
- Enforces genuine foundation TabPFN execution across `pfn_phys` (14 features) and `pfn_champ` (20 features) using `ignore_pretraining_limits=True`.
- Eliminates any CatBoost surrogate fallback branches; if GPU or TabPFN is unavailable, raises an explicit error or allows clean gating via `run_tabpfn=False`.

### 3.4. Notebook Integration (`pipeline_reproduction.ipynb`)
- **Status**: **STREAMLINED WITH DUAL EXECUTION**
  - Section A: Clean, leak-free submission pipeline execution.
  - Section B: Preserved Stage 1 through Stage 5 reproduction pipeline for tournament auditing.

---

## 4. Run Manifest Specification (`experiment.json`)

The clean pipeline outputs a standardized machine-readable JSON manifest:
```json
{
  "timestamp": "2026-09-29T...",
  "git_commit": "<hash>",
  "seed": 42,
  "n_splits": 10,
  "fold_local_feature_selection": true,
  "fold_local_target_encoding": true,
  "pseudo_labeling": false,
  "temperature_sharpening": false,
  "prevalence_adjustment": false,
  "oof_hill_climbing": false,
  "tabpfn_enabled": true,
  "tabpfn_views": ["pfn_phys", "pfn_champ"],
  "tabpfn_surrogate_fallback": false,
  "models_trained": ["cb_d7_dom26", "xgb_d4_dom35", "lgb_extra", "pfn_phys", "pfn_champ"],
  "selected_features_per_fold": [60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
  "oof_evaluations": {
    "auc": 0.73...,
    "logloss": 0.23...,
    "comp_score": 0.73...
  },
  "leakage_audit_status": "PASS",
  "submission_path": "submissions/submission_clean.csv"
}
```
