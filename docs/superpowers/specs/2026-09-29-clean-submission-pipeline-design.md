# Design Document: Clean Validation, Independent Domain Learning, and Stabilized Competition Submission Pipeline

**Date**: 2026-09-29  
**Branch**: `002-clean-submission-pipeline`  
**Status**: Approved  
**Author**: Antigravity Assistant & Competition Engineering Team  

---

## 1. Core Architectural Principle

The competition solution is currently ranked approximately **#2 on the leaderboard**. This design preserves the existing multi-stage tournament architecture while making **Stage 2 an independent domain-specialization stage** and establishing **Stage 5 as the primary meeting point where Stage 1 and Stage 2 predictions interact through genuine out-of-fold predictions**.

The pipeline maintains clear role separation:
1. **Stage 1**: General-purpose baseline & champion.
2. **Stage 2**: Independent domain-specialized model zoo.
3. **Stage 3**: Additional model family & diversity source (TabPFN).
4. **Stage 4**: Additional diversity experiments (MultiStrata, TabMLP, Distillation).
5. **Stage 5**: Meta-stacking / final combination of genuinely out-of-fold predictions.

### Critical Stage 2 Rule
**Stage 2 must NOT receive the Stage 1 champion as an input to its own model fitting, blend optimization, or calibration.**
Stage 2 evaluates whether domain-specialized models learn complementary predictive signal independently. The Stage 2 ensemble contains only Stage 2 domain models (`cb_d7_dom26`, `cb_d6_dom35`, `xgb_d4_dom35`, `xgb_d4_triage`, `lgb_extra`), never `oof_stage1_anchor`.

---

## 2. Architecture & Data Flow

```text
               [Raw Train.csv & Test.csv]
                           │
                           ▼
          [Target-Independent Feature Engineering]
                           │
                           ▼
             [10-Fold Stratified Split seed 42]
                           │
       ┌───────────────────┴───────────────────┐
       │                                       │
       ▼                                       ▼
  [STAGE 1]                               [STAGE 2]
General-Purpose Champion             Independent Domain Model Zoo
- Baseline GBDTs                     - Fold-local screening strictly on (x_tr, y_tr)
- Produces:                          - CatBoost, XGBoost, LightGBM Extra
  * oof_stage1                       - NO Stage 1 anchor in blend/calibration
  * test_stage1                      - Produces:
                                       * oof_stage2 (pure domain ensemble)
                                       * test_stage2
       │                                       │
       │                                       │
       └───────────────────┬───────────────────┘
                           │
                     OOF Predictions
                           │
                           ▼
                       [STAGE 5]
              Grand-Teacher Meta-Stacker
       - Consumes genuine OOF predictions:
         X_meta = [oof_stage1, oof_stage2, (oof_stage3), (oof_stage4)]
       - Logit-space regularized meta-model
       - Cross-fitted calibration
       - Fixed `best_floor` runtime bug -> uses `FLOOR`
                           │
                           ▼
                 [Final Submission]
          submissions/submission_clean.csv
```

---

## 3. Pipeline Modes: Clean Evaluation vs Experimental Competition

### 3.1. Clean Evaluation Mode
- **Feature Selection**: Strictly fold-local on `(x_tr_raw, y_tr)`.
- **Target Encoding**: Strictly fold-local on `(x_tr, y_tr)` with inner CV.
- **Pseudo-Labeling**: Disabled.
- **Temperature Sharpening**: Disabled ($T=1.0$).
- **Prevalence Adjustment**: Disabled (no test mean scaling).
- **OOF Hill Climbing**: Disabled for reporting validation estimates.
- **Calibration**: Cross-fitted or omitted; zero in-sample calibration reported as OOF performance.
- **Stage 2**: Independent domain learning without Stage 1 anchor.
- **Stage 5**: Trains meta-model strictly on genuine OOF predictions.

### 3.2. Experimental Competition Mode
- Preserves existing advanced competition components:
  - MultiStrata cohort ensembling
  - Self-trained distillation students
  - Adaptive ensemble optimization / hill climbing
  - Experimental sharpening / calibration variants
- All experimental outputs are explicitly demarcated in manifests and logs.

---

## 4. Key Component Audits & Modifications

### 4.1. Stage 2 (`src/stages/stage2_gbdt_zoo.py`)
- **Modification**: Decouple Stage 2 from the Stage 1 anchor:
  - Add flag `include_anchor=False` (default for clean domain learning).
  - Blend optimization optimizes weights exclusively over Stage 2 domain architectures (`cb_d7_dom26`, `cb_d6_dom35`, `xgb_d4_dom35`, `xgb_d4_triage`, `lgb_extra`).
  - Preserves fold-local `screen_features(x_tr_raw, y_tr, ...)`.

### 4.2. Stage 5 (`src/stages/stage5_meta_stacker.py`)
- **Bug Fix**: Line 332: Replace undefined `best_floor` with `FLOOR`.
- **Validation**: Verify all inputs (`oof_stage1`, `oof_stage2`, `oof_stage3`, `oof_stage4`) are genuine out-of-fold arrays.
- **Integration**: Serve as the official meeting point where general champion and independent domain signals unite.

### 4.3. Clean Runner Module (`src/clean_pipeline.py`)
- Single unified entry point orchestrating:
  - Clean Stage 1 (or baseline anchor)
  - Clean Independent Stage 2 (no anchor)
  - Clean Stage 5 meta-stacking
  - OOF coverage check (`assert oof_filled.all()`)
  - Saving raw `.npy` arrays
  - Generating `experiment.json` manifest with `LEAKAGE CHECK: PASS`

---

## 5. Artifact & Manifest Specification (`experiment.json`)

```json
{
  "timestamp": "2026-09-29T...",
  "git_commit": "...",
  "mode": "clean_evaluation",
  "seed": 42,
  "n_splits": 10,
  "fold_local_feature_selection": true,
  "fold_local_target_encoding": true,
  "pseudo_labeling": false,
  "temperature_sharpening": false,
  "prevalence_adjustment": false,
  "oof_hill_climbing": false,
  "stage2_independent_domain_learning": true,
  "stage2_consumed_stage1_anchor": false,
  "stage5_meta_inputs": ["oof_stage1", "oof_stage2"],
  "oof_metrics": {
    "stage1_comp": 0.7352,
    "stage2_independent_comp": 0.7335,
    "stage5_final_comp": 0.7365
  },
  "leakage_audit_status": "PASS",
  "submission_path": "submissions/submission_clean.csv"
}
```
