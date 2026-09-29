# Implementation Plan: Clean Validation, Stage 5 Preservation & Auditable Submission Pipeline

**Branch**: `002-clean-submission-pipeline` | **Date**: 2026-09-29 | **Spec**: [specs/002-clean-submission-pipeline/spec.md](spec.md)

**Input**: Feature specification from `specs/002-clean-submission-pipeline/spec.md`

## Summary

This plan implements a clean, auditable, leak-free validation and submission pipeline for `Chrisolande/liquidity` while preserving the existing ~#2 leaderboard tournament architecture (Stages 1 through 5, GBDT zoo, distillation, and Stage 5 meta-stacker). 

Key deliverables:
1. Fix known runtime defect in `src/stages/stage5_meta_stacker.py` (`best_floor` reference error).
2. Enforce TabPFN foundation integrity across `pfn_phys` and `pfn_champ` views with 0% surrogate fallback.
3. Implement `src/clean_pipeline.py` with fold-local feature screening, fold-local target encoding, raw prediction exports, single-pass regularized blending, and `experiment.json` manifest generation.
4. Streamline `pipeline_reproduction.ipynb` with dual execution (Clean Path + Preserved Tournament Path).
5. Add automated tests verifying leakage invariance and determinism.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: LightGBM, CatBoost, XGBoost, PyTorch, Scikit-Learn, SciPy, Pandas, NumPy, optional TabPFN  
**Target Platform**: Linux (x86_64) / Kaggle GPU Environment  
**Execution Budget**: < 4 hours total  
**Evaluation Invariants**: `TARGET="liquidity_stress_next_30d"`, `ID_COL="ID"`, `FLOOR=0.0020`, `CEIL=0.9995`, `N_SPLITS=10`, `SEED=42`  

## Proposed Changes

### Component 1: Runtime Bug Fixes & Code Health
- **`src/stages/stage5_meta_stacker.py`**: Fix line 332 replacing undefined `best_floor` with `FLOOR`.
- **`src/models/tabpfn_model.py`**: Eliminate surrogate fallback branches; enforce genuine TabPFN execution on GPU or explicit error/skip.

### Component 2: Dedicated Clean Pipeline Module
- **`src/clean_pipeline.py`** [NEW]:
  - `seed_everything(seed=42)` for deterministic initialization.
  - Target-independent feature engineering via `engineer_features()`.
  - 10-fold Stratified CV with fold-isolated `screen_features(x_tr_raw, y_tr, ...)`.
  - Domain views: `dom2_cols`, `dom3_cols`, `dom_triage_cols`, and `selected_cols`.
  - Fold-local target encoding via `apply_fold_target_encoding()` where needed.
  - Model training: CatBoost (`cb_d7_dom26`), XGBoost (`xgb_d4_dom35`), LightGBM (`lgb_extra`), and TabPFN (`pfn_phys`, `pfn_champ`).
  - Strict assertion: `assert oof_filled.all()`.
  - Raw prediction artifact export: `artifacts/oof_*.npy`, `artifacts/test_*.npy`.
  - Bounded regularized blend ($w_i \ge 0, \sum w_i = 1$) minimizing LogLoss on OOF.
  - Final outputs: `submissions/submission_clean.csv` and `artifacts/experiment.json` with `LEAKAGE CHECK: PASS`.

### Component 3: Notebook Streamlining & Dual Execution
- **`pipeline_reproduction.ipynb`**:
  - Cell 1: Setup & Environment Check.
  - Cell 2: Clean Pipeline execution (`run_clean_pipeline()`).
  - Cell 3: Clean Submission validation.
  - Downstream Cells: Preserved Stage 1–5 tournament reproduction pipeline.

### Component 4: Test Suite
- **`tests/test_clean_pipeline.py`** [NEW]: Fast unit tests for fold-local screening invariance, OOF coverage verification, and bounded blend weights.

## Verification Plan

1. `python -m compileall src`
2. `pytest tests/test_clean_pipeline.py tests/test_leakage_invariance.py -v`
3. Smoke run: `python -c "from src.clean_pipeline import run_clean_pipeline; run_clean_pipeline(seed=42, n_splits=3, k_top_features=20, run_tabpfn=False, output_dir='artifacts/smoke', sub_dir='submissions/smoke')"`
4. Inspect `submissions/submission_clean.csv` and `artifacts/experiment.json`.
