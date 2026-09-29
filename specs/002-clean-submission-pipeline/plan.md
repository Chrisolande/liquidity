# Implementation Plan: Clean Validation, Independent Domain Learning & Stage 5 Meta-Stacking

**Branch**: `002-clean-submission-pipeline` | **Date**: 2026-09-29 | **Spec**: [specs/002-clean-submission-pipeline/spec.md](spec.md)

**Input**: Feature specification from `specs/002-clean-submission-pipeline/spec.md`

## Summary

This plan implements the updated specification for `Chrisolande/liquidity` to:
1. Preserve the multi-stage tournament architecture (Stages 1 through 5).
2. Make Stage 2 an independent domain-specialized model zoo (learns domain structure independently; does NOT include Stage 1 anchor in its blend/calibration).
3. Establish Stage 5 as the primary meeting point where the general-purpose champion (Stage 1) and the independent domain ensemble (Stage 2)—plus optional Stage 3/4 diversity—meet via genuine OOF predictions.
4. Fix the Stage 5 `best_floor` runtime bug.
5. Distinguish Clean Evaluation Mode from Experimental Competition Mode.
6. Ensure full auditability and raw prediction preservation.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: LightGBM, CatBoost, XGBoost, Scikit-Learn, SciPy, Pandas, NumPy, PyTorch, optional TabPFN  
**Target Platform**: Linux (x86_64) / Kaggle GPU Environment  
**Execution Budget**: < 4 hours total  
**Evaluation Invariants**: `TARGET="liquidity_stress_next_30d"`, `ID_COL="ID"`, `FLOOR=0.0020`, `CEIL=0.9995`, `N_SPLITS=10`, `SEED=42`  

## Proposed Changes

### Component 1: Fix Stage 5 Runtime Defect & Meta-Feature Verification
- **`src/stages/stage5_meta_stacker.py`**: Fix line 332 replacing undefined `best_floor` with `FLOOR`.
- Verify inputs (`oof_stage1`, `oof_stage2`, etc.) are genuine OOF arrays.

### Component 2: Stage 2 Decoupling from Stage 1 Anchor
- **`src/stages/stage2_gbdt_zoo.py`**:
  - Add parameter `include_anchor: bool = False` (default `False` in clean mode).
  - When `False`, blend optimization optimizes exclusively over Stage 2 domain architectures (`cb_d7_dom26`, `cb_d6_dom35`, `xgb_d4_dom35`, `xgb_d4_triage`, `lgb_extra`).
  - Preserve fold-local screening via `screen_features()` strictly on outer training slices.

### Component 3: Dedicated Clean Pipeline Module
- **`src/clean_pipeline.py`** [NEW]:
  - Coordinates Clean Evaluation Mode.
  - Runs or loads clean Stage 1 general champion.
  - Runs clean independent Stage 2 (without Stage 1 anchor).
  - Runs Stage 5 meta-stacking combining Stage 1 and Stage 2 genuine OOF predictions.
  - Asserts OOF completeness (`oof_filled.all()`).
  - Saves raw arrays and `experiment.json` manifest with `LEAKAGE CHECK: PASS`.

### Component 4: Notebook Streamlining & Dual Execution
- **`pipeline_reproduction.ipynb`**:
  - Expose Clean Evaluation Mode in dedicated cells.
  - Retain preserved Stage 1–5 tournament reproduction pipeline.

### Component 5: Test Suite
- **`tests/test_clean_pipeline.py`** [NEW]:
  - Test Stage 2 independence (anchor omitted).
  - Test fold-local screening invariance.
  - Test Stage 5 execution and meta-feature alignment.

## Verification Plan

1. `python -m compileall src`
2. `pytest tests/test_clean_pipeline.py tests/test_leakage_invariance.py -v`
3. Smoke run: `python -c "from src.clean_pipeline import run_clean_pipeline; run_clean_pipeline(seed=42, n_splits=3, k_top_features=20, run_stage2=True, run_tabpfn=False, output_dir='artifacts/smoke', sub_dir='submissions/smoke')"`
4. Verify `submissions/submission_clean.csv` and `artifacts/experiment.json`.
