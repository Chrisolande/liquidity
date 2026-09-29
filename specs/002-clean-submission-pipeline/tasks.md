# Tasks: Clean Validation, Independent Domain Learning & Stabilized Competition Submission Pipeline

**Feature Directory**: `specs/002-clean-submission-pipeline`  
**Status**: Completed  
**Created**: 2026-09-29  

---

## Phase 1: Setup (Infrastructure & Test Harness)

**Purpose**: Establish test files and verification scaffolding.

- [X] T001 Create unit test harness for clean pipeline validation in `tests/test_clean_pipeline.py`
- [X] T002 [P] Verify and ensure target evaluation and baseline configurations in `src/config.py`

---

## Phase 2: Foundational (Blocking Prerequisites & Defect Fixes)

**Purpose**: Core runtime fixes and stage interfaces that MUST be complete before pipeline integration.

- [X] T003 Fix undefined `best_floor` reference error on line 332 in `src/stages/stage5_meta_stacker.py` using `FLOOR`
- [X] T004 [P] Enforce genuine TabPFN foundation priors and eliminate surrogate CatBoost fallback in `src/models/tabpfn_model.py`
- [X] T005 [P] Update `src/stages/stage2_gbdt_zoo.py` to add `include_anchor: bool = False` decoupling parameter for independent domain learning

**Checkpoint**: Core defects fixed and stage interfaces ready for independent domain learning.

---

## Phase 3: User Story 3 - Stage 2 Independent Domain Model Zoo (Priority: P1) 🎯 MVP Component

**Goal**: Enable Stage 2 to operate as a pure independent domain-specialized zoo without consuming Stage 1 predictions in its fitting, blending, or calibration.

**Independent Test**: Run `stage2_gbdt_zoo.py` with `include_anchor=False` and confirm blend candidates contain only Stage 2 architectures (`cb_d7_dom26`, `cb_d6_dom35`, `xgb_d4_dom35`, `xgb_d4_triage`, `lgb_extra`) and that `oof_champ_train.npy` is not in the blend.

- [X] T006 [US3] Implement anchor decoupling in `src/stages/stage2_gbdt_zoo.py` to omit `oof_champ_train.npy` from `blend_candidates_oof` and `blend_candidates_test` when `include_anchor=False`
- [X] T007 [US3] Verify fold-local feature screening strictly on outer training slices `(x_tr_raw, y_tr)` in `src/stages/stage2_gbdt_zoo.py`
- [X] T008 [US3] Export independent domain predictions `oof_stage2_domain.npy` and `test_stage2_domain.npy` in `src/stages/stage2_gbdt_zoo.py`
- [X] T009 [US3] Add unit test verifying Stage 2 blend candidates do not contain Stage 1 anchor in `tests/test_clean_pipeline.py`

**Checkpoint**: Stage 2 operates 100% independently of Stage 1.

---

## Phase 4: User Story 4 & 7 - Stage 5 Meta-Stacking Convergence (Priority: P1)

**Goal**: Establish Stage 5 as the primary meeting point where Stage 1 (general champion) and Stage 2 (independent domain ensemble) interact through genuine out-of-fold predictions.

**Independent Test**: Run `stage5_meta_stacker.py` and confirm it loads genuine OOF predictions from Stage 1 and Stage 2, executes logit-space regularized meta-stacking, applies cross-fitted calibration, and saves valid deliverables.

- [X] T010 [US4] Audit and verify meta-feature input loading in `src/stages/stage5_meta_stacker.py` ensuring inputs (`oof_champ_train.npy`, `gbdt_zoo_4seed.npz` or `oof_stage2_domain.npy`) are genuine OOF arrays
- [X] T011 [US4] Verify meta-model $L_2$ logit optimization and cross-fitted Platt scaling in `src/stages/stage5_meta_stacker.py`
- [X] T012 [US4] Add unit test asserting Stage 5 meta-features have identical lengths and valid probability bounds in `tests/test_clean_pipeline.py`

**Checkpoint**: Stage 5 reliably unites Stage 1 and Stage 2 predictions via genuine OOF meta-stacking.

---

## Phase 5: User Story 1, 2 & 6 - Clean Pipeline Runner Orchestration (Priority: P1)

**Goal**: Provide a standalone, leak-free runner module `src/clean_pipeline.py` that executes clean 10-fold CV, enforces in-model determinism, disables pseudo-labeling/sharpening/prevalence hacking, and asserts 100% OOF coverage.

**Independent Test**: Execute `python -m src.clean_pipeline --seed 42 --n_splits 10` and verify it runs end-to-end without leakage, asserts `oof_filled.all()`, and generates `submissions/submission_clean.csv`.

- [X] T013 [US1] Implement `seed_everything(seed=42)` and deterministic framework controls in `src/clean_pipeline.py`
- [X] T014 [US1] Implement 10-fold Stratified CV loop with fold-local `screen_features(x_tr_raw, y_tr, ...)` and fold-local target encoding in `src/clean_pipeline.py`
- [X] T015 [US2] Implement bounded regularized linear blending ($w_i \ge 0, \sum w_i = 1$) minimizing LogLoss on OOF without iterative hill climbing in `src/clean_pipeline.py`
- [X] T016 [US6] Ensure test pseudo-labeling, temperature sharpening ($T=1.0$), and test prevalence adjustment multipliers are strictly disabled in `src/clean_pipeline.py`
- [X] T017 [US1] Implement OOF coverage assertion (`assert oof_filled.all()` and `len(oof) == len(y_true)`) in `src/clean_pipeline.py`

**Checkpoint**: Clean pipeline module executes leak-free 10-fold CV with full verification gates.

---

## Phase 6: User Story 9 & 10 - Artifacts, Audit Manifest & Run Diagnostics (Priority: P2)

**Goal**: Save raw prediction arrays, generate machine-readable `experiment.json` manifest, and format clean submission deliverables.

**Independent Test**: Confirm existence and schema validity of `artifacts/experiment.json`, raw `.npy` arrays, and `submissions/submission_clean.csv`.

- [X] T018 [US9] Implement raw prediction persistence (`artifacts/oof_*.npy`, `artifacts/test_*.npy`) for all models before blending in `src/clean_pipeline.py`
- [X] T019 [US9] Implement `experiment.json` manifest writer recording Git commit, seed, fold count, feature counts per fold, experimental flags, and `LEAKAGE CHECK: PASS` in `src/clean_pipeline.py`
- [X] T020 [US10] Implement submission exporter writing `submissions/submission_clean.csv` formatted with `[ID, Target]` and bounded in $[0.0020, 0.9995]$ in `src/clean_pipeline.py`

**Checkpoint**: Complete auditable artifacts and manifest generated upon pipeline completion.

---

## Phase 7: Polish, Verification & Notebook Integration

**Purpose**: End-to-end static checks, automated test suite execution, and dual-mode notebook integration.

- [X] T021 [P] Run static compilation check `python -m compileall src` ensuring zero syntax/import errors
- [X] T022 [P] Execute automated test suite `pytest tests/test_clean_pipeline.py tests/test_leakage_invariance.py -v`
- [X] T023 Update `pipeline_reproduction.ipynb` with clean execution cells (Section A), single-seed/multi-seed toggle, while preserving historical Stage 1–5 reproduction cells (Section B)
- [X] T024 Execute smoke test run of `src/clean_pipeline.py` (3 splits, GBDT only) to verify submission and manifest creation

---

## Dependencies & Execution Order

```text
Phase 1 (Setup)
   │
   ▼
Phase 2 (Foundational Defect Fixes & Interfaces)
   │
   ├──────────────────────────────┐
   ▼                              ▼
Phase 3 (Stage 2 Decoupling)   Phase 4 (Stage 5 Integration)
   │                              │
   └──────────────┬───────────────┘
                  │
                  ▼
Phase 5 (Clean Pipeline Runner Orchestration)
                  │
                  ▼
Phase 6 (Artifacts & Manifest Export)
                  │
                  ▼
Phase 7 (Polish, Verification & Notebook)
```
