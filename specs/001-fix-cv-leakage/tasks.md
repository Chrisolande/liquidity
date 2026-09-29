# Tasks: Leak-Free Cross-Validation & Calibration Pipeline

**Canonical Master Plan**: [`docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md`](../../docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md)  
**Feature Branch**: `001-fix-cv-leakage`  
**Specification**: [spec.md](spec.md)  
**Implementation Plan**: [plan.md](plan.md)  

## Phase 1: Setup & Foundational Prerequisites

**Purpose**: Test directory structure and environment readiness.

- [X] T001 Set up test harness and directory structure for unit tests and benchmarks in tests/ and experiments/
- [X] T002 [P] Verify pytest test runner configuration and dependencies in requirements.txt

---

## Phase 2: User Story 3 - Fix Stacking In-Sample Calibration (Priority: P1)

**Source**: Task 1 in `docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md`  
**Goal**: Eliminate in-sample evaluation in `blend_and_calibrate()` by delegating calibration to `platt_scaling_calibrate()` so every evaluated probability is genuinely out-of-sample.  
**Independent Test**: Assert that `oof_cal` predictions in `blend_and_calibrate()` are generated via cross-fitting and do not match a global in-sample fit.

- [X] T003 [P] [US3] Create cross-fitted calibration unit test in tests/test_calibration_crossfit.py
- [X] T004 [US3] Update blend_and_calibrate() in src/ensemble/stacking.py to delegate to platt_scaling_calibrate()

---

## Phase 3: User Stories 1 & 2 - Nested Feature Screening & Resilient Fold Caching in Stage 2 (Priority: P1) 🎯 MVP

**Source**: Task 2 in `docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md`  
**Goal**: Invert Stage 2 fold loop to execute `screen_features()` strictly on outer-training splits ($X_{tr}, y_{tr}$) with explicit per-fold metadata resolution (`active_cats` and domain views).  
**Independent Test**: Assert that permuting validation labels produces zero difference in `selected_cols`, and a 2-fold smoke test completes without `KeyError`.

- [X] T005 [P] [US1] Create target leakage invariance test in tests/test_leakage_invariance.py
- [X] T006 [P] [US2] Create Stage 2 multi-architecture smoke test in tests/test_stage2_smoke.py
- [X] T007 [US1] Remove global run_feature_engine_selection() and invert loop to Folds (outer) -> Models (inner) in src/stages/stage2_gbdt_zoo.py
- [X] T008 [US2] Implement explicit per-fold active_cats and domain column views resolution in src/stages/stage2_gbdt_zoo.py
- [X] T009 [US1] Export raw uncalibrated prediction arrays in checkpoints/gbdt_zoo_4seed.npz from src/stages/stage2_gbdt_zoo.py

---

## Phase 4: User Story 4 - De-risk Stage 4 Pseudo-Labeling & Sharpening (Priority: P2)

**Source**: Task 3 in `docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md`  
**Goal**: Remove test-set pseudo-label injection and temperature sharpening ($T=0.85$) from student training.  
**Independent Test**: Unit test confirming student training does not append test data and does not apply $T < 1.0$ probability sharpening.

- [X] T010 [P] [US4] Create Stage 4 de-risking test in tests/test_stage4_derisk.py
- [X] T011 [US4] Remove test-set pseudo-label injection and temperature sharpening (T=0.85) from src/stages/stage4_diversity.py

---

## Phase 5: User Story 4 & 5 - Harden Stage 5 Meta-Stacker & Purge Prevalence Manipulation (Priority: P2)

**Source**: Task 4 in `docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md`  
**Goal**: Remove greedy OOF hill-climbing, purge multiplicative test prevalence hacking (`0.15340`), and apply single-stage cross-fitted Platt calibration.  
**Independent Test**: Verify that test predictions preserve calibrated odds without multiplicative prevalence factor.

- [X] T012 [P] [US4] Create Stage 5 stacker and prevalence test in tests/test_stage5_stacker.py
- [X] T013 [US4] Remove greedy hill_climb_blend() and delete multiplicative prevalence scaling in src/stages/stage5_meta_stacker.py
- [X] T014 [US5] Apply single-stage cross-fitted platt_scaling_calibrate() at the output of Stage 5 in src/stages/stage5_meta_stacker.py

---

## Phase 6: User Story 5 - Shadow Holdout Verification Benchmark (Priority: P2)

**Source**: Task 5 in `docs/superpowers/plans/2026-09-29-leak-free-cv-pipeline.md`  
**Goal**: Evaluate clean leak-free pipeline against an immutable 20% shadow holdout (8,000 samples) to prove generalization advantage.  
**Independent Test**: Execute benchmark script and confirm clean out-of-sample evaluation on holdout data.

- [X] T015 [P] [US5] Implement 20% immutable shadow holdout verification benchmark in experiments/run_shadow_holdout_benchmark.py

---

## Phase 7: Polish & Regression Verification

**Purpose**: Verify all tests pass and update documentation.

- [X] T016 [P] Update pipeline documentation and run logs in README.md
- [X] T017 Run full test suite across tests/ to verify all leak-free and regression tests pass

---

## Dependencies & Execution Order

1. **Phase 1 (Setup)**: `T001`, `T002`
2. **Phase 2 (Stacking Calibration Fix)**: `T003`, `T004` (Unblocks clean calibration)
3. **Phase 3 (Stage 2 Zoo Refactor)**: `T005`–`T009` (Unblocks clean base predictions)
4. **Phase 4 (Stage 4 De-risking)**: `T010`, `T011`
5. **Phase 5 (Stage 5 Hardening)**: `T012`–`T014`
6. **Phase 6 (Shadow Benchmark)**: `T015`
7. **Phase 7 (Regression Suite)**: `T016`, `T017`
