# Tasks: Leak-Free Cross-Validation & Calibration Pipeline

**Feature Branch**: `001-fix-cv-leakage`  
**Specification**: [spec.md](spec.md)  
**Implementation Plan**: [plan.md](plan.md)  

## Phase 1: Setup (Shared Infrastructure & Test Harness)

**Purpose**: Establish test directories, fixtures, and execution dependencies.

- [ ] T001 Set up test harness and directory structure for unit tests and benchmarks in tests/ and experiments/
- [ ] T002 [P] Verify pytest test runner configuration and dependencies in requirements.txt

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core fixtures and mock data generation needed across all tests.

- [ ] T003 Create shared synthetic test data fixture utility for leak-free pipeline verification in tests/conftest.py

---

## Phase 3: User Story 1 - Fold-Isolated Supervised Feature Selection (Priority: P1) 🎯 MVP

**Goal**: Move `screen_features()` strictly inside each outer CV fold so feature screening only observes training split samples ($X_{tr}, y_{tr}$), eliminating validation target leakage.

**Independent Test**: Assert that permuting or inverting validation fold labels ($y_{va}$) produces zero change in selected columns or fold models.

### Tests for User Story 1
- [ ] T004 [P] [US1] Create target leakage invariance test in tests/test_leakage_invariance.py

### Implementation for User Story 1
- [ ] T005 [US1] Refactor src/stages/stage2_gbdt_zoo.py to remove global run_feature_engine_selection() and nest screen_features() strictly inside each outer CV fold

**Checkpoint**: Feature screening is 100% fold-isolated; validation folds remain completely unobserved during selection.

---

## Phase 4: User Story 2 - Resilient Fold Data Caching and Metadata Retention (Priority: P1)

**Goal**: Invert the Stage 2 execution loop to `Folds (outer) -> Models (inner)` to resolve `active_cat_cols` and domain views explicitly per fold, eliminating cache miss/hit bugs and memory leaks.

**Independent Test**: Execute a multi-architecture smoke test across CatBoost, XGBoost, and LightGBM confirming zero `KeyError` or unbound variable exceptions.

### Tests for User Story 2
- [ ] T006 [P] [US2] Create Stage 2 multi-architecture smoke test in tests/test_stage2_smoke.py

### Implementation for User Story 2
- [ ] T007 [US2] Invert fold loop in src/stages/stage2_gbdt_zoo.py to Folds (outer) -> Models (inner) with explicit per-fold active_cats and domain column views resolution

**Checkpoint**: Stage 2 executes sequentially across all 5 architectures on each fold with zero metadata corruption or missing columns.

---

## Phase 5: User Story 3 - Strictly Cross-Fitted Probability Calibration (Priority: P1)

**Goal**: Eliminate in-sample evaluation in `blend_and_calibrate()` by delegating calibration to `platt_scaling_calibrate()` so every evaluated probability is genuinely out-of-sample.

**Independent Test**: Assert that `oof_cal` predictions in `blend_and_calibrate()` are generated via cross-fitting and do not match an in-sample fit.

### Tests for User Story 3
- [ ] T008 [P] [US3] Create cross-fitted calibration unit test in tests/test_calibration_crossfit.py

### Implementation for User Story 3
- [ ] T009 [US3] Update blend_and_calibrate() in src/ensemble/stacking.py to delegate to platt_scaling_calibrate()

**Checkpoint**: Out-of-fold calibrated probabilities are evaluated strictly out-of-sample with zero in-sample calibration leakage.

---

## Phase 6: User Story 4 - Unbiased Ensemble Model Selection and Optimization (Priority: P2)

**Goal**: Prevent ensembling layers from overfitting validation noise by removing greedy OOF hill-climbing, eliminating test pseudo-label temperature sharpening in Stage 4, and purging arbitrary test prevalence manipulation in Stage 5.

**Independent Test**: Verify that Stage 5 executes regularized logit stacking without hill-climbing and preserves calibrated odds without multiplicative prevalence scaling.

### Tests for User Story 4
- [ ] T010 [P] [US4] Create Stage 4 de-risking test in tests/test_stage4_derisk.py
- [ ] T012 [P] [US4] Create Stage 5 stacker and prevalence test in tests/test_stage5_stacker.py

### Implementation for User Story 4
- [ ] T011 [US4] Remove test-set pseudo-label injection and temperature sharpening (T=0.85) from src/stages/stage4_diversity.py
- [ ] T013 [US4] Remove greedy hill_climb_blend() and purge multiplicative test prevalence hacking (0.15340) in src/stages/stage5_meta_stacker.py

**Checkpoint**: Stage 4 and Stage 5 ensembling are regularized, leak-free, and protected against adaptive overfitting.

---

## Phase 7: User Story 5 - Streamlined Single-Stage Calibration Hierarchy & Verification (Priority: P2)

**Goal**: Export raw uncalibrated probabilities from Stage 2 for downstream stacking, apply single cross-fitted calibration at the end of Stage 5, and verify generalization on an untouched 20% shadow holdout.

**Independent Test**: Execute `experiments/run_shadow_holdout_benchmark.py` to compare clean pipeline against overfitted pipeline on 8,000 untouched holdout samples.

### Implementation for User Story 5
- [ ] T014 [US5] Update Stage 2 export in src/stages/stage2_gbdt_zoo.py to persist raw uncalibrated prediction arrays in checkpoints/gbdt_zoo_4seed.npz
- [ ] T015 [US5] Apply single-stage cross-fitted platt_scaling_calibrate() at the output of Stage 5 in src/stages/stage5_meta_stacker.py
- [ ] T016 [P] [US5] Implement 20% immutable shadow holdout verification benchmark in experiments/run_shadow_holdout_benchmark.py

**Checkpoint**: Clean pipeline generalization is empirically proven on the untouched shadow holdout.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation updates and full test suite verification.

- [ ] T017 [P] Update pipeline documentation and run logs in README.md
- [ ] T018 Run full test suite across tests/ to verify all leak-free and regression tests pass

---

## Dependencies & Execution Order

### Phase Dependencies
- **Phase 1 (Setup)**: Can start immediately.
- **Phase 2 (Foundational)**: Depends on Phase 1 completion.
- **Phase 3 (User Story 1 - P1)**: Depends on Phase 2. Delivers MVP leak-free feature screening.
- **Phase 4 (User Story 2 - P1)**: Depends on Phase 3. Resolves fold loop and cache handling.
- **Phase 5 (User Story 3 - P1)**: Depends on Phase 2. Can execute in parallel with Phase 3/4.
- **Phase 6 (User Story 4 - P2)**: Depends on Phases 3, 4, 5. De-risks Stage 4 and Stage 5 stacking.
- **Phase 7 (User Story 5 - P2)**: Depends on Phase 6. Establishes single-stage calibration and shadow holdout benchmark.
- **Phase 8 (Polish)**: Depends on all prior phases.

### Parallel Opportunities
- T004, T006, T008, T010, T012 (all test creation tasks) can be written in parallel.
- Phase 5 (Stacking calibration fix) can be implemented in parallel with Phase 3 & 4 (Stage 2 GBDT zoo refactor).
- T016 (Shadow holdout benchmark script) can be created in parallel with Stage 5 refactoring.

---

## Implementation Strategy

### MVP First (Phases 1, 2, 3)
1. Complete Setup and Foundational fixtures (T001–T003).
2. Complete User Story 1 (T004–T005): Implement fold-isolated feature screening.
3. Validate User Story 1 independently with `tests/test_leakage_invariance.py`.

### Incremental Delivery
1. Add User Story 2 (T006–T007): Invert loop and fix fold metadata. Run multi-architecture smoke test.
2. Add User Story 3 (T008–T009): Fix calibration in `stacking.py`. Run cross-fitting test.
3. Add User Story 4 (T010–T013): Remove pseudo-label sharpening and prevalence hacking.
4. Add User Story 5 (T014–T016): Save raw zoo predictions, finalize single-stage calibration, and run shadow holdout benchmark.
