# Feature Specification: Clean Validation, Stage 5 Preservation & Auditable Submission Pipeline

**Feature Branch**: `002-clean-submission-pipeline`

**Created**: 2026-09-29  
**Last Amended**: 2026-09-29  

**Status**: Approved / Ready for Planning  

**Input**: User description: "Revise feature specification 002-clean-submission-pipeline: The competition solution is currently around #2 on the leaderboard. Preserve Stage 1 through Stage 5 (including Stage 5 grand-teacher/meta-stacking, distillation, and diversity modeling). Make the final evaluation/submission path scientifically defensible and leakage-safe. Clearly distinguish the clean validation/submission path from experimental components (pseudo-labeling, temperature sharpening, prevalence adjustment, adaptive OOF hill climbing). Make the repository auditable and reproducible for code review. Fix runtime defects (Stage 5 best_floor). Avoid broad architectural rewrites."

---

## 1. Context & Architectural Principles

The repository contains an advanced, high-performing solution ranked approximately #2 on the competition leaderboard. The objective is **not** to simplify the model into something primitive or delete sophisticated experimental pipelines. 

Instead, the objective is to establish an **auditable, scientifically defensible, leak-free clean validation and submission path** while **fully preserving** the existing multi-stage architecture (Stage 1 through Stage 5), the GBDT zoo, distillation students, diversity models, and the Stage 5 grand-teacher / meta-stacker.

### Core Distinctions
1. **Clean Validation & Submission Path**:
   - Strictly leak-free: feature selection and target encodings are fitted solely on training splits within outer CV folds.
   - Evaluated once on genuine out-of-fold (OOF) predictions.
   - Uses either a simple regularized linear blend or an explicitly documented, cross-fitted OOF meta-model.
   - Zero test pseudo-labeling, zero temperature sharpening ($T=1.0$), zero test prevalence adjustment multipliers, zero unconstrained iterative OOF hill-climbing.
2. **Experimental Optimization Path**:
   - Advanced competition techniques (Stage 5 grand-teacher stacking, distillation students, adaptive hill climbing, temperature tuning, pseudo-labeling experiments) are preserved and documented.
   - They remain available for research reproducibility and competition benchmarking, but their adaptive nature is transparently documented and separated from the unbiased clean validation estimate.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 - Trustworthy Out-of-Fold Validation via Fold-Local Isolation (Priority: P1)

As a quantitative researcher and competition reviewer, I need all feature selection, feature screening, and target encoding in the clean pipeline to occur strictly within each outer training fold, so that out-of-fold metrics represent an unbiased estimate of generalization.

**Why this priority**: Global feature selection or target encoding leaks validation and test target information into feature subsets, creating over-optimistic cross-validation estimates and invalidating scientific claims.

**Independent Test**: Can be independently verified by checking that feature selector objects and target encoders are fitted solely on `(X_train_fold, y_train_fold)`. Holdout indices and test data are strictly transformed out-of-sample. Permuting or withholding validation labels must not change the selected feature set.

**Acceptance Scenarios**:
1. **Given** a partitioned training dataset in an outer CV fold, **When** feature screening executes, **Then** feature ranking and selection criteria are evaluated solely on the fold's training slice without observing validation or test data.
2. **Given** categorical columns requiring target encoding, **When** encoding transformations are computed, **Then** the encoder is fitted solely on the training fold target (using inner CV on train fold to prevent self-target leakage) and transformed onto validation and test splits.
3. **Given** outer CV completion, **When** checking OOF predictions, **Then** every single training observation receives exactly one valid OOF prediction produced by a model that never trained on that observation.

---

### User Story 2 - Separation of Clean Evaluation from Experimental Optimization (Priority: P1)

As a competition practitioner preparing for final submission, I need the clean submission path to be explicitly separated from experimental optimization mechanisms (pseudo-labeling, temperature sharpening, prevalence hacking, and unconstrained OOF hill climbing), while preserving those experimental components in the repository.

**Why this priority**: Optimizing weights across thousands of iterations directly against the same OOF targets that are reported as validation scores creates severe adaptive overfitting. Artificial probability distortions (temperature sharpening, forced prevalence multipliers) violate proper calibration and carry severe LogLoss penalties.

**Independent Test**: Can be verified by running the clean path end-to-end and checking that final probabilities and metrics are generated without temperature scaling exponents, without prevalence adjustment factors, and without test pseudo-labels appended to training data.

**Acceptance Scenarios**:
1. **Given** the clean execution path, **When** generating validation and test predictions, **Then** test-set pseudo-labels and distillation students trained on pseudo-labels are disabled.
2. **Given** raw model probability outputs in the clean path, **When** preparing final probabilities, **Then** temperature sharpening ($T=1.0$) and test prevalence adjustment multipliers are disabled.
3. **Given** multi-model OOF predictions in the clean path, **When** ensembling, **Then** weights are determined via a bounded regularized blend or cross-fitted meta-model without multi-round iterative hill climbing against validation targets.
4. **Given** existing experimental modules (in Stage 1, Stage 4, Stage 5, and distillation), **When** inspecting the codebase, **Then** these components remain intact, documented, and callable for experimental comparison.

---

### User Story 3 - Stage 5 Grand Teacher / Meta-Stacker Audit & Defect Fix (Priority: P1)

As an ML engineer reviewing the full tournament architecture, I need Stage 5 (`src/stages/stage5_meta_stacker.py`) audited, its known runtime bug (`best_floor`) resolved, and its meta-stacking pipeline verified, so that it remains fully functional and scientifically defensible.

**Why this priority**: Stage 5 represents the tournament champion meta-stacker combining diverse model streams (GBDT zoo, TabPFN, multi-strata, and distillation). Crashing due to an undefined variable prevents execution and reproducibility.

**Independent Test**: Can be verified by running `stage5_meta_stacker.py` and confirming it executes without undefined variable errors, correctly consumes genuine OOF prediction inputs, and produces valid deliverables.

**Acceptance Scenarios**:
1. **Given** the final logging and output block in `stage5_meta_stacker.py`, **When** the script completes optimization, **Then** all variables in print statements (specifically replacing `best_floor` with the module constant `FLOOR`) are fully defined in scope.
2. **Given** the meta-stacker training loop, **When** meta-models are trained, **Then** input features are genuine OOF prediction arrays and holdout validation slices are strictly respected.
3. **Given** Stage 5 execution, **When** reporting scores, **Then** any adaptive optimization against OOF targets is explicitly labeled as experimental.

---

### User Story 4 - Auditability, Manifest Generation & Prediction Preservation (Priority: P2)

As a code reviewer auditing competition submissions, I need all raw model predictions preserved to disk and an experiment manifest generated, so that pipeline integrity and model diversity can be verified independently without retraining.

**Why this priority**: Transparency and reproducibility require verifiable artifacts that document exactly what flags were active, which models were run, and how predictions were formed.

**Independent Test**: Can be verified by inspecting the output directory after execution to verify the existence of `experiment.json` and raw prediction arrays (`.npy`) for all participating models.

**Acceptance Scenarios**:
1. **Given** individual model runs (LightGBM, CatBoost, XGBoost, and optional TabPFN), **When** cross-validation completes, **Then** unblended raw out-of-fold and test predictions are exported to disk.
2. **Given** pipeline completion, **When** the run manifest `experiment.json` is generated, **Then** it records Git commit hash, random seed, fold count, feature counts per fold, model configurations, and explicit boolean flags for `pseudo_labeling: false`, `temperature_sharpening: false`, `prevalence_adjustment: false`, `oof_hill_climbing: false`, `fold_local_feature_selection: true`, and `fold_local_target_encoding: true`.
3. **Given** final reporting, **When** output is printed, **Then** an explicit `LEAKAGE CHECK: PASS` banner is displayed with supporting criteria.

---

### User Story 5 - In-Model Determinism and Runtime Safety (Priority: P2)

As a pipeline operator executing on Kaggle under a tight deadline, I need strict in-model determinism and zero runtime errors across the codebase, ensuring full reproducibility and completion within practical time limits.

**Why this priority**: Non-deterministic seeds or silent import/syntax errors cause failed submission runs or irreproducible scores during final tournament evaluation.

**Independent Test**: Can be verified by running `python -m compileall src` (must pass with 0 errors) and verifying identical OOF predictions across runs with the same seed.

**Acceptance Scenarios**:
1. **Given** the codebase, **When** running static compilation across `src/`, **Then** all modules compile with zero syntax or import errors.
2. **Given** identical input data and seed, **When** the pipeline runs, **Then** fold partitions, selected features, and prediction probabilities match deterministically ($\Delta \le 10^{-6}$).
3. **Given** the clean pipeline, **When** executed with 10 folds on single-seed (42), **Then** execution completes within the remaining time budget on standard Kaggle GPU environments.

---

### User Story 6 - Foundation TabPFN Integrity & Elimination of Surrogate Fallback (Priority: P2)

As a researcher leveraging foundation models, I need TabPFN (when active) to run genuine foundation TabPFN priors across its two canonical views (`pfn_phys` and `pfn_champ`) with zero silent fallback to decision tree surrogates.

**Why this priority**: Silent fallbacks (e.g. falling back to CatBoost when TabPFN or GPU is unavailable) mask environment failures and misrepresent model diversity in an ensemble.

**Independent Test**: Can be verified by inspecting the TabPFN execution path to confirm that surrogate fallback code (such as CatBoost surrogates) is eliminated, and that TabPFN executes genuine foundation priors when enabled.

**Acceptance Scenarios**:
1. **When** TabPFN is enabled, **Then** it trains on both `pfn_phys` (14 physics features) and `pfn_champ` (20 features: physics + digital/bank features) using `ignore_pretraining_limits=True`.
2. **When** TabPFN or GPU is unavailable, **Then** the pipeline cleanly handles it via an explicit configuration flag or error, never silently running CatBoost as a disguise for TabPFN.

---

### Edge Cases

- **Missing Test Set Labels**: Test labels are never available and must never be accessed or estimated via heuristic prevalence assumptions.
- **Constant Features within a Fold**: If a feature becomes zero-variance within an outer training slice, the fold-local feature screener must drop it gracefully without column misalignment.
- **TabPFN GPU Memory Constraints**: TabPFN inference on 30k test records must execute in batches (e.g., `batch_size=5000`) to prevent CUDA out-of-memory errors.
- **OOF Incomplete Coverage**: If any sample is omitted or duplicated, the assertion `assert oof_filled.all()` must trigger an immediate fatal failure before any downstream processing.

---

## 3. Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Feature selection MUST be strictly fold-local in the clean path, fitted exclusively on $(X_{\text{train\_fold}}, y_{\text{train\_fold}})$.
- **FR-002**: Target encoding MUST be strictly fold-local in the clean path, fitted exclusively on the current training fold target and transformed out-of-sample.
- **FR-003**: Validation labels MUST NOT influence feature selection, encoding, model fitting, calibration, or validation prediction generation.
- **FR-004**: Test labels MUST never be used under any circumstances.
- **FR-005**: Test-set pseudo-labeling and pseudo-label student models MUST be disabled in the clean path.
- **FR-006**: Arbitrary temperature sharpening (e.g., $T=0.85$) MUST be disabled in the clean path ($T=1.0$).
- **FR-007**: Test prevalence adjustment multipliers and target-mean forcing MUST be disabled in the clean path.
- **FR-008**: Unconstrained iterative out-of-fold hill climbing MUST NOT define the clean validation estimate.
- **FR-009**: Existing experimental optimization components (in Stages 1, 4, 5, distillation, etc.) MAY remain available outside the clean path and must not be deleted.
- **FR-010**: Stage 5 grand-teacher / meta-stacking architecture MUST be preserved and audited rather than deleted.
- **FR-011**: Stage 5's undefined variable runtime bug (`best_floor`) MUST be fixed using the module's configured `FLOOR` constant.
- **FR-012**: Every training observation MUST receive exactly one out-of-fold prediction across all cross-validation folds.
- **FR-013**: Raw model predictions (OOF and test) MUST be persisted to disk before downstream blending.
- **FR-014**: A reproducibility run manifest (`experiment.json`) MUST be generated containing Git commit hash, seed, fold count, feature counts, and component enablement flags.
- **FR-015**: The clean path MUST have one clearly documented, dedicated execution entry point callable directly from `pipeline_reproduction.ipynb` or CLI.
- **FR-016**: The repository documentation MUST clearly distinguish experimental competition optimization from leakage-safe validation.
- **FR-017**: When TabPFN is active, it MUST execute genuine foundation TabPFN priors across `pfn_phys` and `pfn_champ` views, and surrogate fallbacks (such as CatBoost surrogates) MUST be strictly eliminated.
- **FR-018**: The clean pipeline MUST enforce in-model determinism (global seeding, LightGBM `deterministic=True`, CatBoost `random_seed`, XGBoost `random_state`).

### Key Entities

- **Clean Pipeline**: The leak-free, auditable execution path producing defensible CV estimates and verified competition submissions.
- **Experimental Pipeline**: Advanced tournament optimization components (distillation, Stage 5 meta-stacking, adaptive ensembles) preserved for research and high-ranking competition benchmarking.
- **Run Manifest**: JSON document recording environment parameters, model choices, feature dimensions, and explicit leakage audit verification.
- **Raw Prediction Artifacts**: Pre-blend probability vectors (`.npy`) preserving individual model outputs for independent inspection.

---

## 4. Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of feature screening and target encoding in the clean pipeline is performed within outer CV folds without seeing validation labels.
- **SC-002**: 100% of training samples have exactly one validated out-of-fold prediction verified by boolean mask assertion.
- **SC-003**: The clean pipeline executes successfully end-to-end within practical runtime limits on Kaggle GPU.
- **SC-004**: Zero syntax or import errors when running `python -m compileall src`.
- **SC-005**: Stage 5 executes without crashing from undefined variables or missing references.
- **SC-006**: A valid, complete `submission.csv` is generated with exactly 30,000 test predictions bounded in $[0.0020, 0.9995]$.
- **SC-007**: 100% of individual model raw out-of-fold and test predictions are exported prior to ensembling.
- **SC-008**: An explicit `LEAKAGE CHECK: PASS` status is documented with verification criteria.
- **SC-009**: 100% preservation of Stages 1–5, GBDT zoo, distillation, and meta-stacking architectures in the codebase.
- **SC-010**: In-model determinism produces bitwise or near-bitwise identical predictions ($\Delta \le 10^{-6}$) across identical runs with seed 42.

---

## 5. Assumptions

- **Tournament Position**: The current codebase ranks ~#2 on the leaderboard. Changes must be surgical and non-destructive.
- **Dual Pipeline Roles**: The clean pipeline provides the scientifically defensible baseline; Stage 5 provides the grand-teacher meta-stacking tournament submission. Both must be executable and auditable.
- **Calibration Stance**: Post-hoc calibration in the clean path is either cross-fitted or omitted to ensure holdout evaluation integrity. In-sample calibration on OOF predictions is prohibited.
- **TabPFN Execution**: TabPFN operates on GPU over `pfn_phys` and `pfn_champ`. If TabPFN or GPU is omitted, it is handled via a transparent configuration toggle, never masked by a fake surrogate model.
