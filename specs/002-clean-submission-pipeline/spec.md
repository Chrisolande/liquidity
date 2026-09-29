# Feature Specification: Clean Validation, Independent Domain Learning, and Stabilized Competition Submission Pipeline

**Feature Branch:** `002-clean-submission-pipeline`  
**Created:** 2026-09-29  
**Last Amended:** 2026-09-29  
**Status:** Approved  
**Objective:** Preserve the existing multi-stage architecture while making Stage 2 an independent domain-specialization stage and establishing Stage 5 as the point where Stage 1 and Stage 2 predictions are allowed to interact.

---

## Core Architectural Principle

The pipeline must preserve the distinction between:

1. **Stage 1:** General-purpose baseline/champion.
2. **Stage 2:** Independent domain-specialized model zoo.
3. **Stage 3:** Additional model family / diversity source (TabPFN).
4. **Stage 4:** Additional diversity experiments (MultiStrata, TabMLP, Distillation).
5. **Stage 5:** Meta-stacking / final combination of genuinely out-of-fold predictions.

### Critical Stage 2 Rule

**Stage 2 must NOT receive the Stage 1 champion as an input to its own model fitting, blend optimization, or calibration.**

Stage 2 is intended to answer:

> Can domain-specialized models learn complementary signal independently of the general Stage 1 solution?

Therefore, Stage 2 must learn the domain structure independently.

The Stage 1 champion and Stage 2 domain ensemble may meet later at Stage 5 through genuine OOF predictions.

---

# User Story 1: Fold-Local Feature Selection and Target Encoding

As a competition researcher, I want all target-dependent transformations to be performed within the outer training fold so that OOF predictions remain valid for evaluation and downstream stacking.

### Requirements

* Feature selection must be performed separately inside every outer CV fold.
* Target encoding must be performed separately inside every outer CV fold.
* Validation rows must never influence feature selection or target encoding for that fold.
* Test data must never influence target-dependent transformations.
* Selected training features must be applied unchanged to the corresponding validation/test rows.
* Record the selected feature count and selected feature names for every fold.

---

# User Story 2: Stage 1 General-Purpose Champion

Stage 1 may continue to produce the strongest general-purpose model/champion.

However, the clean evaluation path must distinguish between:

* raw model OOF predictions
* calibrated predictions
* adaptively optimized predictions
* final champion predictions

Any OOF hill climbing or optimization directly against the same OOF targets must be classified as experimental.

The clean path must not use unconstrained OOF hill climbing.

---

# User Story 3: Stage 2 Independent Domain Model Zoo

Stage 2 must be treated as a **domain-specialization stage**, not as another copy of Stage 1 with the Stage 1 champion included in the blend.

### Stage 2 Input

Stage 2 receives:

* training features
* training labels
* fold assignments
* domain/strata information where appropriate

Stage 2 does **not** receive:

* Stage 1 OOF predictions as a blend candidate
* Stage 1 test predictions as a blend candidate
* Stage 1 champion predictions as a model feature
* Stage 1 predictions for calibration
* Stage 1 predictions for adaptive optimization

unless a future experiment explicitly defines a separate stacking experiment.

### Stage 2 Model Zoo

The existing domain-specialized models should be preserved, including the current GBDT architectures such as:

* CatBoost domain model(s)
* XGBoost domain model(s)
* LightGBM / extra model(s)
* other existing domain-specific configurations

Do not delete existing architectures merely to simplify the clean path.

### Stage 2 Ensemble

The Stage 2 ensemble must contain only Stage 2 domain models.

For example:

```text
Stage 2 OOF candidates:

cb_d7_dom26
cb_d6_dom35
xgb_d4_dom35
xgb_d4_triage
lgb_extra
```

It must NOT contain:

```text
oof_stage1_anchor
```

as a Stage 2 blend member.

### Stage 2 Objective

The Stage 2 evaluation should determine:

1. how well domain-specialized models perform independently;
2. whether they produce prediction errors that differ from Stage 1;
3. whether their OOF predictions contain complementary information useful to Stage 5.

A Stage 2 score lower than Stage 1 does NOT automatically mean Stage 2 is useless.

The relevant downstream question is whether Stage 2 provides complementary OOF signal.

---

# User Story 4: Stage 1 and Stage 2 Meet at Stage 5

The primary meeting point between Stage 1 and Stage 2 must be **Stage 5**, through genuine OOF predictions.

Conceptually:

```text
                    ┌─────────────────┐
                    │     Stage 1     │
                    │ General Champion│
                    └────────┬────────┘
                             │
                         OOF Stage 1
                             │
                             ▼
                       ┌───────────┐
                       │           │
                       │  Stage 5  │
                       │ Meta Stack│
                       │           │
                       └───────────┘
                             ▲
                             │
                         OOF Stage 2
                             │
                    ┌────────┴────────┐
                    │     Stage 2     │
                    │ Domain Ensemble │
                    └─────────────────┘
```

Stage 5 may additionally consume OOF predictions from Stage 3 and Stage 4 where those predictions are valid and independently generated.

The meta-model input may therefore look conceptually like:

```python
X_meta = np.column_stack([
    oof_stage1,
    oof_stage2,
    oof_stage3,
    oof_stage4,
])
```

The exact implementation must follow the existing architecture rather than introducing an unnecessary rewrite.

### Critical Requirement

Every prediction supplied to Stage 5 must correspond to an OOF prediction for that training row.

Stage 5 must never train a meta-model on predictions generated from a model that was trained using that same row's target.

---

# User Story 5: Preserve Diversity Instead of Saturating Stage 2

Do not use the Stage 1 champion to make Stage 2 look stronger through blending.

The purpose of Stage 2 is to generate a different predictive representation.

Therefore:

```text
Stage 1:
general signal
        ↓
        ↓
        └──────────────┐
                       │
Stage 2:              │
domain signal         │
        ↓             │
        └──────────────┤
                       ▼
                    Stage 5
                  meta-stacker
                       │
                       ▼
                 final prediction
```

This separation should be maintained even if Stage 1 has a substantially higher standalone OOF score.

---

# User Story 6: Disable Leakage-Prone Techniques in Clean Path

The clean path must disable:

* pseudo-labeling
* test-set pseudo-label training
* temperature sharpening
* prevalence adjustment based on test predictions
* unconstrained OOF hill climbing
* in-sample calibration
* any target-dependent transformation performed globally before outer CV

Experimental implementations may remain in the repository.

They must be explicitly classified as experimental and must not silently execute as part of the clean pipeline.

---

# User Story 7: Stage 5 Audit

Audit `src/stages/stage5_meta_stacker.py`.

Specifically:

1. Fix the undefined `best_floor` reference.
2. Verify the correct `FLOOR` constant/configuration is used.
3. Verify exactly which Stage outputs are consumed.
4. Verify Stage 1 and Stage 2 predictions are genuine OOF predictions.
5. Verify Stage 3 and Stage 4 predictions are genuine OOF predictions before using them.
6. Ensure the meta-model does not train on predictions generated from models that saw the corresponding target.
7. Identify any adaptive optimization performed directly against the same OOF targets.
8. Classify such optimization as experimental unless it is properly cross-fitted.
9. Preserve Stage 5 rather than deleting it.
10. Ensure Stage 5 is the principal integration point for independent model families.

---

# User Story 8: Calibration

Calibration must not use the complete OOF prediction vector to fit a calibration model and then report the resulting score as unbiased OOF performance.

Acceptable approaches:

* cross-fitted calibration;
* calibration performed only after model selection using a separate calibration split;
* skip calibration in the clean evaluation path.

Existing experimental calibration code may remain available.

---

# User Story 9: Raw Prediction Artifacts

The clean pipeline must save:

* fold assignments
* raw Stage 1 OOF predictions
* raw Stage 1 test predictions
* raw Stage 2 OOF predictions
* raw Stage 2 test predictions
* Stage 3 OOF/test predictions when executed
* Stage 4 OOF/test predictions when executed
* final Stage 5 OOF prediction
* final test prediction
* final ensemble weights/meta-model configuration
* feature counts per fold
* random seeds
* CV configuration
* model configuration
* git commit/version information where available

This allows every stage to be independently inspected.

---

# User Story 10: Explicit Pipeline Modes

The repository should distinguish between:

### Clean Evaluation Mode

```text
Stage 1
   ↓
Stage 2 independent domain learning
   ↓
Stage 3 optional
   ↓
Stage 4 optional
   ↓
Stage 5 integration
   ↓
final submission
```

with:

```text
pseudo_labeling = false
temperature_sharpening = false
prevalence_adjustment = false
oof_hill_climbing = false
fold_local_feature_selection = true
fold_local_target_encoding = true
```

### Experimental Competition Mode

Existing advanced techniques may remain available, including:

* OOF optimization
* pseudo-labeling
* sharpening
* prevalence adjustment
* experimental calibration
* adaptive ensemble search

But these must be clearly documented as experimental and must not be represented as leakage-free validation.

---

# User Story 11: Stage 2 Execution

Stage 2 should remain available as a real experiment.

However, because Stage 2 is computationally expensive, its execution should be configurable.

For example:

```text
RUN_STAGE2=true
```

should execute the independent domain zoo.

When disabled, the pipeline must not pretend that Stage 2 predictions exist.

If Stage 5 requires Stage 2 predictions, the pipeline should either:

1. execute Stage 2; or
2. load a previously generated, version-compatible Stage 2 OOF artifact.

Do not silently substitute Stage 1 predictions for Stage 2 predictions.

---

# Functional Requirements

### FR-001
All target-dependent feature selection must be fold-local.

### FR-002
All target encoding must be fold-local.

### FR-003
Stage 1 must remain available as the general-purpose champion.

### FR-004
Stage 2 must operate independently of Stage 1 predictions.

### FR-005
Stage 2 blend optimization must not include `oof_stage1_anchor`.

### FR-006
Stage 1 and Stage 2 may interact at Stage 5 through genuine OOF predictions.

### FR-007
Stage 5 must validate that all meta-features are genuinely OOF.

### FR-008
Pseudo-labeling, sharpening, prevalence adjustment, and unconstrained OOF hill climbing must be disabled in clean mode.

### FR-009
In-sample calibration must not be used to claim clean OOF performance.

### FR-010
Fix the Stage 5 `best_floor` runtime defect.

### FR-011
The clean path must produce exactly one OOF prediction per training row.

### FR-012
Raw OOF/test predictions must be persisted.

### FR-013
The pipeline must distinguish experimental components from clean components.

### FR-014
Existing experimental architectures must not be deleted merely because they are excluded from clean mode.

### FR-015
Stage 2 must preserve its original purpose as an independent domain learner.

---

# Validation Checklist

Before submission:

* [ ] `compileall` passes.
* [ ] Clean pipeline executes without runtime errors.
* [ ] Every training row receives exactly one OOF prediction.
* [ ] Feature selection is fold-local.
* [ ] Target encoding is fold-local.
* [ ] Stage 1 does not leak validation labels.
* [ ] Stage 2 does not consume Stage 1 predictions.
* [ ] Stage 2 blend contains only Stage 2 model predictions.
* [ ] Stage 1 and Stage 2 meet at Stage 5.
* [ ] Stage 5 meta-features are genuine OOF predictions.
* [ ] `best_floor` is fixed.
* [ ] No pseudo-labeling in clean mode.
* [ ] No temperature sharpening in clean mode.
* [ ] No prevalence adjustment in clean mode.
* [ ] No unconstrained OOF hill climbing in clean mode.
* [ ] No in-sample calibration presented as clean OOF validation.
* [ ] Raw OOF/test predictions are saved.
* [ ] Pipeline manifest records all experimental flags.
* [ ] Final submission is generated successfully.
* [ ] Existing experimental functionality remains available.
* [ ] Stage 2 is documented as an independent domain-learning experiment.
