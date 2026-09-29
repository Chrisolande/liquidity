# Implementation Plan: Leak-Free Cross-Validation & Calibration Pipeline

**Branch**: `001-fix-cv-leakage` | **Date**: 2026-09-29 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-fix-cv-leakage/spec.md`

## Summary

Eliminate validation target leakage, in-sample calibration evaluation, greedy OOF hill-climbing, test pseudo-label sharpening, and arbitrary test prevalence scaling across Stages 2–5. The technical approach inverts the Stage 2 cross-validation loop to run `screen_features()` strictly inside each outer fold, resolves categorical metadata explicitly per fold, delegates `blend_and_calibrate()` to cross-fitted `platt_scaling_calibrate()`, replaces Stage 5 greedy hill-climbing with regularized $L_2$ logit stacking, removes multiplicative prevalence scaling, and establishes an immutable 20% shadow holdout benchmark.

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: LightGBM, CatBoost, XGBoost, Scikit-learn, SciPy, Pandas, NumPy  
**Storage**: Parquet / NumPy NPZ (`checkpoints/`, `submissions/`)  
**Testing**: pytest  
**Target Platform**: Linux server / Kaggle / local environment  
**Project Type**: Machine Learning Cross-Validation & Ensembling Pipeline  
**Performance Goals**: 10-fold nested feature screening across all folds in <30s total; zero target leakage across folds  
**Constraints**: Zero changes to the competition metric definition ($0.40 \cdot \text{AUC} + 0.60 \cdot (1 - \text{LogLoss} / 0.595)$); all probabilities strictly bounded in $[0.0020, 0.9980]$  
**Scale/Scope**: 40,000 training observations, 30,000 test observations, 200+ engineered features, 5 GBDT architectures, 5 pipeline stages  

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- All changes adhere to leak-free evaluation principles: no validation sample or label may be observed during feature selection, model fitting, or probability calibration.
- Metric integrity: `src/metrics.py` competition metric formula is preserved without alteration.

## Project Structure

### Documentation (this feature)

```text
specs/001-fix-cv-leakage/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan
├── checklists/          # Requirements and quality checklists
└── tasks.md             # Execution task list
```

### Source Code (repository root)

```text
src/
├── config.py            # Global constants (ID_COL, TARGET, SEED, FLOOR, CEIL, EPS)
├── metrics.py           # Competition score and log loss / AUC evaluation
├── ensemble/
│   ├── calibration.py   # Cross-fitted Platt and Beta probability calibration
│   └── stacking.py      # Multi-model logit stacking and blend routines
├── features/
│   ├── domain.py        # Domain-specific physics and liquidity ratios
│   ├── encoding.py      # Feature screening and domain subset extraction
│   ├── monthly.py       # Longitudinal trend and summary aggregations
│   ├── pipeline.py      # Feature engineering pipeline
│   └── selection.py     # Feature engine selection routines
└── stages/
    ├── stage2_gbdt_zoo.py       # 10-fold nested GBDT zoo pipeline
    ├── stage4_diversity.py      # De-risked student distillation pipeline
    └── stage5_meta_stacker.py   # Regularized logit stacker and calibration

tests/
├── test_calibration_crossfit.py  # Platt cross-fitting unit test
├── test_leakage_invariance.py    # Target leakage invariance test
├── test_stage2_smoke.py          # Stage 2 multi-architecture smoke test
├── test_stage4_derisk.py         # Stage 4 de-risked distillation test
└── test_stage5_stacker.py        # Stage 5 stacker & prevalence test

experiments/
└── run_shadow_holdout_benchmark.py  # Immutable 20% shadow holdout benchmark
```

**Structure Decision**: The codebase maintains its existing modular layout in `src/`, with fixes localized to `src/ensemble/stacking.py`, `src/stages/stage2_gbdt_zoo.py`, `src/stages/stage4_diversity.py`, and `src/stages/stage5_meta_stacker.py`, accompanied by dedicated verification tests in `tests/` and benchmark scripts in `experiments/`.
