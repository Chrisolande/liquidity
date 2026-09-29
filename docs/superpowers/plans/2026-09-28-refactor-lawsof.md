# Refactor `lawsof (1).py` Into Modular Package (`src/`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surgically refactor the 4,300-line monolithic `lawsof (1).py` script into a clean, DRY, modular Python package located in `src/`, eliminating massive code duplication while preserving 100% of pipeline capabilities.

**Architecture:** Split by domain responsibility:
- `src/config.py`: Core competition invariants, paths, seeds, hardware detection, budget presets, feature subsets.
- `src/metrics.py`: Unified competition loss metric ($0.40 \cdot \text{AUC} + 0.60 \cdot (1 - \frac{\text{LL}}{0.595})$), paired bootstrap, segment loss tables, correlation matrices.
- `src/features/`: Modularized feature engineering (`monthly.py`, `domain.py`, `encoding.py`, `pipeline.py`).
- `src/models/`: Model trainers for LightGBM, CatBoost, XGBoost (`gbdt.py`), PyTorch TabMLP (`neural.py`), and TabPFN (`tabpfn_model.py`).
- `src/ensemble/`: Probability calibration, NNLS blending, Nelder-Mead optimizer, and Hill Climbing (`calibration.py`, `stacking.py`).
- `src/stages/`: Individual competition stages (Stages 1 through 5 and Audit).
- `src/main.py`: Unified pipeline orchestrator.

**Tech Stack:** Python 3.10+, NumPy, Pandas, Scikit-learn, Scipy, LightGBM, XGBoost, CatBoost, PyTorch, Optuna, TabPFN, Hillclimbers.

---

## Tasks

### Task 1: Package Scaffolding & Configuration Module (`src/config.py`)
- Create `src/__init__.py` and `src/config.py`.
- Extract global constants (`SEED`, `N_SPLITS`, `TARGET`, `ID_COL`, `FLOOR`, `CEIL`, `EPS`).
- Extract hardware checks (`HAS_GPU`, `GPU_NAME`, `NUM_GPUS`) and `seed_everything`.
- Extract feature constant lists (`CHAMPION_14`, `MACRO_SOLVENCY_10`, `MACRO_TRIAGE_10`).
- Extract path resolvers (`find_dataset_file`, `find_file`) and `_PRESETS`.

### Task 2: Evaluation Metrics & Diagnostics (`src/metrics.py`)
- Create `src/metrics.py`.
- Unify `competition_score`, `comp_score`, and `comp_metric_eval` into a single canonical `competition_score` implementation.
- Include `paired_bootstrap`, `print_correlation_matrix`, and `print_segment_loss_table`.

### Task 3: Modular Feature Engineering (`src/features/`)
- Create `src/features/__init__.py`.
- Create `src/features/monthly.py`: `month_columns`, `add_monthly_summary_features`, `add_cross_feature_ratios`.
- Create `src/features/domain.py`: `add_entropy_features`, `add_behavioral_shift_features`, `add_liquidity_runway_and_exhaustion_features`, `add_chris_deotte_features`, `add_longitudinal_stress_features`, `add_solvency_and_burn_collapse_features`, `add_v3_features`.
- Create `src/features/encoding.py`: `apply_fold_target_encoding`, `screen_features`, `extract_domain_feature_subsets`.
- Create `src/features/pipeline.py`: Unified `engineer_features(df, use_advanced=True, ...)`.

### Task 4: Ensembling & Calibration (`src/ensemble/`)
- Create `src/ensemble/__init__.py`.
- Create `src/ensemble/calibration.py`: Platt scaling / sigmoid calibration, isotonic regression.
- Create `src/ensemble/stacking.py`: Scipy NNLS stacker, `solve_weights`, Nelder-Mead weight optimizer, `blend_and_calibrate`, and HillClimber integration.

### Task 5: Model Architectures & Runners (`src/models/`)
- Create `src/models/__init__.py`.
- Create `src/models/gbdt.py`: Unified GBDT runner (`fit_gbdt_runner`), hyperparameter tuning functions (`tune_catboost_gpu`, `tune_lightgbm`), and stage-specific runners (`run_catboost_model`, `run_xgboost_model`, `run_lightgbm_model`, `run_pseudo_student`).
- Create `src/models/tabpfn_model.py`: TabPFN classifier wrapper, QuantileTransformer RankGauss normalization, multi-view setups.
- Create `src/models/neural.py`: PyTorch `TabMLP` model, dataset loader, training loop.

### Task 6: Modular Pipeline Stages (`src/stages/`)
- Create `src/stages/__init__.py`.
- Create `src/stages/stage1_reproduce.py`: Best metrics reproduction pipeline.
- Create `src/stages/stage2_gbdt_zoo.py`: 4-seed 10-fold domain tree zoo.
- Create `src/stages/stage3_tabpfn_priors.py`: Dual TabPFN foundation priors.
- Create `src/stages/stage4_diversity.py`: Diversity base learners, self-training distillation.
- Create `src/stages/stage5_meta_stacker.py`: Step 13 meta-stacker & hill climbing ensembling.
- Create `src/stages/audit.py`: Final competition invariant checks and submission file verification.

### Task 7: Orchestrator & Smoke Test Verification (`src/main.py`)
- Create `src/main.py` entry point with stage dispatching and CLI argument parsing.
- Run `py_compile` across all files in `src/`.
- Run modular import verification and unit test assertions to ensure zero regressions.
