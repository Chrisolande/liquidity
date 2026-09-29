# AI4EAC Liquidity Stress Prediction Pipeline

End-to-end machine learning pipeline for predicting customer 30-day liquidity stress risk (`liquidity_stress_next_30d`) from multi-month mobile money transaction histories, agent banking behavior, and account dynamics.

---

## Table of Contents
1. [Overview & Metric](#overview--metric)
2. [Directory Structure](#directory-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Two Execution Paths](#two-execution-paths)
   - [Path A: Clean Leak-Free Pipeline (Recommended for Final Submission)](#path-a-clean-leak-free-pipeline-recommended)
   - [Path B: Tournament Multi-Stage Pipeline (Stages 1-5)](#path-b-tournament-multi-stage-pipeline-stages-1-5)
6. [Kaggle & Jupyter Notebook Guide (`pipeline_reproduction.ipynb`)](#kaggle--jupyter-notebook-guide-pipeline_reproductionipynb)
   - [Interactive Toggles](#execution-mode-toggles)
   - [Section A vs. Section B](#section-a-vs-section-b)
7. [Why Clean Validation Differs From Old Baseline](#why-clean-validation-differs-from-old-baseline)
8. [Audit & Verification](#audit--verification)
9. [Remote Kaggle Execution](#remote-kaggle-execution)
10. [Hardware & Runtime Estimates](#hardware--runtime-estimates)

---

## Overview & Metric

The objective is to produce calibrated probabilities minimizing cross-entropy while maximizing tail discrimination based on the official competition metric:

$$\text{Composite Score} = 0.40 \times \text{ROC-AUC} + 0.60 \times \left(1.0 - \frac{\text{LogLoss}}{0.595}\right)$$

All outputs are automatically calibrated and bounded within $[\text{FLOOR}=0.0020, \text{CEIL}=0.9995]$.

---

## Directory Structure

```text
├── documentation.md          # Technical architecture & methodology documentation
├── requirements.txt          # Python dependencies
├── src/
│   ├── main.py               # Orchestrator CLI entrypoint
│   ├── config.py             # Global constants, hyperparameter presets, GPU detection
│   ├── metrics.py            # Competition metric calculation & evaluation utilities
│   ├── features/             # Feature engineering pipeline
│   │   ├── pipeline.py       # Full feature extraction (rolling, slope, CV, trends)
│   │   ├── monthly.py        # Monthly aggregation & chronological transforms
│   │   ├── domain.py         # Solvency, runway, & liquidity shock features
│   │   └── encoding.py       # Leak-free CV target encoding & category mappings
│   ├── models/               # Model implementations
│   │   ├── gbdt.py           # CatBoost, XGBoost, LightGBM trainers
│   │   ├── multistrata.py    # Multi-strata iterative stratified models
│   │   ├── tabpfn_model.py   # TabPFN foundation priors
│   │   ├── neural.py         # PyTorch TabMLP neural architecture
│   │   └── distillation.py   # Temperature-sharpened knowledge distillation
│   ├── ensemble/             # Ensembling & calibration
│   │   ├── stacking.py       # Blending & Nelder-Mead / Hill-Climb optimizers
│   │   └── calibration.py    # Platt scaling & isotonic probability calibration
│   └── stages/               # Modular stage runners
│       ├── stage1_reproduce.py      # Baseline Champion models
│       ├── stage2_gbdt_zoo.py       # 4-seed 10-fold GBDT Zoo
│       ├── stage3_tabpfn_priors.py  # Foundation model priors
│       ├── stage4_diversity.py      # Distillation students & TabMLP
│       ├── stage5_meta_stacker.py   # Ensembling & final submission generation
│       └── audit.py                 # Final submission invariants check
├── kaggle_runner.py          # Script for remote execution on Kaggle GPU instances
└── README.md                 # This run guide
```

---

## Prerequisites & Installation

### Environment Requirements
- **Python**: 3.10+ (tested with 3.10–3.14)
- **CUDA GPU**: Recommended (Tesla T4, V100, A100, or RTX 3080+). Automatically falls back to multi-threaded CPU if no GPU is available.

### Setup Instructions
1. Clone or navigate to the repository directory:
   ```bash
   cd /home/olande/PycharmProjects/final
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Dataset Preparation

Place `Train.csv` and `Test.csv` in one of the auto-detected search paths:
- `./Train.csv` and `./Test.csv`
- `chrisolande/zindi-competition/Train.csv`
- `/kaggle/input/datasets/chrisolande/zindi-competition/Train.csv`

Alternatively, you can provide explicit paths using the `--train` and `--test` CLI arguments.

Expected dimensions:
- `Train.csv`: 40,000 rows $\times$ 184 columns (includes `liquidity_stress_next_30d` and `ID`)
- `Test.csv`: 30,000 rows $\times$ 183 columns (includes `ID`)

---

## Two Execution Paths

This repository provides two distinct, well-documented pipelines:

### Path A: Clean Leak-Free Pipeline (Recommended)
Designed for scientifically defensible, leak-free validation and the official final competition submission:
- **Strict Fold-Local Feature Screening**: All feature selection (mutual information) is fitted strictly inside the training fold (36,000 rows). The 4,000 validation rows are never touched during screening.
- **Strict Fold-Local Target Encoding**: Encoders are trained exclusively on outer fold training partitions.
- **Zero Test Manipulation**: Zero pseudo-labeling, zero temperature sharpening ($T=1.0$), zero target-mean prevalence forcing.
- **Regularized Bounded Blending**: Blends domain GBDTs using L2-regularized bounded optimization rather than greedy OOF hill-climbing.
- **Audited Manifest**: Exports an audit manifest with 100% OOF validation assertions.

**Run via CLI:**
```bash
# Fast single-seed run (~8-10 minutes on GPU)
python3 -m src.clean_pipeline --seed 42 --n_splits 10

# Multi-seed tournament ensemble (seeds 42 and 2026)
python3 -m src.clean_pipeline --seeds 42 2026 --n_splits 10
```

**Deliverables:**
- Clean submission: `submissions/submission_clean.csv` (and copied to `submissions/submission.csv`)
- Experiment audit manifest: `artifacts/experiment.json` (includes `"leakage_audit_status": "PASS"`)
- Raw OOF arrays: `artifacts/oof_clean.npy` and `artifacts/test_clean.npy`

---

### Path B: Tournament Multi-Stage Pipeline (Stages 1-5)
Preserves the complete experimental tournament architecture:
- **Stage 1 (Baseline Anchor & Hill-Climbing)**: CatBoost + XGBoost + HistGB baseline anchor followed by metric-direct stepwise hill climbing.
- **Stage 2 (Independent Domain GBDT Zoo)**: Independent domain learners (`cb_d7_dom26`, `cb_d6_dom35`, `xgb_d4_dom35`, `xgb_d4_triage`, `lgb_extra`) trained with fold-local screening and decoupled from the Stage 1 anchor.
- **Stage 3 (TabPFN Foundation Priors)**: Genuine foundation TabPFN execution on physical and champion feature views with execution toggle.
- **Stage 4 (Diversity & Distillation)**: MultiStrata 5-fold iterative stratification, 10-fold teacher-student distillation, and PyTorch TabMLP.
- **Stage 5 (Grand-Teacher Meta-Stacker)**: The primary meeting point where Stage 1 anchor, Stage 2 domain zoo, Stage 3, and Stage 4 OOF predictions interact through L2 regularized logit-space meta-stacking and cross-fitted Platt calibration.

**Run via CLI:**
```bash
# Fast run (Stages 1, 2, 4, 5 — skips slow 2h TabPFN):
python3 src/main.py --stage all

# Full run including Stage 3 TabPFN (~2.5 hours):
python3 src/main.py --stage all --enable-stage3

# Run individual stages:
python3 src/main.py --stage 1
python3 src/main.py --stage 2
python3 src/main.py --stage 3  # TabPFN
python3 src/main.py --stage 4
python3 src/main.py --stage 5
```

---

## Kaggle & Jupyter Notebook Guide (`pipeline_reproduction.ipynb`)

The primary reproduction notebook is [`pipeline_reproduction.ipynb`](pipeline_reproduction.ipynb).

### Execution Mode Toggles
At the top of the notebook (Cell 1), two interactive toggles control execution:
```python
# 1. Single-Seed Mode:
# Set to True for rapid iteration (seed 42 only, ~8-10 min).
# Set to False for full tournament multi-seed averaging (seeds 42, 2026).
SINGLE_SEED_MODE = True

# 2. Stage 3 (TabPFN) Toggle:
# Set to False to skip TabPFN foundation training and iterate rapidly.
# Set to True when ready to run full GPU TabPFN training (~2h).
ENABLE_STAGE3 = False
```

### Section A vs. Section B

| Notebook Section | Purpose | Runtime | When to Run | Output |
| :--- | :--- | :---: | :--- | :--- |
| **Section A: Clean Leak-Free Pipeline** | Complete self-contained 10-fold CV, fold-local screening, regularized blending, live metrics | ~8-10 min | **For the final, clean competition submission** | `submissions/submission_clean.csv`<br>`submissions/submission.csv` |
| **Section B: Tournament Reproduction** | Stages 1 through 7 (Baseline, GBDT Zoo, TabPFN, Distillation, Meta-Stacker, Audit) | ~12 min (with TabPFN skipped) or ~2.5h | To reproduce the experimental multi-stage tournament models | `submissions/submission_stage5_final.csv` |

> **Key Takeaway**: You do **not** need to run Section B if you only want the clean submission. Running the **Setup Cell** + **Section A** generates the complete, verified, leak-free submission and experiment manifest.

---

## Why Clean Validation Differs From Old Baseline

You may observe that the Clean Pipeline reports a composite OOF score of **~0.732–0.733** (single-seed) or **~0.735–0.736** (multi-seed), compared to the old baseline's single-seed ~0.735:

1. **Honest Fold-Local Screening vs. Global 40k Screening (~+0.002 bias eliminated)**:
   In the old baseline, feature screening was performed on all 40,000 rows globally before splitting folds. The validation fold targets leaked into the feature selection. The clean pipeline screens features exclusively using the current fold's training slice.
2. **Regularized Log-Loss vs. Direct OOF Hill-Climbing (~+0.002 bias eliminated)**:
   The old baseline ran greedy SLSQP directly against `-comp` on the same 40k OOF predictions it reported. The clean pipeline uses L2-regularized bounded blending, avoiding blend-weight overfitting.
3. **Single-Seed vs. Multi-Seed Averaging (~+0.003 difference)**:
   Averaging tree predictions across seeds `(42, 2026)` reduces variance and boosts score by ~0.003–0.004. Running with `SINGLE_SEED_MODE = False` applies this boost cleanly without any data leakage.

---

## Audit & Verification

To verify that generated submissions strictly satisfy all competition rules and invariants (row counts, column headers, probability bounds, zero NaNs, natural prevalence):

```bash
python3 src/main.py --stage audit
```

**Checked Invariants:**
1. Exactly 30,000 test rows matching input `ID` values.
2. Columns strictly formatted as `['ID', 'Target']`.
3. Zero `NaN`, `null`, or infinite values.
4. Bounded probabilities within $[\text{FLOOR}=0.0020, \text{CEIL}=0.9995]$.
5. Natural empirical training prevalence calibration (~0.1500).

---

## Leak-Free Validation & Benchmarking

To run the automated leak-free regression test suite:

```bash
pytest tests/ -v
```

**Verified Invariants:**
1. **Fold-Isolated Feature Screening**: Feature screening runs strictly within outer training folds with zero validation target leakage (`tests/test_leakage_invariance.py`).
2. **Cross-Fitted Calibration**: `blend_and_calibrate()` and Stage 5 perform strictly out-of-sample probability calibration (`tests/test_calibration_crossfit.py`).
3. **De-risked Distillation**: Stage 4 students train exclusively on labeled fold data without test pseudo-labeling or probability sharpening (`tests/test_stage4_derisk.py`).
4. **Regularized Stacking**: Greedy OOF hill-climbing and multiplicative prevalence scaling are purged in Stage 5 (`tests/test_stage5_stacker.py`).

To run the 20% shadow holdout benchmark:

```bash
python3 experiments/run_shadow_holdout_benchmark.py
```

---

## Remote Kaggle Execution

For running on Kaggle GPU kernels via proxy WebSocket:

```bash
# Set your Kaggle session proxy URL if needed
export KAGGLE_PROXY_URL="<your_kaggle_proxy_url>"

# Execute runner
python3 kaggle_runner.py
```

---

## Hardware & Runtime Estimates

| Component | Hardware | Full Run (`max`) | Fast Run (`fast`) |
| :--- | :--- | :---: | :---: |
| **Feature Extraction Engine** | Multi-threaded CPU | ~3 min | ~45 sec |
| **Stage 1 (Baseline)** | Tesla T4 GPU / CPU | ~12 min | ~3 min |
| **Stage 2 (GBDT Zoo)** | Tesla T4 GPU | ~1.5 hours | ~8 min |
| **Stage 3 (TabPFN Priors)** | Tesla T4 GPU / CPU | ~40 min | Skipped |
| **Stage 4 (Distillation & TabMLP)** | Tesla T4 GPU | ~45 min | ~5 min |
| **Stage 5 (Meta-Stacker & Calibration)** | CPU | ~5 min | ~1 min |
| **Total** | **1x Tesla T4 + 4 vCPUs** | **~3.5 hours** | **~18 minutes** |

---

## Technical Documentation
For full architectural details, mathematical formulations, and validation metrics, refer to [`documentation.md`](file:///home/olande/PycharmProjects/final/documentation.md).
