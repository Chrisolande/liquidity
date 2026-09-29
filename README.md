# AI4EAC Liquidity Stress Prediction Pipeline

End-to-end machine learning pipeline for predicting customer 30-day liquidity stress risk (`liquidity_stress_next_30d`) from multi-month mobile money transaction histories, agent banking behavior, and account dynamics.

---

## Table of Contents
1. [Overview & Metric](#overview--metric)
2. [Directory Structure](#directory-structure)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Dataset Preparation](#dataset-preparation)
5. [How to Run End-to-End](#how-to-run-end-to-end)
   - [Single-Command Execution](#1-single-command-end-to-end)
   - [Stage-by-Stage Execution](#2-stage-by-stage-modular-execution)
   - [Execution Budgets (`LSEW_BUDGET`)](#3-execution-budgets)
6. [Audit & Verification](#audit--verification)
7. [Remote Kaggle Execution](#remote-kaggle-execution)
8. [Hardware & Runtime Estimates](#hardware--runtime-estimates)

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

## How to Run End-to-End

### 1. Single-Command End-to-End
To run the entire pipeline through feature extraction, base models, foundation priors, distillation, stacking, and submission generation:

```bash
python3 src/main.py --stage all --train path/to/Train.csv --test path/to/Test.csv
```

Outputs will be saved to:
- Checkpoints & out-of-fold predictions: `checkpoints/`
- Final submission CSVs: `submissions/`

---

### 2. Stage-by-Stage Modular Execution
You can run individual stages sequentially:

#### **Stage 1: Baseline Champion Reproduction**
Extracts baseline features and trains the benchmark CatBoost + HistGB ensemble:
```bash
python3 src/main.py --stage 1 --train path/to/Train.csv --test path/to/Test.csv
```
*Artifacts*: `checkpoints/oof_champ_train.npy`, `submissions/submission_best_0.73731.csv`

#### **Stage 2: Multi-Seed 10-Fold GBDT Zoo**
Trains CatBoost, XGBoost, and LightGBM across multiple random seeds with iterative stratification:
```bash
python3 src/main.py --stage 2 --train path/to/Train.csv --test path/to/Test.csv
```
*Artifacts*: `checkpoints/gbdt_zoo_4seed.npz`, `submissions/submission_s2_gbdt_zoo_0.73733.csv`

#### **Stage 3: TabPFN Foundation Priors**
Fits chunked prior models on physics and domain feature views:
```bash
python3 src/main.py --stage 3 --train path/to/Train.csv --test path/to/Test.csv
```
*Artifacts*: `checkpoints/tabpfn.npz`

#### **Stage 4: Distillation Students & TabMLP**
Computes soft teacher pseudo-labels with temperature sharpening ($T=0.85$) and trains regression students alongside PyTorch TabMLP:
```bash
python3 src/main.py --stage 4 --train path/to/Train.csv --test path/to/Test.csv
```
*Artifacts*: `checkpoints/diversity_stage4.npz`

#### **Stage 5: Meta-Stacker, Hill Climbing & Final Blending**
Blends all out-of-fold prediction streams using Nelder-Mead and hill-climbing optimization, followed by Platt/Isotonic probability calibration:
```bash
python3 src/main.py --stage 5 --test path/to/Test.csv
```
*Artifacts*: `submissions/submission_stage5_meta_stacker.csv` and calibrated deliverables.

---

### 3. Execution Budgets

Control the training depth and runtime using the `LSEW_BUDGET` environment variable:

```bash
# Smoke test (quick sanity check on 3 folds, ~1 minute)
LSEW_BUDGET=smoke python3 src/main.py --stage all

# Fast preset (5-fold, 2 seeds, reduced iterations, ~15-20 minutes on GPU)
LSEW_BUDGET=fast python3 src/main.py --stage all

# Balanced preset (10-fold, 3 seeds, TabPFN enabled, ~1.5 hours on GPU)
LSEW_BUDGET=balanced python3 src/main.py --stage all

# Max production preset (10-fold, 4 seeds, full feature set, ~3.5 hours on GPU)
LSEW_BUDGET=max python3 src/main.py --stage all
```

| Budget | Folds | GBDT Seeds | TabPFN | TabMLP | Screened Features | Typical GPU Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `smoke` | 3 | 1 (42) | No | No | 30 | ~1 min |
| `fast` | 5 | 2 (42, 100) | No | Yes (12 ep) | 80 | ~15-20 min |
| `balanced` | 10 | 3 (42, 100, 2024) | Yes (2 views) | Yes (25 ep) | 120 | ~1.5 hours |
| `max` | 10 | 4 (42, 100, 2024, 777) | Yes (3 views) | Yes (40 ep) | Full (All) | ~3.5 hours |

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
