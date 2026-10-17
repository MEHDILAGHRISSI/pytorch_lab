# Deep Learning Lab — Regression & Multi-Class Classification with PyTorch

**Author:** Mehdi Laghrissi

**Institution:** Université Abdelmalek Essaâdi — FST Tanger

**Department:** Génie Informatique

**Cycle:** Ingénierie des Logiciels et Systèmes Intelligents

**Course:** Deep Learning

**Professor:** Pr. ELAACHAK LOTFI

**Repository:** [https://github.com/MEHDILAGHRISSI/pytorch_lab](https://github.com/MEHDILAGHRISSI/pytorch_lab)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Running Guide](#running-guide)
5. [Part 1 — Regression (NYSE Stock Returns)](#part-1--regression-nyse-stock-returns)
6. [Part 2 — Multi-Class Classification (Predictive Maintenance)](#part-2--multi-class-classification-predictive-maintenance)
7. [Key Findings](#key-findings)
8. [Limitations & Lessons Learned](#limitations--lessons-learned)
9. [References](#references)
10. [Appendix: Execution Screenshots](#appendix-execution-screenshots)

---

## Overview

This repository contains two independent, end-to-end Deep Learning pipelines built with **PyTorch**:

| Part | Task | Target | Dataset |
|------|------|--------|---------|
| **Part 1** | Regression | Next-day stock return | `prices-split-adjusted.csv` (NYSE, Kaggle) |
| **Part 2** | Multi-class Classification | Failure type (6 classes) | `predictive_maintenance.csv` (AI4I 2020, Kaggle) |

Each part follows the same ML/DL methodology:

> **EDA → Cleaning → Feature Engineering → Scaling → PyTorch Model → GridSearch → Learning Curves → Metrics → Regularization**

---

## Project Structure

```text
pytorch_lab/
├── assets/                                # Screenshots for documentation
├── part1/
│   ├── data_set/                          # NYSE CSVs (fundamentals, prices, securities)
│   ├── exploratory/                       # Part 1 — EDA scripts + HTML reports
│   ├── predict_stock_returns.py           # Q1  — Ridge & XGBoost baselines
│   ├── pytorch_model.py                   # Q2  — Baseline PyTorch DNN
│   ├── visualize_net.py                   #     — DNN architecture diagram
│   ├── grid_search_pytorch.py             # Q3  — sklearn GridSearchCV over PyTorch
│   ├── plot_learning_curves.py            # Q4  — Loss/Acc vs Epochs plots
│   ├── regularized_dnn.py                 # Q5  — Regularized DNN
│   ├── best_stock_dnn.pth                 # Saved weights (best checkpoint)
│   ├── dnn_architecture.jpg               # Rendered architecture
│   ├── question4_learning_curves.png      # Q4 output figure
│   └── question5_regularized_curves.png   # Q5 output figure
│
├── part2/
│   ├── dataset/                           # predictive_maintenance.csv + EDA HTML
│   ├── generate_ydata_profile.py          # Q2  — Automated EDA (ydata-profiling)
│   ├── preprocess.py                      # Q1  — Cleaning, feature eng., scaling, encoding
│   ├── dataset.py                         # Q3  — PyTorch Dataset + SMOTE balancing
│   ├── model.py                           # Q4  — PyTorch MLP architecture
│   ├── train.py                           # Q6/Q7 — Training, curves, metrics
│   ├── grid_search.py                     # Q5  — Hyperparameter grid search
│   ├── compare_regularization.py          # Q8  — Regularization comparison
│   ├── grid_search_results.csv            # GridSearch output
│   ├── regularization_comparison.csv      # Q8 output table
│   ├── training_performance.png           # Q6 output figure
│   └── regularization_comparison.png      # Q8 output figure
│
└── requirements.txt
```

---

## Installation

### Prerequisites
- Python 3.11+
- PyTorch (CPU or CUDA — see below)
- (Optional) NVIDIA GPU with a compatible driver

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU with driver **≥ 525**, install the CUDA-enabled PyTorch build:

```bash
# CUDA 11.8 (most compatible with older drivers)
pip install torch --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

# or CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

Verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3. Download the datasets

Both datasets come from Kaggle:

- **Part 1:** [dgawlik/nyse](https://www.kaggle.com/datasets/dgawlik/nyse) → place `prices-split-adjusted.csv` in `part1/data_set/`
- **Part 2:** [shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) → place `predictive_maintenance.csv` in `part2/dataset/`

> Large CSV files are **excluded** from Git via `.gitignore`. You must download them locally to reproduce the results.

---

## Running Guide

### Part 1 — Regression

Run the files **in this order** from the repository root:

| # | Command | What it does |
|---|---------|--------------|
| 1 | `python part1/exploratory/eda_prices_split_adjusted.py` | EDA on the stock price data (distribution, correlation, AAPL time-series) |
| 2 | `python part1/exploratory/eda_fundamentals.py` | EDA on financial fundamentals (missing values, target `Total Revenue`) |
| 3 | `python part1/predict_stock_returns.py` | **Q1** — Trains Ridge and XGBoost baselines to establish a benchmark |
| 4 | `python part1/pytorch_model.py` | **Q2** — Builds and trains the baseline PyTorch DNN |
| 5 | `python part1/visualize_net.py` | Renders the DNN architecture to `dnn_architecture.jpg` |
| 6 | `python part1/grid_search_pytorch.py` | **Q3** — Runs `GridSearchCV` over 16 PyTorch configurations |
| 7 | `python part1/plot_learning_curves.py` | **Q4** — Plots Loss vs Epochs and Accuracy vs Epochs → `question4_learning_curves.png` |
| 8 | `python part1/regularized_dnn.py` | **Q5** — Adds Dropout + Weight Decay and saves `best_stock_dnn.pth` |

### Part 2 — Multi-Class Classification

| # | Command | What it does |
|---|---------|--------------|
| 1 | `python part2/generate_ydata_profile.py` | **Q2** — Generates an interactive HTML EDA report (`predictive_maintenance_report.html`) |
| 2 | `python part2/preprocess.py` | **Q1** — Cleans data, engineers `Temp_Diff`, `Power`, `Overstrain`; scales; splits; encodes labels |
| 3 | `python part2/dataset.py` | **Q3** — Wraps data in a PyTorch `Dataset` and applies SMOTE balancing |
| 4 | `python part2/model.py` | **Q4** — Prints the MLP architecture for validation |
| 5 | `python part2/train.py` | **Q6/Q7** — Trains, plots `training_performance.png`, prints per-class metrics |
| 6 | `python part2/grid_search.py` | **Q5** — Grid-search over lr / batch / dropout / weight decay → `grid_search_results.csv` |
| 7 | `python part2/compare_regularization.py` | **Q8** — Compares 4 regularization strategies → `regularization_comparison.png` |

---

## Part 1 — Regression (NYSE Stock Returns)

### Objective

Predict the **next-day percentage return** of S&P 500 stocks from historical price and volume data.

### Why split-adjusted prices?

A stock split changes the raw price overnight without changing the company’s real value. Using raw prices would inject artificial 50% “crashes” into returns, moving averages, and volatility. `prices-split-adjusted.csv` removes these distortions, which is why it is the correct file for quantitative ML.

### Pipeline walkthrough (per file)

#### `exploratory/eda_prices_split_adjusted.py` — EDA
- Loaded 851,264 rows, 7 columns.
- Confirmed **0 missing values, 0 duplicates**.
- Numeric columns: `open`, `high`, `low`, `close`, `volume`.
- **High correlation** (r > 0.99) between `open`/`high`/`low`/`close` → confirms prices must be transformed to returns before modeling.

#### `exploratory/eda_fundamentals.py` — EDA (alternative dataset)
- 1,781 rows × 79 columns.
- 1.07% missing cells (`Cash Ratio`, `Current Ratio`, `Quick Ratio`, `EPS`, etc.).
- Target `Total Revenue` is highly skewed (min ≈ $1.5M, max ≈ $485B).
- Concluded that `prices-split-adjusted.csv` is better suited for daily-return regression.

#### `predict_stock_returns.py` — Q1 Baselines
**Feature engineering per ticker:**
- `return_today = close.pct_change()`
- `target_return = return_today.shift(-1)`
- Lagged returns (`lag_1`, `lag_2`, `lag_3`, `lag_5`)
- `hl_spread`, `co_spread`
- `sma_20_ratio = close / SMA(close, 20)`
- `volatility_20`, `volume_ratio_20`

**Chronological split:**
- Train: `< 2015-01-01`
- Validation: `2015`
- Test: `≥ 2016`

**Models trained:**

| Model | RMSE | MAE | R² | Directional Acc |
|-------|------|-----|----|-----------------|
| Ridge Regression | 0.02065 | 0.01233 | −0.00244 | **51.79%** |
| XGBoost Regressor | 0.02070 | 0.01235 | −0.00749 | **51.97%** |

> **Interpretation:** ~52% directional accuracy is above the 50% coin-flip baseline. For daily stock returns, this is a realistic ceiling — noise dominates.

#### `pytorch_model.py` — Q2 Baseline DNN
Architecture (10 → 64 → 32 → 16 → 1):
```
Linear(10, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1)
```
Loss: `MSELoss` · Optimizer: `Adam(lr=1e-3)` · 15 epochs

**Result on test (2016):**

| Metric | Value |
|--------|-------|
| RMSE | 0.02095 |
| MAE | 0.01254 |
| R² | −0.03198 |
| Directional Accuracy | **51.79%** |

**Observation:** Training loss decreased steadily (0.001356 → 0.000297), but validation loss bottomed around epoch 5 (0.000306) and then rose — the first sign of **overfitting**.

#### `grid_search_pytorch.py` — Q3 Hyperparameter Tuning
Used `GridSearchCV` with a sklearn-compatible `BaseEstimator` wrapper over:

- `lr ∈ {0.001, 0.0005}`
- `optimizer ∈ {Adam, AdamW}`
- `hidden_units ∈ {[64,32], [128,64,32]}`
- `epochs ∈ {10, 15}`

**Best configuration:** `lr=0.001, optimizer=Adam, hidden=[128,64,32], epochs=15, CV MSE=0.000292`

| Metric | Value |
|--------|-------|
| RMSE | 0.02113 |
| MAE | 0.01262 |
| R² | −0.05031 |
| Directional Accuracy | 51.52% |

> **Interpretation:** Giving the unregularized model *more capacity* improved cross-validation loss but worsened out-of-sample generalization. This is a textbook **capacity ↔ overfitting** trade-off.

#### `plot_learning_curves.py` — Q4 Curves
Trained the winning architecture (128→64→32) for 20 epochs and recorded:

- `Training Loss` vs `Validation Loss`
- `Training Directional Accuracy` vs `Validation Directional Accuracy`

Saved as **`question4_learning_curves.png`**.

**Interpretation:**
- Train loss falls (0.000447 → 0.000283).
- Val loss minimum ≈ epoch 5 (0.000306), then diverges (up to 0.000326+).
- Train accuracy rises to 53.94%; val accuracy oscillates around 49.5–51%.
- **Diagnosis: overfitting without regularization.**

#### `regularized_dnn.py` — Q5 Regularization
Added (light configuration):
- `Dropout(p=0.05)` after each hidden layer
- `AdamW(lr=5e-4, weight_decay=1e-5)`

**Result:**

| Metric | Value |
|--------|-------|
| RMSE | 0.02096 |
| MAE | 0.01252 |
| R² | −0.03290 |
| Directional Accuracy | 51.43% |

**Final Part 1 comparison:**

| Model | RMSE | MAE | R² | Dir. Acc |
|-------|------|-----|----|----------|
| Ridge (baseline) | 0.02065 | 0.01233 | −0.00244 | 51.79% |
| XGBoost (baseline) | 0.02070 | 0.01235 | −0.00749 | **51.97%** |
| Q2 Baseline DNN | 0.02095 | 0.01254 | −0.03198 | 51.79% |
| Q3 GridSearch DNN | 0.02113 | 0.01262 | −0.05031 | 51.52% |
| Q5 Regularized DNN | 0.02096 | 0.01252 | −0.03290 | 51.43% |

> **Interpretation:** No PyTorch model beat the classical baselines on directional accuracy. This is expected in financial time-series: signal-to-noise is extremely low, and expressive models mostly memorize noise. Regularization narrowed the train/val gap but did not create signal that wasn’t there.

> **Note:** A heavier regularization trial (`Dropout=0.2`, `weight_decay=1e-4`) reduced test RMSE to **0.02076** (best continuous error) but pushed directional accuracy to **48.07%** — a classic case of **over-smoothing** where predictions collapse toward the mean.

---

## Part 2 — Multi-Class Classification (Predictive Maintenance)

### Objective

Given sensor telemetry from industrial machines, predict the **type of failure** among 6 classes:

| ID | Class | Count (full dataset) |
|----|-------|----------------------|
| 0 | Heat Dissipation Failure | 112 |
| 1 | No Failure | 9,652 |
| 2 | Overstrain Failure | 78 |
| 3 | Power Failure | 95 |
| 4 | Random Failures | 19 |
| 5 | Tool Wear Failure | 45 |

The dataset is **severely imbalanced**: 96.5% of samples are `No Failure`.

### Pipeline walkthrough (per file)

#### `generate_ydata_profile.py` — Q2 Automated EDA
Generated an interactive HTML report highlighting:

- 10,000 rows × 10 columns, 0 missing values
- High correlation between `Air temperature` and `Process temperature`
- High correlation between `Rotational speed` and `Torque`
- Alerts on class imbalance (`Target`: 96.6% vs 3.4%; `Failure Type`: 88.7% for the dominant class)
- Unique IDs (`UDI`, `Product ID`) flagged for removal

#### `preprocess.py` — Q1 Preprocessing
**Cleaning:**
- Dropped `UDI`, `Product ID` (identifiers).
- Dropped `Target` (redundant when predicting `Failure Type` — prevents **target leakage**).

**Domain feature engineering:**
- `Temp_Diff = Process temperature - Air temperature` (heat dissipation load)
- `Power = Torque × Rotational speed` (mechanical power stress)
- `Overstrain = Tool wear × Torque` (accumulated wear + force)

**Encoding:**
- `Type ∈ {L, M, H}` → ordinal `{0, 1, 2}`.
- `Failure Type` string labels → `LabelEncoder` → integers `0…5` (required by `CrossEntropyLoss`).

**Scaling:** `StandardScaler` fit **only on training data**.

**Split:** `train_test_split(test_size=0.2, stratify=y)` → 8,000 train / 2,000 test, preserving the rare-class ratio.

**Class distribution after split (train):**
```
1  No Failure:                    7722
0  Heat Dissipation Failure:        90
3  Power Failure:                   76
2  Overstrain Failure:              62
5  Tool Wear Failure:               36
4  Random Failures:                 14
```

#### `dataset.py` — Q3 Balancing + PyTorch Dataset
- Wrapped data in a custom `PredictiveMaintenanceDataset(Dataset)` returning `(float32 feature, long label)`.
- Applied **SMOTE** (`imblearn`) **only on training data**.
- **Result:** each class has exactly 7,722 samples → 46,332 training rows total.
- Test set untouched (2,000 rows, original distribution).
- `DataLoader(batch_size=64)`.

#### `model.py` — Q4 Architecture
```
Linear(9, 128) → BatchNorm → ReLU → Dropout(0.2)
Linear(128, 64) → BatchNorm → ReLU → Dropout(0.2)
Linear(64, 32) → BatchNorm → ReLU → Dropout(0.2)
Linear(32, 6)   ← 6 logits (no softmax, CrossEntropy handles it)
```

Loss: `nn.CrossEntropyLoss` · Optimizer: `Adam` · 30 epochs · batch 64.

#### `train.py` — Q6/Q7 Training + Metrics

Training curve shape:
- Train loss: 0.4927 → 0.1103 (epoch 30)
- Test loss: 0.8007 → 0.3441

**Classification report on test set:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Heat Dissipation Failure | 0.4490 | **1.0000** | 0.6197 |
| No Failure | 0.9914 | 0.8943 | 0.9403 |
| Overstrain Failure | 0.6818 | **0.9375** | 0.7895 |
| Power Failure | 0.5769 | 0.7895 | 0.6667 |
| Random Failures | 0.0000 | 0.0000 | 0.0000 |
| Tool Wear Failure | 0.0000 | 0.0000 | 0.0000 |

| Metric | Value |
|--------|-------|
| Accuracy | 0.8890 |
| Macro F1 | 0.5027 |
| Weighted F1 | 0.9269 |

**Interpretation:**
- Physical failure modes (Heat Dissipation, Overstrain, Power) are captured with high recall — the engineered features align with the underlying physics.
- **Random Failures** and **Tool Wear Failure** fail completely. Reason: only 14 and 36 original training samples, respectively. SMOTE interpolation from so few points creates synthetic samples that overlap with `No Failure`.

#### `grid_search.py` — Q5 Hyperparameter Tuning
Grid over 16 combinations of `lr`, `batch_size`, `dropout_rate`, `weight_decay` (20 epochs each).

**Best configuration (by Macro F1):**
`lr=0.0005, batch_size=128, dropout=0.2, weight_decay=0.0001`

| Metric | Value |
|--------|-------|
| Test Accuracy | 88.85% |
| Test Loss | 0.3345 |
| Macro F1 | **0.5031** |

**Insights:**
- Batch size 128 consistently outperformed 64 on the SMOTE-expanded dataset.
- Dropout 0.2 beat 0.4 — heavy dropout on only 9 features destroys too much signal.

#### `compare_regularization.py` — Q8 Regularization Comparison

| Config | Train Acc | Test Acc | Gap | Test Loss | Macro F1 | Weighted F1 |
|--------|-----------|----------|-----|-----------|----------|-------------|
| No regularization | 99.10% | **94.00%** | 5.10% | 0.3689 | **0.5633** | 0.9577 |
| Dropout (0.3) only | 98.30% | 89.85% | 8.45% | 0.3476 | 0.4848 | 0.9307 |
| Weight Decay (1e-3) only | 98.79% | 91.90% | 6.89% | **0.2804** | 0.4995 | 0.9423 |
| Combined (BN + DO + L2) | 98.44% | 89.85% | 8.59% | 0.3206 | 0.4874 | 0.9311 |

**Interpretation:**
- The **unregularized model achieved the highest test accuracy (94%)** — SMOTE already provides an implicit form of regularization.
- **Weight Decay (L2)** produced the **lowest test loss (0.2804)** and a smaller train-test gap.
- **Combined regularization** is not automatically better — heavy dropout on a 9-feature MLP under-fits.
- **Key lesson:** *more regularization ≠ better results*. The optimal strategy depends on feature dimensionality and the natural noise of the data.

---

## Key Findings

### Part 1 — Regression
1. **Baselines matter.** Ridge and XGBoost reached ~51.8–52.0% directional accuracy — a realistic ceiling for daily stock returns.
2. **More capacity hurts.** GridSearch-selected deeper/wider DNNs improved CV loss but worsened out-of-sample generalization.
3. **Regularization narrows the train/val gap** but does not create signal where the signal-to-noise ratio is near zero.
4. **MSE and directional accuracy diverge.** Lower MSE does not imply better directional accuracy — heavy regularization can collapse predictions toward the mean and flip signs randomly.

### Part 2 — Multi-Class Classification
1. **Physical feature engineering is powerful.** `Temp_Diff`, `Power`, and `Overstrain` allowed the MLP to detect physical failure modes (Heat Dissipation, Overstrain, Power) with recall up to 100%.
2. **SMOTE is not a silver bullet.** For classes with <20 real samples, synthetic oversampling produces noisy, overlapping points that are not learnable.
3. **Stratified splitting + SMOTE on train only** preserves the test set’s true distribution, giving realistic metrics.
4. **Weighted F1 (~0.93–0.96)** shows excellent overall performance, but **Macro F1 (~0.50)** exposes the model’s weakness on the rarest classes.
5. **Regularization trade-off:** L2 weight decay was the most balanced single technique; combined heavy regularization degraded accuracy.

---

## Limitations & Lessons Learned

- **Very low signal-to-noise in daily returns** caps what any model can achieve without additional data (order book, news sentiment, macro factors, alternative data).
- **Class imbalance in failure detection** cannot be fixed by SMOTE alone when minority counts are extremely low. Real solutions require:
  - Collecting more rare-class examples
  - Anomaly detection on top of the classifier
  - Cost-sensitive loss (`CrossEntropyLoss(weight=...)`)
- **GridSearch is exhaustive but expensive.** Randomized search or Optuna would be more efficient for a larger grid.
- **Chronological splits are mandatory** for time-series regression, but they introduce drift (2015–2016 market regimes differ from pre-2015).
- **PyTorch GPU acceleration on small tabular MLPs is marginal.** PCIe transfer overhead dominates for models this small; CPU is often equivalent. Real GPU gains would require deeper networks or sequence models (LSTM/Transformer).

---

## References

- **Datasets**
  - NYSE dataset: [Kaggle — dgawlik/nyse](https://www.kaggle.com/datasets/dgawlik/nyse)
  - AI4I 2020 Predictive Maintenance: [Kaggle — shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
- **Libraries**
  - PyTorch, scikit-learn, imbalanced-learn, pandas, matplotlib, seaborn, ydata-profiling
- **Course**
  - Deep Learning — Pr. ELAACHAK LOTFI — Université Abdelmalek Essaâdi — FST Tanger



---

**Author:** Mehdi Laghrissi
**Course:** Deep Learning (Pr. ELAACHAK LOTFI)
**Academic Year:** 2025–2026
