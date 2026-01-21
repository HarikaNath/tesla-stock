# Tesla Stock Trend & Risk Prediction (End-to-End ML Project)

This repository contains an end-to-end machine learning workflow using **Tesla (TSLA)** historical stock data to build:

- **Trend Model (Regression):** Predict next-day return  
- **Risk Model (Classification):** Predict whether the next day is a **high-volatility (high-risk)** day  
- **Evaluation & Visualizations:** Model performance metrics and charts  
- **(Optional) Decision Impact:** Simple strategy backtest vs Buy & Hold

This project is designed to be **interview-ready** by demonstrating:
- time-series best practices (no shuffling / leakage prevention)
- feature engineering (returns, volatility, momentum, moving averages)
- baseline → advanced modeling
- explainability through feature importance
- results storytelling (charts + business framing)

---

## Repository Contents

- **`Tesla.py`**  
  Main Python script with the full ML pipeline (loading, cleaning, feature engineering, modeling, evaluation, and plots).  
  👉 Open here: [`Tesla.py`](./Tesla.py)

- **`Tesla Data Set.zip`**  
  Tesla dataset archive used for this project (uploaded for convenience).  

- **`README.md`**  
  Project documentation (this file)

- **`.gitignore`**  
  Git ignore rules for local files

---

## Dataset

This project uses historical TSLA stock data with standard market columns:
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

If you want to use a different Tesla dataset, the pipeline will still work as long as these columns exist.

---

## Features Engineered

The model uses financial time-series features such as:

- **Returns:** daily return, log return  
- **Volatility:** rolling standard deviation of log returns (5/10/20 days)  
- **Trend signals:** moving averages (SMA 10/20/50)  
- **Momentum:** price momentum over multiple windows  
- **Volume features:** volume change, rolling volume averages  

---

## Targets

### 1) Trend Prediction (Regression)
Predict **next-day return**:
\[
r_{t+1} = \frac{Close_{t+1}}{Close_t} - 1
\]

### 2) Risk Prediction (Classification)
Predict whether the next day is a **high-volatility day**, defined as being above a percentile threshold (ex: top 25% volatility days).

---

## Models

### Baseline Models
- Ridge Regression (returns)
- Logistic Regression (risk)

### Advanced Models (Final)
- ExtraTrees Regressor
- ExtraTrees Classifier

Why ExtraTrees?
- strong performance without heavy tuning
- robust for noisy time-series features
- works well for non-linear relationships


