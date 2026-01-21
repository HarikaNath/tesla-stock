#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.path.isfile("Tesla Data Set.zip")


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)


# In[11]:


import os

os.listdir(".")


# In[13]:


['Tesla Data Set.zip', ...]


# In[15]:


import zipfile

zip_path = "Tesla Data Set.zip"   # exact name from your error context
extract_path = "data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ ZIP extracted into /data folder")


# In[17]:


import os

os.listdir("data")


# In[19]:


from pathlib import Path
import pandas as pd

data_dir = Path("data")

csv_files = list(data_dir.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in data/")

file_path = csv_files[0]   # take the first CSV
df = pd.read_csv(file_path)

print("Loaded file:", file_path.name)
print("Shape:", df.shape)
df.head()


# In[21]:


# Clean column names
df.columns = [c.strip() for c in df.columns]

# Normalize Close/Last → Close
if "Close/Last" in df.columns:
    df = df.rename(columns={"Close/Last": "Close"})

# Detect date column
if "Date" not in df.columns:
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df = df.rename(columns={date_col: "Date"})

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Clean numeric columns (handles $ and commas)
for c in df.columns:
    if c != "Date":
        df[c] = (
            df[c].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

df.head()


# In[23]:


df.columns
df.tail()


# In[25]:


import numpy as np
import pandas as pd

# Ensure correct order
df = df.sort_values("Date").reset_index(drop=True)

# Set Date as index (time-series best practice)
df.set_index("Date", inplace=True)

df.info()


# In[27]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.plot(df.index, df["Close"])
plt.title("Tesla Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()


# In[29]:


plt.figure(figsize=(10,4))
plt.plot(df.index, df["Volume"], color="orange")
plt.title("Tesla Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.show()


# In[31]:


# Returns
df["daily_return"] = df["Close"].pct_change()
df["log_return"] = np.log(df["Close"]).diff()

# Rolling volatility
df["vol_5"] = df["log_return"].rolling(5).std()
df["vol_10"] = df["log_return"].rolling(10).std()
df["vol_20"] = df["log_return"].rolling(20).std()

df[["daily_return", "vol_5", "vol_10", "vol_20"]].tail()


# In[33]:


plt.figure(figsize=(10,4))
plt.plot(df.index, df["vol_20"])
plt.title("20-Day Rolling Volatility (Tesla)")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()


# In[35]:


# Moving averages
df["sma_10"] = df["Close"].rolling(10).mean()
df["sma_20"] = df["Close"].rolling(20).mean()
df["sma_50"] = df["Close"].rolling(50).mean()

# Momentum
df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

df[["sma_10","sma_20","sma_50","momentum_5"]].tail()


# In[37]:


plt.figure(figsize=(10,4))
plt.plot(df.index, df["Close"], label="Close")
plt.plot(df.index, df["sma_20"], label="SMA 20")
plt.plot(df.index, df["sma_50"], label="SMA 50")
plt.legend()
plt.title("Tesla Price with Moving Averages")
plt.grid(True)
plt.show()


# In[39]:


# Regression target
df["target_next_return"] = df["Close"].shift(-1) / df["Close"] - 1

# Classification target (high risk day)
vol_threshold = df["vol_20"].quantile(0.75)
df["target_high_risk"] = (df["vol_20"].shift(-1) > vol_threshold).astype(int)

df[["target_next_return", "target_high_risk"]].tail()


# In[41]:


df_ml = df.dropna()

features = [
    "daily_return","vol_5","vol_10","vol_20",
    "sma_10","sma_20","sma_50",
    "momentum_5","momentum_10",
    "Volume"
]

X = df_ml[features]
y_reg = df_ml["target_next_return"]
y_clf = df_ml["target_high_risk"]

print("Final dataset shape:", X.shape)


# In[43]:


split = int(len(df_ml) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_reg_train, y_reg_test = y_reg.iloc[:split], y_reg.iloc[split:]
y_clf_train, y_clf_test = y_clf.iloc[:split], y_clf.iloc[split:]

print("Train:", X_train.shape, "Test:", X_test.shape)


# In[45]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

# Baseline Regression
reg_base = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

# Baseline Classification
clf_base = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000))
])

reg_base.fit(X_train, y_reg_train)
clf_base.fit(X_train, y_clf_train)

pred_ret_base = reg_base.predict(X_test)
prob_risk_base = clf_base.predict_proba(X_test)[:, 1]

print("✅ BASELINE — Regression (Next-day return)")
print("MAE :", mean_absolute_error(y_reg_test, pred_ret_base))
print("RMSE:", mean_squared_error(y_reg_test, pred_ret_base, squared=False))
print("R2  :", r2_score(y_reg_test, pred_ret_base))

print("\n✅ BASELINE — Classification (High-risk day)")
print("AUC :", roc_auc_score(y_clf_test, prob_risk_base))


# In[47]:


from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier

reg_adv = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

clf_adv = ExtraTreesClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

reg_adv.fit(X_train, y_reg_train)
clf_adv.fit(X_train, y_clf_train)

pred_ret_adv = reg_adv.predict(X_test)
prob_risk_adv = clf_adv.predict_proba(X_test)[:, 1]
pred_risk_adv = (prob_risk_adv >= 0.5).astype(int)

print("✅ ADVANCED — Regression (ExtraTrees)")
print("MAE :", mean_absolute_error(y_reg_test, pred_ret_adv))
print("RMSE:", mean_squared_error(y_reg_test, pred_ret_adv, squared=False))
print("R2  :", r2_score(y_reg_test, pred_ret_adv))

print("\n✅ ADVANCED — Classification (ExtraTrees)")
print("AUC :", roc_auc_score(y_clf_test, prob_risk_adv))


# In[49]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay

# 1) Actual vs predicted (regression)
plt.figure(figsize=(10,4))
plt.plot(y_reg_test.values, label="Actual next-day return")
plt.plot(pred_ret_adv, label="Predicted next-day return")
plt.title("Trend Model: Actual vs Predicted Next-Day Returns (Test set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Confusion matrix (classification)
cm = confusion_matrix(y_clf_test.values, pred_risk_adv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("Risk Model Confusion Matrix (Test set)")
plt.tight_layout()
plt.show()

# 3) ROC curve (classification)
RocCurveDisplay.from_predictions(y_clf_test.values, prob_risk_adv)
plt.title("Risk Model ROC Curve (Test set)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[51]:


import pandas as pd

imp_reg = pd.Series(reg_adv.feature_importances_, index=X_train.columns).sort_values(ascending=False)
imp_clf = pd.Series(clf_adv.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print("Top 10 features driving RETURN prediction:")
display(imp_reg.head(10))

print("Top 10 features driving RISK prediction:")
display(imp_clf.head(10))

# Plot top features for regression
plt.figure(figsize=(8,4))
imp_reg.head(10).sort_values().plot(kind="barh")
plt.title("Top Features (Trend Model)")
plt.tight_layout()
plt.show()

# Plot top features for classification
plt.figure(figsize=(8,4))
imp_clf.head(10).sort_values().plot(kind="barh")
plt.title("Top Features (Risk Model)")
plt.tight_layout()
plt.show()


# In[53]:


from sklearn.base import clone

def walk_forward_regression(model, X, y, initial_train_frac=0.7, step=50):
    n = len(X)
    start = int(n * initial_train_frac)
    preds, actuals = [], []

    for i in range(start, n, step):
        X_tr, y_tr = X.iloc[:i], y.iloc[:i]
        X_te, y_te = X.iloc[i:i+step], y.iloc[i:i+step]
        if len(X_te) == 0:
            break

        m = clone(model)
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)

        preds.append(p)
        actuals.append(y_te.values)

    return np.concatenate(actuals), np.concatenate(preds)

# Use entire df_ml (not just test) for walk-forward
actual_wf, pred_wf = walk_forward_regression(
    ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=2),
    X, y_reg, initial_train_frac=0.7, step=50
)

print("✅ WALK-FORWARD REGRESSION RESULTS (more realistic than one split)")
print("MAE :", mean_absolute_error(actual_wf, pred_wf))
print("RMSE:", mean_squared_error(actual_wf, pred_wf, squared=False))
print("R2  :", r2_score(actual_wf, pred_wf))

plt.figure(figsize=(10,4))
plt.plot(actual_wf, label="Actual")
plt.plot(pred_wf, label="Predicted")
plt.title("Walk-Forward Validation: Actual vs Predicted Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[55]:


# Strategy: go long if predicted return > 0, else stay out (0)
signal = (pred_ret_adv > 0).astype(int)

strategy_ret = signal * y_reg_test.values
buy_hold_ret = y_reg_test.values

cum_strategy = np.cumprod(1 + strategy_ret)
cum_buyhold  = np.cumprod(1 + buy_hold_ret)

plt.figure(figsize=(10,4))
plt.plot(cum_strategy, label="ML Strategy (Long if pred>0)")
plt.plot(cum_buyhold, label="Buy & Hold")
plt.title("Backtest (Test Period): Strategy vs Buy & Hold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final cumulative return (Strategy):", float(cum_strategy[-1]))
print("Final cumulative return (Buy & Hold):", float(cum_buyhold[-1]))

# Optional: basic risk stats
print("\nStrategy avg daily return:", float(np.mean(strategy_ret)))
print("Strategy daily volatility:", float(np.std(strategy_ret)))


# In[57]:


import joblib

joblib.dump(reg_adv, "tsla_trend_model_expert.joblib")
joblib.dump(clf_adv, "tsla_risk_model_expert.joblib")

print("✅ Saved:")
print(" - tsla_trend_model_expert.joblib")
print(" - tsla_risk_model_expert.joblib")


# In[ ]:




