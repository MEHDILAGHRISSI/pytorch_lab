import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ==========================================
# 1. LOAD & ENGINEER FEATURES
# ==========================================
print("Loading dataset...")
# Update path if the file is inside data_set/
df = pd.read_csv("data_set/prices-split-adjusted.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)


def create_features(group):
    group = group.copy()

    # Base daily return
    group["return_today"] = group["close"].pct_change()

    # Target: Next-day return (shifted back by -1)
    group["target_return"] = group["return_today"].shift(-1)

    # Lagged return signals (past history)
    group["return_lag_1"] = group["return_today"].shift(1)
    group["return_lag_2"] = group["return_today"].shift(2)
    group["return_lag_3"] = group["return_today"].shift(3)
    group["return_lag_5"] = group["return_today"].shift(5)

    # Intraday spreads and volatility
    group["hl_spread"] = (group["high"] - group["low"]) / group["low"]
    group["co_spread"] = (group["close"] - group["open"]) / group["open"]

    # Moving average and volume ratios
    sma_20 = group["close"].rolling(20).mean()
    group["sma_20_ratio"] = group["close"] / sma_20
    group["volatility_20"] = group["return_today"].rolling(20).std()
    group["volume_ratio_20"] = group["volume"] / group["volume"].rolling(20).mean()

    return group


print("Engineering features per ticker symbol...")
df_featured = df.groupby("symbol", group_keys=False).apply(create_features, include_groups=False)
df_clean = df_featured.dropna().reset_index(drop=True)

# ==========================================
# 2. CHRONOLOGICAL DATA SPLIT
# ==========================================
train_df = df_clean[df_clean["date"] < "2015-01-01"]
val_df = df_clean[(df_clean["date"] >= "2015-01-01") & (df_clean["date"] < "2016-01-01")]
test_df = df_clean[df_clean["date"] >= "2016-01-01"]

feature_cols = [
    "return_today", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "hl_spread", "co_spread", "sma_20_ratio", "volatility_20", "volume_ratio_20"
]

X_train, y_train = train_df[feature_cols], train_df["target_return"]
X_val, y_val = val_df[feature_cols], val_df["target_return"]
X_test, y_test = test_df[feature_cols], test_df["target_return"]

# ==========================================
# 3. FEATURE SCALING (FIT ON TRAIN ONLY)
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
# Baseline 1: Ridge Linear Regression
print("Training Ridge Regression baseline...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Baseline 2: XGBoost Regressor
print("Training XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=False
)


# ==========================================
# 5. EVALUATION METRICS & DIRECTIONAL ACCURACY
# ==========================================
def evaluate_predictions(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Financial Hit Rate: % of time the model gets the sign right (+ or -)
    directional_acc = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100

    print(f"\n--- {model_name} Test Performance ---")
    print(f"RMSE:                 {rmse:.5f}")
    print(f"MAE:                  {mae:.5f}")
    print(f"R² Score:             {r2:.5f}")
    print(f"Directional Accuracy: {directional_acc:.2f}%")


# Generate test predictions
ridge_preds = ridge_model.predict(X_test_scaled)
xgb_preds = xgb_model.predict(X_test_scaled)

# Print evaluation reports
evaluate_predictions(y_test, ridge_preds, "Ridge Regression")
evaluate_predictions(y_test, xgb_preds, "XGBoost Regressor")