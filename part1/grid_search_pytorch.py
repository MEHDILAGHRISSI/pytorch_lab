import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seed for reproducibility
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
np.random.seed(TORCH_SEED)

# Select GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. DATA PREPARATION & FEATURE ENGINEERING
# ==========================================
csv_paths = ["prices-split-adjusted.csv", "data_set/prices-split-adjusted.csv"]
data_path = next((p for p in csv_paths if os.path.exists(p)), None)

if data_path is None:
    raise FileNotFoundError("Could not find 'prices-split-adjusted.csv'.")

df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

def create_features(group):
    group = group.copy()
    group["return_today"] = group["close"].pct_change()
    group["target_return"] = group["return_today"].shift(-1)

    group["return_lag_1"] = group["return_today"].shift(1)
    group["return_lag_2"] = group["return_today"].shift(2)
    group["return_lag_3"] = group["return_today"].shift(3)
    group["return_lag_5"] = group["return_today"].shift(5)

    group["hl_spread"] = (group["high"] - group["low"]) / group["low"]
    group["co_spread"] = (group["close"] - group["open"]) / group["open"]

    sma_20 = group["close"].rolling(20).mean()
    group["sma_20_ratio"] = group["close"] / sma_20
    group["volatility_20"] = group["return_today"].rolling(20).std()
    group["volume_ratio_20"] = group["volume"] / group["volume"].rolling(20).mean()
    return group

df_featured = df.groupby("symbol", group_keys=False).apply(create_features, include_groups=False)
df_clean = df_featured.dropna().reset_index(drop=True)

train_df = df_clean[df_clean["date"] < "2015-01-01"]
test_df  = df_clean[df_clean["date"] >= "2016-01-01"]

feature_cols = [
    "return_today", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "hl_spread", "co_spread", "sma_20_ratio", "volatility_20", "volume_ratio_20"
]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test  = scaler.transform(test_df[feature_cols])

y_train = train_df["target_return"].values.astype(np.float32)
y_test  = test_df["target_return"].values.astype(np.float32)

# ==========================================
# 2. SKLEARN WRAPPER FOR PYTORCH MODEL
# ==========================================
class PyTorchStockRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=10, hidden_units=[64, 32], lr=0.001, optimizer_type='adam', epochs=10, batch_size=1024):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None

    def _build_model(self):
        layers = []
        curr_dim = self.input_dim
        for h in self.hidden_units:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = self._build_model().to(device)
        criterion = nn.MSELoss()

        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model_.train()
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = self.model_(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_tensor).squeeze(-1).cpu().numpy()
        return preds

# ==========================================
# 3. QUESTION 3: GRIDSEARCHCV EXECUTION
# ==========================================
print("\n--- Running sklearn GridSearchCV on PyTorch Model ---")

param_grid = {
    'lr': [0.001, 0.0005],
    'optimizer_type': ['adam', 'adamw'],
    'hidden_units': [[64, 32], [128, 64, 32]],
    'epochs': [10, 15]
}

base_model = PyTorchStockRegressor(input_dim=len(feature_cols))

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print("\n================ GridSearch Results ================")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Best CV Loss (MSE):  {-grid_search.best_score_:.6f}")

# Predict on Test Data (2016)
best_model = grid_search.best_estimator_
test_preds = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, test_preds))
mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)
directional_acc = np.mean(np.sign(test_preds) == np.sign(y_test)) * 100.0

print("\n--- Best Model Out-of-Sample Test Performance ---")
print(f"RMSE:                 {rmse:.5f}")
print(f"MAE:                  {mae:.5f}")
print(f"R² Score:             {r2:.5f}")
print(f"Directional Accuracy: {directional_acc:.2f}%")