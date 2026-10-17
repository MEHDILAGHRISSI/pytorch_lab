import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set seed for reproducibility
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
np.random.seed(TORCH_SEED)

# Device Configuration (GPU if available, otherwise CPU)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
if use_cuda:
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

dataloader_kwargs = {"pin_memory": True} if use_cuda else {}

# ==========================================
# 1. DATA PREPARATION & FEATURE ENGINEERING
# ==========================================
csv_paths = ["prices-split-adjusted.csv", "data_set/prices-split-adjusted.csv"]
data_path = next((p for p in csv_paths if os.path.exists(p)), None)

if data_path is None:
    raise FileNotFoundError("Could not find 'prices-split-adjusted.csv' in current directory or 'data_set/'.")

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

def create_features(group):
    group = group.copy()
    group["return_today"] = group["close"].pct_change()

    # Regression Target Y_{t+1}: Next-day percentage return
    group["target_return"] = group["return_today"].shift(-1)

    # Input features (day t and earlier)
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

# Chronological Train / Validation / Test Split
train_df = df_clean[df_clean["date"] < "2015-01-01"]
val_df   = df_clean[(df_clean["date"] >= "2015-01-01") & (df_clean["date"] < "2016-01-01")]
test_df  = df_clean[df_clean["date"] >= "2016-01-01"]

feature_cols = [
    "return_today", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "hl_spread", "co_spread", "sma_20_ratio", "volatility_20", "volume_ratio_20"
]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_val   = scaler.transform(val_df[feature_cols])
X_test  = scaler.transform(test_df[feature_cols])

y_train = train_df["target_return"].values.astype(np.float32)
y_val   = val_df["target_return"].values.astype(np.float32)
y_test  = test_df["target_return"].values.astype(np.float32)

# Convert to PyTorch DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))
test_dataset  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, **dataloader_kwargs)
val_loader   = DataLoader(val_dataset, batch_size=2048, shuffle=False, **dataloader_kwargs)
test_loader  = DataLoader(test_dataset, batch_size=2048, shuffle=False, **dataloader_kwargs)

# ==========================================
# 2. PYTORCH DEEP NEURAL NETWORK (Q2 ARCHITECTURE)
# ==========================================
class StockRegressionDNN(nn.Module):
    def __init__(self, input_dim):
        super(StockRegressionDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # Single continuous output prediction
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

input_dim = len(feature_cols)
model = StockRegressionDNN(input_dim).to(device)

print("\n--- Deep Neural Network Architecture ---")
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
epochs = 15
print("\nTraining PyTorch Deep Neural Network...")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    train_loss = running_loss / len(train_dataset)

    # Validation evaluation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            preds = model(X_batch)
            val_loss += criterion(preds, y_batch).item() * X_batch.size(0)

    val_loss /= len(val_dataset)
    print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss (MSE): {train_loss:.6f} | Val Loss (MSE): {val_loss:.6f}")

# ==========================================
# 4. TEST EVALUATION
# ==========================================
model.eval()
test_preds, test_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device, non_blocking=True)
        preds = model(X_batch).cpu().numpy()
        test_preds.extend(preds)
        test_targets.extend(y_batch.numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)

rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
mae = mean_absolute_error(test_targets, test_preds)
r2 = r2_score(test_targets, test_preds)
directional_acc = np.mean(np.sign(test_preds) == np.sign(test_targets)) * 100.0

print("\n--- PyTorch DNN Regression Test Performance (2016) ---")
print(f"RMSE:                 {rmse:.5f}")
print(f"MAE:                  {mae:.5f}")
print(f"R² Score:             {r2:.5f}")
print(f"Directional Accuracy: {directional_acc:.2f}%")