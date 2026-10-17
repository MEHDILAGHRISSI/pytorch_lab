import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Reproducibility
TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)
np.random.seed(TORCH_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. DATA PREPARATION
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
val_df   = df_clean[(df_clean["date"] >= "2015-01-01") & (df_clean["date"] < "2016-01-01")]

feature_cols = [
    "return_today", "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "hl_spread", "co_spread", "sma_20_ratio", "volatility_20", "volume_ratio_20"
]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_val   = scaler.transform(val_df[feature_cols])

y_train = train_df["target_return"].values.astype(np.float32)
y_val   = val_df["target_return"].values.astype(np.float32)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
val_dataset   = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2048, shuffle=False)

# ==========================================
# 2. BEST GRIDSEARCH ARCHITECTURE
# ==========================================
class OptimalStockDNN(nn.Module):
    def __init__(self, input_dim):
        super(OptimalStockDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)

model = OptimalStockDNN(len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 3. TRAINING & METRICS LOGGING
# ==========================================
epochs = 20
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("\nTraining best GridSearch DNN model...")
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = model(X_b)
        loss = criterion(preds, y_b)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_b.size(0)
        correct_train += torch.sum(torch.sign(preds) == torch.sign(y_b)).item()
        total_train += y_b.size(0)

    train_loss = running_loss / total_train
    train_acc = (correct_train / total_train) * 100.0

    # Validation
    model.eval()
    val_running_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b)
            loss = criterion(preds, y_b)

            val_running_loss += loss.item() * X_b.size(0)
            correct_val += torch.sum(torch.sign(preds) == torch.sign(y_b)).item()
            total_val += y_b.size(0)

    val_loss = val_running_loss / total_val
    val_acc = (correct_val / total_val) * 100.0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# ==========================================
# 4. QUESTION 4: PLOT LOSS & ACCURACY CURVES
# ==========================================
epochs_range = range(1, epochs + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss Graph
axes[0].plot(epochs_range, history["train_loss"], 'b-o', label="Training Loss (MSE)")
axes[0].plot(epochs_range, history["val_loss"], 'r-o', label="Validation Loss (MSE)")
axes[0].set_title("Loss (MSE) vs. Epochs", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Mean Squared Error")
axes[0].legend()
axes[0].grid(True)

# Accuracy Graph
axes[1].plot(epochs_range, history["train_acc"], 'b-o', label="Training Directional Accuracy (%)")
axes[1].plot(epochs_range, history["val_acc"], 'r-o', label="Validation Directional Accuracy (%)")
axes[1].set_title("Directional Accuracy (%) vs. Epochs", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("question4_learning_curves.png", dpi=300)
print("\nSaved Question 4 curves to 'question4_learning_curves.png'")