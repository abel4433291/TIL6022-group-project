import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

tomtom = pd.read_parquet("tomtom_data.parquet")
vessels = pd.read_parquet("vessels_data.parquet")
sensors = pd.read_csv("TIL6022-group-project/sensor-location.xlsx - Sheet1.csv", parse_dates=["timestamp"])
sensors_location = pd.read_csv("TIL6022-group-project/sensordata_SAIL2025.csv")


# ---------------------------------------------------
# 2️⃣ Compute visitor flow (people / meter / minute)
# ---------------------------------------------------



# ---------------------------------------------------
# 3️⃣ Merge with vessel data
# ---------------------------------------------------
sensors["timestamp_3min"] = sensors["timestamp"].dt.floor("3min")
vessels["timestamp_3min"] = pd.to_datetime(vessels["timestamp"]).dt.floor("3min")

vessels_agg = (
    vessels.groupby("timestamp_3min")
    .agg(
        vessel_count=("imo-number", "nunique"),
        avg_length=("length", "mean"),
        mean_lat=("lat", "mean"),
        mean_lon=("lon", "mean"),
    )
    .reset_index()
)

df = flows.merge(vessels_agg, on="timestamp_3min", how="left").fillna(0)
df = df.sort_values("timestamp_3min")

# ---------------------------------------------------
# 4️⃣ Create features and targets
# ---------------------------------------------------
feature_cols = sensor_cols + ["vessel_count", "avg_length", "mean_lat", "mean_lon"]
target_cols = sensor_cols

df["target_time"] = df["timestamp_3min"].shift(-1)
target = df[target_cols].shift(-1).dropna()
X = df.loc[target.index, feature_cols]
y = target

# Normalize features
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / (X_std + 1e-6)

# ---------------------------------------------------
# 5️⃣ Split train / val / test (80 / 15 / 5)
# ---------------------------------------------------
n = len(X)
train_end = int(0.8 * n)
val_end = int(0.95 * n)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val     = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test   = X.iloc[val_end:], y.iloc[val_end:]

def to_tensor(a): return torch.tensor(a.values, dtype=torch.float32)

BATCH_SIZE = 64  # 🔹 define batch size here

train_dl = DataLoader(TensorDataset(to_tensor(X_train), to_tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(TensorDataset(to_tensor(X_val),   to_tensor(y_val)),   batch_size=BATCH_SIZE)
test_dl  = DataLoader(TensorDataset(to_tensor(X_test),  to_tensor(y_test)),  batch_size=BATCH_SIZE)

print(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ---------------------------------------------------
# 6️⃣ Define model
# ---------------------------------------------------
class FlowPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = FlowPredictor(input_dim=X.shape[1], output_dim=y.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------
# 7️⃣ Training loop with tqdm
# ---------------------------------------------------
EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            preds = model(xb)
            val_loss += criterion(preds, yb).item()

    train_loss /= len(train_dl)
    val_loss /= len(val_dl)

    # Print progress every 10 epochs
    if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# ---------------------------------------------------
# 8️⃣ Evaluate on test set
# ---------------------------------------------------
model.eval()
test_loss = 0.0
with torch.no_grad():
    for xb, yb in test_dl:
        preds = model(xb)
        test_loss += criterion(preds, yb).item()
test_loss /= len(test_dl)
print(f"✅ Final Test Loss: {test_loss:.4f}")

# ---------------------------------------------------
# 9️⃣ Save model
# ---------------------------------------------------
torch.save({
    "model_state": model.state_dict(),
    "X_mean": X_mean.to_dict(),
    "X_std": X_std.to_dict(),
    "feature_cols": feature_cols,
    "target_cols": target_cols,
}, "visitor_flow_predictor.pth")

print("✅ Model trained and saved as visitor_flow_predictor.pth")

