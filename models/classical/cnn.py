import pandas as pd
import numpy as np
import yaml
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["paths"]["raw_data"]
CNN_CONFIG = config["classical"]["cnn"]
SEQ_LEN = 400  # fixed length for all light curves

class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LightCurveCNN(nn.Module):
    def __init__(self):
        super(LightCurveCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # Calculate flattened size
        self.fc1 = nn.Linear(64 * (SEQ_LEN // 8), 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()

def load_raw_sequences():
    meta = pd.read_csv(os.path.join(RAW_PATH, "metadata.csv"))
    X, y = [], []

    for _, row in meta.iterrows():
        star = row["star_name"]
        label = row["label"]
        mission = row["mission"]

        safe_name = star.replace("-", "_").replace(" ", "_")
        fname = os.path.join(RAW_PATH, f"{safe_name}_{mission}_label{label}.csv")

        if not os.path.exists(fname):
            fname = os.path.join(RAW_PATH, f"{safe_name}_label{label}.csv")

        if not os.path.exists(fname):
            continue

        df = pd.read_csv(fname)
        if "flux" not in df.columns or len(df) < SEQ_LEN:
            continue

        flux = df["flux"].values[:SEQ_LEN]
        flux = (flux - flux.mean()) / (flux.std() + 1e-8)
        X.append(flux)
        y.append(label)

    return np.array(X), np.array(y)

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4)
    }

def train():
    print("Loading raw light curve sequences...")
    X, y = load_raw_sequences()
    print(f"Dataset: {X.shape[0]} stars, sequence length {SEQ_LEN}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    train_ds = LightCurveDataset(X_train, y_train)
    test_ds = LightCurveDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=CNN_CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CNN_CONFIG["batch_size"])

    model = LightCurveCNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_CONFIG["learning_rate"])

    print("\nTraining CNN...")
    for epoch in range(CNN_CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{CNN_CONFIG['epochs']} - Loss: {total_loss:.4f}")

    # Evaluate
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            probs = model(X_batch)
            preds = (probs > 0.5).float()
            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_true.extend(y_batch.numpy())

    metrics = evaluate(all_true, all_preds, all_probs)

    print("\n--- CNN Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    results = {
        "model": "cnn",
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(all_true, all_preds).tolist(),
        "y_test": [int(x) for x in all_true],
        "y_pred": [int(x) for x in all_preds],
        "y_prob": [float(x) for x in all_probs]
    }

    with open("experiments/cnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to experiments/cnn_results.json")
    return model, metrics

if __name__ == "__main__":
    train()