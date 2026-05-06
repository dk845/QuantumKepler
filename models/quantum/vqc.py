import pandas as pd
import numpy as np
import yaml
import os
import json
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

FEATURES_PATH = config["paths"]["features"]
N_QUBITS = config["quantum"]["n_qubits"]
N_LAYERS = config["quantum"]["n_layers"]
BACKEND = config["quantum"]["backend"]
VQC_CONFIG = config["quantum"]["vqc"]

dev = qml.device(BACKEND, wires=N_QUBITS)

def ansatz(weights, x):
    """
    Variational circuit:
    1. Encode data via RY rotations
    2. Apply trainable layers with CNOT entanglement
    """
    # Data encoding
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)

    # Variational layers
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev)
def circuit(weights, x):
    ansatz(weights, x)
    return qml.expval(qml.PauliZ(0))

def predict_proba(weights, X):
    probs = []
    for x in X:
        val = circuit(weights, x)
        prob = (float(val) + 1) / 2  # map [-1,1] to [0,1]
        probs.append(prob)
    return np.array(probs)

def cost(weights, X, y):
    preds = []
    for x in X:
        val = circuit(weights, x)
        preds.append(val)
    preds = pnp.array(preds)
    # Map labels from {0,1} to {-1,1}
    y_mapped = 2 * pnp.array(y, requires_grad=False) - 1
    loss = pnp.mean((preds - y_mapped) ** 2)
    return loss

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4)
    }

def train():
    df = pd.read_csv(os.path.join(FEATURES_PATH, "features.csv"))
    feature_cols = [c for c in df.columns if c not in ["star_name", "label", "mission"]]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Use first N_QUBITS features
    X_train_q = X_train[:, :N_QUBITS]
    X_test_q = X_test[:, :N_QUBITS]

    # Initialize random weights
    np.random.seed(VQC_CONFIG["random_state"])
    weights = pnp.array(
        np.random.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS, 2)),
        requires_grad=True
    )

    optimizer = qml.AdamOptimizer(stepsize=VQC_CONFIG["learning_rate"])

    print("Training VQC...")
    steps = VQC_CONFIG["steps"]
    for step in range(steps):
        weights, loss_val = optimizer.step_and_cost(
            lambda w: cost(w, X_train_q, y_train),
            weights
        )
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{steps} - Loss: {float(loss_val):.4f}")

    # Evaluate
    print("\nEvaluating VQC...")
    y_prob = predict_proba(weights, X_test_q)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = evaluate(y_test, y_pred, y_prob)

    print("\n--- VQC Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    results = {
        "model": "vqc",
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist()
    }

    with open("experiments/vqc_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to experiments/vqc_results.json")
    return weights, metrics

if __name__ == "__main__":
    train()