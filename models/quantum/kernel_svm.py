import pandas as pd
import numpy as np
import yaml
import os
import json
import pennylane as qml
from sklearn.svm import SVC
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

dev = qml.device(BACKEND, wires=N_QUBITS)

@qml.qnode(dev)
def quantum_kernel_circuit(x1, x2):
    """
    Compute quantum kernel between two data points.
    Encodes x1, then applies inverse encoding of x2.
    Inner product in quantum feature space.
    """
    # Encode x1
    for i in range(N_QUBITS):
        qml.RY(x1[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    # Inverse encode x2
    for i in range(N_QUBITS - 1, 0, -1):
        qml.CNOT(wires=[i - 1, i])
    for i in range(N_QUBITS):
        qml.RY(-x2[i], wires=i)

    return qml.probs(wires=range(N_QUBITS))

def quantum_kernel(X1, X2):
    """Build the full kernel matrix between X1 and X2."""
    kernel_matrix = np.zeros((len(X1), len(X2)))
    total = len(X1) * len(X2)
    print(f"  Computing kernel matrix ({len(X1)}x{len(X2)} = {total} evaluations)...")
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            probs = quantum_kernel_circuit(x1, x2)
            kernel_matrix[i, j] = probs[0]  # probability of |00...0> state
        if i % 5 == 0:
            print(f"  Row {i+1}/{len(X1)} done")
    return kernel_matrix

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

    # Use only first N_QUBITS features for quantum encoding
    X_train_q = X_train[:, :N_QUBITS]
    X_test_q = X_test[:, :N_QUBITS]

    print("Building quantum kernel matrices...")
    K_train = quantum_kernel(X_train_q, X_train_q)
    K_test = quantum_kernel(X_test_q, X_train_q)

    print("\nTraining Quantum Kernel SVM...")
    model = SVC(kernel="precomputed", probability=True)
    model.fit(K_train, y_train)

    y_pred = model.predict(K_test)
    y_prob = model.predict_proba(K_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_prob)

    print("\n--- Quantum Kernel SVM Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    results = {
        "model": "quantum_kernel_svm",
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "kernel_matrix_train": K_train.tolist()
    }

    with open("experiments/quantum_kernel_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to experiments/quantum_kernel_results.json")
    return model, metrics

if __name__ == "__main__":
    train()