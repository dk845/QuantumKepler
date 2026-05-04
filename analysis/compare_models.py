import json
import pandas as pd
import os

def load_results():
    with open("experiments/random_forest_results.json") as f:
        rf = json.load(f)
    with open("experiments/quantum_kernel_results.json") as f:
        qk = json.load(f)
    return rf, qk

def compare():
    rf, qk = load_results()

    print("=" * 50)
    print("   QUANTUMKEPLER — MODEL COMPARISON")
    print("=" * 50)

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []

    for m in metrics:
        rf_val = rf["metrics"][m]
        qk_val = qk["metrics"][m]
        winner = "Quantum ✅" if qk_val > rf_val else "Classical ✅"
        rows.append({
            "Metric": m,
            "Random Forest": rf_val,
            "Quantum Kernel SVM": qk_val,
            "Winner": winner
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n" + "=" * 50)
    qk_wins = sum(1 for r in rows if "Quantum" in r["Winner"])
    rf_wins = len(rows) - qk_wins
    print(f"  Quantum wins : {qk_wins}/{len(rows)} metrics")
    print(f"  Classical wins: {rf_wins}/{len(rows)} metrics")
    print("=" * 50)

if __name__ == "__main__":
    compare()