import json
import pandas as pd
import os

def load_results():
    models = {}
    files = {
        "Random Forest": "experiments/random_forest_results.json",
        "CNN": "experiments/cnn_results.json",
        "Quantum Kernel SVM": "experiments/quantum_kernel_results.json",
        "VQC": "experiments/vqc_results.json"
    }
    for name, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                models[name] = json.load(f)
    return models

def compare():
    models = load_results()
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print("=" * 70)
    print("         QUANTUMKEPLER — FULL MODEL COMPARISON")
    print("=" * 70)

    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        best_val = -1
        best_model = ""
        for name, data in models.items():
            val = data["metrics"][metric]
            row[name] = val
            if val > best_val:
                best_val = val
                best_model = name
        row["Winner"] = f"{best_model} ✅"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("WINS PER MODEL:")
    win_counts = {}
    for row in rows:
        winner = row["Winner"].replace(" ✅", "")
        win_counts[winner] = win_counts.get(winner, 0) + 1
    for model, wins in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {wins}/{len(metrics)} metrics")
    print("=" * 70)

if __name__ == "__main__":
    compare()