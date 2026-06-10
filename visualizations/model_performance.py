import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def load_results():
    models = {
        "Random Forest": "experiments/random_forest_results.json",
        "CNN": "experiments/cnn_results.json",
        "Quantum Kernel SVM": "experiments/quantum_kernel_results.json",
        "VQC": "experiments/vqc_results.json"
    }
    data = {}
    for name, path in models.items():
        if os.path.exists(path):
            with open(path) as f:
                data[name] = json.load(f)
    return data

COLORS = {
    "Random Forest": "#F97316",
    "CNN": "#3B82F6",
    "Quantum Kernel SVM": "#A855F7",
    "VQC": "#22C55E"
}

def plot_3d_roc():
    data = load_results()
    fig = go.Figure()

    for idx, (name, result) in enumerate(data.items()):
        y_test = result["y_test"]
        y_prob = result["y_prob"]

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Normalize thresholds to 0-1
        thresh_norm = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min() + 1e-8)

        fig.add_trace(go.Scatter3d(
            x=fpr,
            y=tpr,
            z=thresh_norm,
            mode="lines",
            name=f"{name} (AUC={roc_auc:.2f})",
            line=dict(color=COLORS[name], width=6)
        ))

    fig.update_layout(
        title=dict(
            text="3D ROC Curves — Classical vs Quantum",
            font=dict(size=22, color="white")
        ),
        scene=dict(
            xaxis=dict(title="False Positive Rate", backgroundcolor="#0a0a0a", gridcolor="#333", color="white"),
            yaxis=dict(title="True Positive Rate", backgroundcolor="#0a0a0a", gridcolor="#333", color="white"),
            zaxis=dict(title="Threshold", backgroundcolor="#0a0a0a", gridcolor="#333", color="white"),
            bgcolor="#0a0a0a"
        ),
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333", borderwidth=1),
        height=700
    )

    os.makedirs("visualizations/output", exist_ok=True)
    out = "visualizations/output/3d_roc.html"
    fig.write_html(out)
    print(f"Saved: {out}")
    fig.show()

def plot_3d_metrics():
    data = load_results()
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = list(data.keys())

    fig = go.Figure()

    for m_idx, metric in enumerate(metrics):
        for name in model_names:
            val = data[name]["metrics"][metric]
            fig.add_trace(go.Scatter3d(
                x=[m_idx],
                y=[list(data.keys()).index(name)],
                z=[val],
                mode="markers+text",
                marker=dict(size=12, color=COLORS[name]),
                text=[f"{val:.2f}"],
                textposition="top center",
                name=name,
                showlegend=(m_idx == 0)
            ))

    # Add 3D bars manually using lines
    for m_idx, metric in enumerate(metrics):
        for n_idx, name in enumerate(model_names):
            val = data[name]["metrics"][metric]
            fig.add_trace(go.Scatter3d(
                x=[m_idx, m_idx],
                y=[n_idx, n_idx],
                z=[0, val],
                mode="lines",
                line=dict(color=COLORS[name], width=8),
                showlegend=False
            ))

    fig.update_layout(
        title=dict(
            text="3D Metrics Comparison — All Models",
            font=dict(size=22, color="white")
        ),
        scene=dict(
            xaxis=dict(
                title="Metric",
                tickvals=list(range(len(metrics))),
                ticktext=metrics,
                backgroundcolor="#0a0a0a",
                gridcolor="#333",
                color="white"
            ),
            yaxis=dict(
                title="Model",
                tickvals=list(range(len(model_names))),
                ticktext=model_names,
                backgroundcolor="#0a0a0a",
                gridcolor="#333",
                color="white"
            ),
            zaxis=dict(
                title="Score",
                range=[0, 1.1],
                backgroundcolor="#0a0a0a",
                gridcolor="#333",
                color="white"
            ),
            bgcolor="#0a0a0a"
        ),
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333", borderwidth=1),
        height=700
    )

    out = "visualizations/output/3d_metrics.html"
    fig.write_html(out)
    print(f"Saved: {out}")
    fig.show()

def plot_3d_tsne():
    df = pd.read_csv("data/features/features.csv")
    feature_cols = [c for c in df.columns if c not in ["star_name", "label", "mission"]]
    X = df[feature_cols].values
    y = df["label"].values
    missions = df["mission"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tsne = TSNE(n_components=3, random_state=42, perplexity=min(10, len(X)-1))
    X_3d = tsne.fit_transform(X_scaled)

    colors = ["#A855F7" if label == 1 else "#F97316" for label in y]
    symbols = {"Kepler": "circle", "TESS": "diamond", "K2": "square"}

    fig = go.Figure()

    for mission in ["Kepler", "TESS", "K2"]:
        mask = missions == mission
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0],
            y=X_3d[mask, 1],
            z=X_3d[mask, 2],
            mode="markers",
            marker=dict(
                size=8,
                color=["#A855F7" if label == 1 else "#F97316" for label in y[mask]],
                symbol=symbols[mission],
                line=dict(width=1, color="white")
            ),
            text=[f"{s} ({'Planet' if l==1 else 'No Planet'})"
                  for s, l in zip(df["star_name"].values[mask], y[mask])],
            hoverinfo="text",
            name=mission
        ))

    fig.update_layout(
        title=dict(
            text="3D t-SNE — Feature Space (Purple=Planet, Orange=No Planet)",
            font=dict(size=20, color="white")
        ),
        scene=dict(
            xaxis=dict(title="t-SNE 1", backgroundcolor="#0a0a0a", gridcolor="#333", color="white"),
            yaxis=dict(title="t-SNE 2", backgroundcolor="#0a0a0a", gridcolor="#333", color="white"),
            zaxis=dict(title="t-SNE 3", backgroundcolor="#0a0a0a", gridcolor="#333", color="white"),
            bgcolor="#0a0a0a"
        ),
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333", borderwidth=1),
        height=700
    )

    out = "visualizations/output/3d_tsne.html"
    fig.write_html(out)
    print(f"Saved: {out}")
    fig.show()

if __name__ == "__main__":
    print("Generating 3D ROC curves...")
    plot_3d_roc()
    print("\nGenerating 3D metrics comparison...")
    plot_3d_metrics()
    print("\nGenerating 3D t-SNE...")
    plot_3d_tsne()