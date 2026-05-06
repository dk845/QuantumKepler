import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

RAW_PATH = "data/raw"

def plot_light_curve(star_name, mission, label):
    safe_name = star_name.replace("-", "_").replace(" ", "_")
    fname = os.path.join(RAW_PATH, f"{safe_name}_{mission}_label{label}.csv")

    if not os.path.exists(fname):
        print(f"File not found: {fname}")
        return

    df = pd.read_csv(fname)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["flux"],
        mode="lines",
        line=dict(color="#A855F7", width=0.8),
        name=star_name
    ))

    fig.update_layout(
        title=dict(
            text=f"🪐 {star_name} — Light Curve ({mission})",
            font=dict(size=22, color="white")
        ),
        xaxis=dict(title="Time (days)", color="white", gridcolor="#222"),
        yaxis=dict(title="Normalized Flux", color="white", gridcolor="#222"),
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        hovermode="x unified"
    )

    os.makedirs("visualizations/output", exist_ok=True)
    out = f"visualizations/output/{safe_name}_lightcurve.html"
    fig.write_html(out)
    print(f"Saved: {out}")
    fig.show()


def plot_planet_vs_no_planet():
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Kepler-22 — Confirmed Planet Host ✅", "KIC 3733735 — No Planet ❌"),
        vertical_spacing=0.12
    )

    # Planet host — first 500 points
    df1 = pd.read_csv(os.path.join(RAW_PATH, "Kepler_22_Kepler_label1.csv"))
    df1 = df1.head(500)
    df1["time"] = df1["time"] - df1["time"].min()

    # No planet — first 500 points
    df2 = pd.read_csv(os.path.join(RAW_PATH, "KIC_3733735_Kepler_label0.csv"))
    df2 = df2.head(500)
    df2["time"] = df2["time"] - df2["time"].min()

    fig.add_trace(go.Scatter(
        x=df1["time"], y=df1["flux"],
        mode="lines",
        line=dict(color="#A855F7", width=1),
        name="Kepler-22 (Planet ✅)"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df2["time"], y=df2["flux"],
        mode="lines",
        line=dict(color="#F97316", width=1),
        name="KIC 3733735 (No Planet ❌)"
    ), row=2, col=1)

    fig.update_layout(
        title=dict(
            text="🪐 Planet vs No Planet — Light Curve Comparison",
            font=dict(size=22, color="white")
        ),
        plot_bgcolor="#0a0a0a",
        paper_bgcolor="#0a0a0a",
        font=dict(color="white"),
        hovermode="x unified",
        showlegend=True,
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333", borderwidth=1),
        height=700
    )

    fig.update_xaxes(
        title_text="Time (days, normalized)",
        color="white",
        gridcolor="#222"
    )
    fig.update_yaxes(
        title_text="Normalized Flux",
        color="white",
        gridcolor="#222"
    )

    os.makedirs("visualizations/output", exist_ok=True)
    out = "visualizations/output/planet_vs_no_planet.html"
    fig.write_html(out)
    print(f"Saved: {out}")
    fig.show()


if __name__ == "__main__":
    plot_light_curve("Kepler-22", "Kepler", 1)
    plot_planet_vs_no_planet()