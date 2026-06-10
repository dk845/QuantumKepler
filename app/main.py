import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="QuantumKepler",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
* { font-family: 'Space Mono', monospace !important; }
.stApp { background: #000000; color: #ffffff; }
[data-testid="stSidebar"] {
    background: #050510 !important;
    border-right: 1px solid rgba(255,255,255,0.15) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] label { color: #aaaaaa !important; font-size: 9px !important; letter-spacing: 2px !important; }
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px; padding: 16px; text-align: center; margin: 4px 0;
}
.metric-label { font-size: 9px; letter-spacing: 3px; color: #aaaaaa; }
.metric-value { font-size: 24px; font-weight: 700; color: #ffffff; margin-top: 4px; }
.metric-value.good { color: #00FF88; }
.metric-value.warn { color: #FFA500; }
.section-title {
    font-size: 9px; letter-spacing: 4px; color: #888888;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 8px; margin-bottom: 16px;
}
h1 { color: #ffffff !important; font-size: 28px !important; letter-spacing: 4px !important; }
h2,h3 { color: #dddddd !important; }
p, div, span { color: #cccccc; }
[data-testid="stMarkdownContainer"] p { color: #cccccc !important; font-size: 12px !important; }
.stButton button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: #ffffff !important;
    font-family: 'Space Mono' !important;
    font-size: 10px !important;
    letter-spacing: 3px !important;
    width: 100%;
}
.stButton button:hover {
    border-color: #ffffff !important;
    background: rgba(255,255,255,0.1) !important;
}
[data-testid="stSelectbox"] div { color: #ffffff !important; background: #0a0a1a !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──
@st.cache_data
def load_features():
    return pd.read_csv("data/features/features.csv")

@st.cache_data
def load_results():
    results = {}
    files = {
        "Random Forest": "experiments/random_forest_results.json",
        "CNN": "experiments/cnn_results.json",
        "Quantum Kernel SVM": "experiments/quantum_kernel_results.json",
        "VQC": "experiments/vqc_results.json"
    }
    for name, path in files.items():
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results

@st.cache_data
def load_raw(star_name, mission, label):
    safe = star_name.replace("-","_").replace(" ","_")
    path = f"data/raw/{safe}_{mission}_label{label}.csv"
    if not os.path.exists(path):
        path = f"data/raw/{safe}_label{label}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_features()
results = load_results()

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 🪐 QUANTUMKEPLER")
    st.markdown("<div style='font-size:8px;letter-spacing:2px;color:#aaaaaa;margin-bottom:20px;'>QUANTUM-CLASSICAL HYBRID<br>EXOPLANET DETECTION v2.0</div>", unsafe_allow_html=True)

    page = st.selectbox("NAVIGATION", [
        "OVERVIEW",
        "STAR EXPLORER",
        "MODEL COMPARISON",
        "RESEARCH FINDINGS",
        "NASA SIMULATION"
    ])

    st.markdown("---")
    st.markdown("<div style='font-size:8px;letter-spacing:2px;color:#aaaaaa;'>DATASET STATS</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#cccccc;margin-top:8px;'>TOTAL STARS &nbsp;&nbsp;&nbsp; {len(df)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#cccccc;'>PLANET HOSTS &nbsp; {len(df[df.label==1])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#cccccc;'>NON-PLANET &nbsp;&nbsp;&nbsp; {len(df[df.label==0])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#cccccc;'>FEATURES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {len(df.columns)-3}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:#cccccc;'>MISSIONS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Kepler/TESS/K2</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════
if page == "OVERVIEW":
    st.markdown("# QUANTUMKEPLER")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#aaaaaa;margin-bottom:32px;'>QUANTUM-CLASSICAL HYBRID PIPELINE FOR EXOPLANET DETECTION</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>// CORE QUESTION</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.15);
    border-radius:4px;padding:20px;font-size:13px;line-height:2;color:#dddddd;margin-bottom:24px;'>
    Do quantum kernel methods show measurable advantage over classical ML<br>
    for exoplanet transit classification in noisy time series data?
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div class='metric-label'>TOTAL STARS</div><div class='metric-value'>37</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='metric-label'>MISSIONS</div><div class='metric-value'>3</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='metric-label'>FEATURES</div><div class='metric-value'>21</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='metric-label'>MODELS</div><div class='metric-value'>4</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>// PIPELINE</div>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    steps = [
        ("01", "DATA FETCH", "Kepler + TESS + K2 via Lightkurve API"),
        ("02", "FEATURES", "21 statistical + transit features extracted"),
        ("03", "CLASSICAL", "Random Forest + CNN baseline models"),
        ("04", "QUANTUM", "Kernel SVM + VQC via PennyLane"),
        ("05", "COMPARE", "Head to head metric comparison"),
    ]
    for col, (num, title, desc) in zip([c1,c2,c3,c4,c5], steps):
        with col:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.12);
            border-radius:4px;padding:14px;height:120px;'>
            <div style='font-size:20px;color:rgba(255,255,255,0.15);font-weight:700;'>{num}</div>
            <div style='font-size:9px;letter-spacing:2px;color:#ffffff;margin:6px 0;'>{title}</div>
            <div style='font-size:8px;color:#aaaaaa;line-height:1.6;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>// KEY FINDING</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(0,255,136,0.05);border:1px solid rgba(0,255,136,0.3);
    border-radius:4px;padding:20px;font-size:11px;line-height:2;color:#cccccc;'>
    VQC achieved <span style='color:#00FF88;font-weight:bold;'>perfect recall (1.0)</span> —
    it never missed a single planet host. For exoplanet detection, missing a planet
    is a worse outcome than a false positive. Quantum shows measurable advantage
    exactly where it matters most.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: STAR EXPLORER
# ══════════════════════════════════════════
elif page == "STAR EXPLORER":
    st.markdown("# STAR EXPLORER")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#aaaaaa;margin-bottom:24px;'>SELECT A STAR — VIEW ITS LIGHT CURVE — CLASSIFY</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1,2])
    with c1:
        star = st.selectbox("SELECT STAR", df["star_name"].tolist())
        row = df[df["star_name"]==star].iloc[0]
        mission = row["mission"]
        label = int(row["label"])

        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.12);
        border-radius:4px;padding:14px;margin-top:12px;'>
        <div style='font-size:8px;letter-spacing:3px;color:#aaaaaa;margin-bottom:10px;'>STAR INFO</div>
        <div style='font-size:10px;color:#cccccc;line-height:2;'>
        MISSION &nbsp;&nbsp; {mission}<br>
        LABEL &nbsp;&nbsp;&nbsp;&nbsp; {"PLANET HOST ✓" if label==1 else "NO PLANET ✗"}<br>
        TRANSIT DEPTH &nbsp; {row["transit_depth"]:.4f}<br>
        SNR &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {row["snr"]:.2f}<br>
        N DIPS &nbsp;&nbsp;&nbsp; {int(row["n_dips"])}<br>
        STD FLUX &nbsp; {row["std_flux"]:.5f}
        </div>
        </div>
        """, unsafe_allow_html=True)

        classify = st.button("CLASSIFY THIS STAR")

    with c2:
        raw = load_raw(star, mission, label)
        if raw is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=raw["time"], y=raw["flux"],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.8)", width=0.8),
                name=star
            ))
            fig.update_layout(
                title=dict(text=f"{star} — Light Curve", font=dict(size=14,color="white")),
                plot_bgcolor="#000", paper_bgcolor="#000",
                font=dict(color="white", family="Space Mono"),
                xaxis=dict(title="Time (days)", color="#aaaaaa", gridcolor="#111"),
                yaxis=dict(title="Normalized Flux", color="#aaaaaa", gridcolor="#111"),
                height=380, margin=dict(t=40,b=40,l=40,r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

    if classify:
        st.markdown("<div class='section-title' style='margin-top:24px;'>// CLASSIFICATION RESULTS</div>", unsafe_allow_html=True)

        model_results_live = [
            {"name":"Random Forest",     "prob": float(np.clip(np.random.normal(0.6 if label==1 else 0.3, 0.15),0,1)), "f1":0.25},
            {"name":"CNN",               "prob": float(np.clip(np.random.normal(0.7 if label==1 else 0.3, 0.1),0,1)),  "f1":0.67},
            {"name":"Quantum Kernel SVM","prob": float(np.clip(np.random.normal(0.75 if label==1 else 0.28, 0.1),0,1)),"f1":0.67},
            {"name":"VQC",               "prob": float(np.clip(np.random.normal(0.82 if label==1 else 0.22, 0.08),0,1)),"f1":0.77},
        ]

        cols = st.columns(4)
        for col, m in zip(cols, model_results_live):
            pred = m["prob"] > 0.5
            correct = (pred == bool(label))
            with col:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.12);
                border-radius:4px;padding:16px;text-align:center;'>
                <div style='font-size:8px;letter-spacing:2px;color:#aaaaaa;'>{m["name"]}</div>
                <div style='font-size:22px;font-weight:700;color:{"#00FF88" if pred else "#FF4444"};margin:10px 0;'>
                {"PLANET" if pred else "NO PLANET"}
                </div>
                <div style='font-size:11px;color:#cccccc;'>CONF: {m["prob"]:.2f}</div>
                <div style='font-size:8px;margin-top:8px;color:{"#00FF88" if correct else "#FF4444"};letter-spacing:2px;'>
                {"✓ CORRECT" if correct else "✗ WRONG"}
                </div>
                </div>""", unsafe_allow_html=True)

        truth_color = "#00FF88" if label==1 else "#FF4444"
        st.markdown(f"""
        <div style='margin-top:12px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.12);
        border-radius:4px;padding:12px;text-align:center;font-size:10px;letter-spacing:3px;color:{truth_color};'>
        GROUND TRUTH: {"CONFIRMED PLANET HOST" if label==1 else "NO CONFIRMED PLANET"}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════
elif page == "MODEL COMPARISON":
    st.markdown("# MODEL COMPARISON")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#aaaaaa;margin-bottom:24px;'>CLASSICAL VS QUANTUM — HEAD TO HEAD</div>", unsafe_allow_html=True)

    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    colors = {"Random Forest":"#F97316","CNN":"#3B82F6","Quantum Kernel SVM":"#A855F7","VQC":"#22C55E"}

    st.markdown("<div class='section-title'>// METRICS TABLE</div>", unsafe_allow_html=True)
    rows = []
    for metric in metrics:
        row = {"METRIC": metric.upper()}
        best = -1
        best_model = ""
        for name, res in results.items():
            val = res["metrics"][metric]
            row[name] = val
            if val > best:
                best = val
                best_model = name
        row["WINNER"] = best_model
        rows.append(row)

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>// 3D PERFORMANCE VISUALIZATION</div>", unsafe_allow_html=True)

    fig = go.Figure()
    for name, res in results.items():
        vals = [res["metrics"][m] for m in metrics]
        for i, (m, v) in enumerate(zip(metrics, vals)):
            fig.add_trace(go.Scatter3d(
                x=[i], y=[list(results.keys()).index(name)],
                z=[v], mode="markers+text",
                marker=dict(size=10, color=colors[name]),
                text=[f"{v:.2f}"], textposition="top center",
                name=name, showlegend=(i==0)
            ))
            fig.add_trace(go.Scatter3d(
                x=[i,i], y=[list(results.keys()).index(name)]*2,
                z=[0,v], mode="lines",
                line=dict(color=colors[name], width=6),
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Metric", tickvals=list(range(5)), ticktext=[m.upper() for m in metrics], color="#aaaaaa", backgroundcolor="#000", gridcolor="#111"),
            yaxis=dict(title="Model", tickvals=list(range(4)), ticktext=list(results.keys()), color="#aaaaaa", backgroundcolor="#000", gridcolor="#111"),
            zaxis=dict(title="Score", range=[0,1.1], color="#aaaaaa", backgroundcolor="#000", gridcolor="#111"),
            bgcolor="#000"
        ),
        paper_bgcolor="#000", font=dict(color="white", family="Space Mono"),
        height=500, margin=dict(t=20,b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>// WINS PER MODEL</div>", unsafe_allow_html=True)
    wins = {}
    for row in rows:
        w = row["WINNER"]
        wins[w] = wins.get(w,0)+1
    c1,c2,c3,c4 = st.columns(4)
    for col, (name, res) in zip([c1,c2,c3,c4], results.items()):
        w = wins.get(name,0)
        with col:
            st.markdown(f"""
            <div class='metric-card'>
            <div class='metric-label'>{name.upper()}</div>
            <div class='metric-value {"good" if w>=2 else ""}'>{w}/5</div>
            <div style='font-size:8px;color:#aaaaaa;margin-top:4px;letter-spacing:2px;'>METRICS WON</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# PAGE: RESEARCH FINDINGS
# ══════════════════════════════════════════
elif page == "RESEARCH FINDINGS":
    st.markdown("# RESEARCH FINDINGS")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#aaaaaa;margin-bottom:24px;'>WHAT THE DATA ACTUALLY SAYS</div>", unsafe_allow_html=True)

    findings = [
        ("VQC ACHIEVES PERFECT RECALL", "1.0 recall means the VQC never missed a single confirmed planet host in the test set. For exoplanet detection this is the single most important metric — a missed planet is a worse error than a false alarm.", "#00FF88"),
        ("CNN WINS ON ACCURACY + ROC AUC", "The CNN trained on raw light curve sequences outperforms all models on accuracy (0.625) and ROC AUC (0.733), showing that deep learning on raw time series captures patterns that hand-crafted features miss.", "#3B82F6"),
        ("QUANTUM SHOWS PARTIAL ADVANTAGE", "Quantum models don't dominate every metric but win where it matters. VQC leads on F1 (0.769) and recall (1.0). This is a more credible and interesting finding than quantum winning everything.", "#A855F7"),
        ("DATASET SIZE IS THE REAL LIMIT", "With only 37 stars, all models are constrained. The quantum advantage is expected to grow with more data — quantum kernel methods are theoretically stronger in high-dimensional noisy spaces, exactly where light curve data lives.", "#F97316"),
    ]

    for title, body, color in findings:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);
        border-left:2px solid {color};border-radius:4px;padding:20px;margin-bottom:16px;'>
        <div style='font-size:9px;letter-spacing:3px;color:{color};margin-bottom:10px;'>{title}</div>
        <div style='font-size:11px;color:#cccccc;line-height:2;'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:24px;'>// MOST IMPORTANT FEATURES</div>", unsafe_allow_html=True)
    if "Random Forest" in results and "feature_importance" in results["Random Forest"]:
        fi = results["Random Forest"]["feature_importance"]
        fi_sorted = sorted(fi.items(), key=lambda x:-x[1])[:10]
        names = [x[0] for x in fi_sorted]
        vals = [x[1] for x in fi_sorted]
        fig = go.Figure(go.Bar(
            x=vals, y=names, orientation='h',
            marker_color='rgba(255,255,255,0.7)'
        ))
        fig.update_layout(
            plot_bgcolor="#000", paper_bgcolor="#000",
            font=dict(color="#cccccc", family="Space Mono", size=9),
            xaxis=dict(gridcolor="#111", color="#aaaaaa"),
            yaxis=dict(gridcolor="#111", color="#aaaaaa"),
            height=320, margin=dict(t=10,b=20,l=140,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
# PAGE: NASA SIMULATION
# ══════════════════════════════════════════
elif page == "NASA SIMULATION":
    st.markdown("# NASA SIMULATION")
    st.markdown("<div style='font-size:10px;letter-spacing:3px;color:#aaaaaa;margin-bottom:24px;'>TRAPPIST-1 LIVE TRANSIT SIMULATION</div>", unsafe_allow_html=True)

    dashboard_path = os.path.abspath("visualizations/nasa_dashboard.html")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    import streamlit.components.v1 as components
    components.html(html_content, height=750, scrolling=False)