# QuantumKepler 🪐

A quantum-classical hybrid pipeline for exoplanet detection using real NASA light curve data.

🚀 **Live Demo:** https://quantumkepler-dk845.streamlit.app/

## Core Question
Do quantum kernel methods show measurable advantage over classical ML for exoplanet transit classification?

## Data Sources
- NASA Kepler Mission
- TESS Mission
- K2 Mission
- 37 stars total (21 planet hosts, 16 non-planet)

## Stack
- PennyLane — quantum circuits
- Lightkurve — NASA data
- scikit-learn — classical ML
- MLflow — experiment tracking
- Streamlit — interactive web app

## Project Structure
- `data/` — data fetching and feature extraction
- `models/classical/` — Random Forest, CNN
- `models/quantum/` — Quantum Kernel SVM, VQC
- `analysis/` — model comparison
- `visualizations/` — all plots
- `app/` — Streamlit demo

## Live Application

Explore the project here:

👉 https://quantumkepler-dk845.streamlit.app/

## Results (Day 1)

| Metric | Random Forest | Quantum Kernel SVM |
|----------|----------|----------|
| Accuracy | 0.25 | 0.50 |
| F1 | 0.25 | 0.67 |
| ROC AUC | 0.27 | 0.47 |