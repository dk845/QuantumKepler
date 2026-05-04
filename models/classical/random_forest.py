import pandas as pd
import numpy as np
import yaml
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import json

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

FEATURES_PATH = config["paths"]["features"]
RF_CONFIG = config["classical"]["random_forest"]
MLFLOW_URI = config["mlflow"]["tracking_uri"]

def load_data():
    df = pd.read_csv(os.path.join(FEATURES_PATH, "features.csv"))
    feature_cols = [c for c in df.columns if c not in ["star_name", "label", "mission"]]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols, df

def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4)
    }

def train():
    X, y, feature_cols, df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=RF_CONFIG["n_estimators"],
        max_depth=RF_CONFIG["max_depth"],
        random_state=RF_CONFIG["random_state"]
    )

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="random_forest"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_prob)

        mlflow.log_params(RF_CONFIG)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("\n--- Random Forest Results ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Save results for comparison later
        os.makedirs("experiments", exist_ok=True)
        results = {
            "model": "random_forest",
            "metrics": metrics,
            "feature_importance": dict(zip(feature_cols, model.feature_importances_.tolist())),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()
        }
        with open("experiments/random_forest_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nResults saved to experiments/random_forest_results.json")
        return model, metrics

if __name__ == "__main__":
    train()