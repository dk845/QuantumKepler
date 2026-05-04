import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["paths"]["raw_data"]
FEATURES_PATH = config["paths"]["features"]

def extract_features(flux: np.ndarray) -> dict:
    """
    Extract statistical and transit features from a light curve flux array.
    """
    flux = np.array(flux)

    # Basic stats
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    median_flux = np.median(flux)
    min_flux = np.min(flux)
    max_flux = np.max(flux)
    flux_range = max_flux - min_flux

    # Skewness and kurtosis
    skewness = float(pd.Series(flux).skew())
    kurtosis = float(pd.Series(flux).kurtosis())

    # Transit depth (how deep is the dip)
    transit_depth = mean_flux - min_flux

    # Fraction of points below mean (proxy for transit duration)
    below_mean = np.sum(flux < mean_flux) / len(flux)

    # RMS (root mean square — noise level)
    rms = np.sqrt(np.mean(flux**2))

    # Peak to peak
    peak_to_peak = max_flux - min_flux

    # 10th and 90th percentile
    p10 = np.percentile(flux, 10)
    p90 = np.percentile(flux, 90)
    percentile_range = p90 - p10

    # Autocorrelation at lag 1 (periodicity signal)
    if len(flux) > 1:
        autocorr = float(pd.Series(flux).autocorr(lag=1))
    else:
        autocorr = 0.0

    # Number of dips (points more than 2 std below mean)
    n_dips = int(np.sum(flux < mean_flux - 2 * std_flux))

    # Signal to noise ratio
    snr = mean_flux / std_flux if std_flux != 0 else 0.0

    return {
        "mean_flux": mean_flux,
        "std_flux": std_flux,
        "median_flux": median_flux,
        "min_flux": min_flux,
        "max_flux": max_flux,
        "flux_range": flux_range,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "transit_depth": transit_depth,
        "below_mean_fraction": below_mean,
        "rms": rms,
        "peak_to_peak": peak_to_peak,
        "percentile_10": p10,
        "percentile_90": p90,
        "percentile_range": percentile_range,
        "autocorr_lag1": autocorr,
        "n_dips": n_dips,
        "snr": snr
    }

def build_feature_matrix():
    os.makedirs(FEATURES_PATH, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(os.path.join(RAW_PATH, "metadata.csv"))
    print(f"Processing {len(meta)} stars...\n")

    rows = []

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        star = row["star_name"]
        label = row["label"]
        mission = row["mission"]

        safe_name = star.replace("-", "_").replace(" ", "_")
        fname = os.path.join(RAW_PATH, f"{safe_name}_{mission}_label{label}.csv")

        # Handle old Kepler-only files too
        if not os.path.exists(fname):
            fname = os.path.join(RAW_PATH, f"{safe_name}_label{label}.csv")

        if not os.path.exists(fname):
            print(f"  File not found: {fname}")
            continue

        df = pd.read_csv(fname)

        if "flux" not in df.columns or len(df) < 10:
            print(f"  Skipping {star} — not enough data")
            continue

        features = extract_features(df["flux"].values)
        features["star_name"] = star
        features["label"] = label
        features["mission"] = mission
        rows.append(features)

    feature_df = pd.DataFrame(rows)

    # Save
    out_path = os.path.join(FEATURES_PATH, "features.csv")
    feature_df.to_csv(out_path, index=False)

    print(f"\nDone! Feature matrix shape: {feature_df.shape}")
    print(f"Saved to: {out_path}")
    print(feature_df.head())

if __name__ == "__main__":
    build_feature_matrix()