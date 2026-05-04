import lightkurve as lk
import pandas as pd
import os
import yaml
from tqdm import tqdm

with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_PATH = config["paths"]["raw_data"]

# Label 1 = confirmed planet host
PLANET_HOSTS = {
    "Kepler": [
        "Kepler-7", "Kepler-10", "Kepler-11", "Kepler-16",
        "Kepler-22", "Kepler-62", "Kepler-186", "Kepler-442",
        "Kepler-452", "Kepler-1649"
    ],
    "TESS": [
        "TOI-700", "TOI-1338", "TOI-2257",
        "LHS 3844", "GJ 357", "HD 21749"
    ],
    "K2": [
        "K2-18", "K2-55", "K2-141",
        "K2-229", "TRAPPIST-1"
    ]
}

# Label 0 = no confirmed planet
NON_PLANET_HOSTS = {
    "Kepler": [
        "KIC 3733735", "KIC 4914423", "KIC 5955122",
        "KIC 6289650", "KIC 7103006", "KIC 8077137",
        "KIC 9410930"
    ],
    "TESS": [
        "TIC 167602025", "TIC 259962054",
        "TIC 394050135", "TIC 441798995"
    ],
    "K2": [
        "EPIC 201170410", "EPIC 201498078",
        "EPIC 202059229", "EPIC 211305568",
        "EPIC 212297394"
    ]
}

def fetch_light_curve(star_name, label, mission):
    try:
        search = lk.search_lightcurve(star_name, mission=mission)
        if len(search) == 0:
            print(f"  No data found: {star_name} [{mission}]")
            return None
        lc = search[0].download()
        lc = lc.normalize().remove_nans().remove_outliers()
        return {
            "star_name": star_name,
            "label": label,
            "mission": mission,
            "time": lc.time.value.tolist(),
            "flux": lc.flux.value.tolist(),
            "flux_err": lc.flux_err.value.tolist()
        }
    except Exception as e:
        print(f"  Error: {star_name} [{mission}]: {e}")
        return None

def fetch_group(star_dict, label, group_name):
    records = []
    print(f"\nFetching {group_name} stars (label={label})...")
    for mission, stars in star_dict.items():
        print(f"  Mission: {mission}")
        for star in tqdm(stars):
            result = fetch_light_curve(star, label=label, mission=mission)
            if result:
                records.append(result)
                df = pd.DataFrame({
                    "time": result["time"],
                    "flux": result["flux"],
                    "flux_err": result["flux_err"]
                })
                safe_name = star.replace("-", "_").replace(" ", "_")
                fname = os.path.join(RAW_PATH, f"{safe_name}_{mission}_label{label}.csv")
                df.to_csv(fname, index=False)
                print(f"    Saved: {fname}")
    return records

def fetch_all():
    os.makedirs(RAW_PATH, exist_ok=True)

    planet_records = fetch_group(PLANET_HOSTS, label=1, group_name="Planet host")
    no_planet_records = fetch_group(NON_PLANET_HOSTS, label=0, group_name="Non-planet")

    all_records = planet_records + no_planet_records

    meta = pd.DataFrame([{
        "star_name": r["star_name"],
        "label": r["label"],
        "mission": r["mission"],
        "n_points": len(r["time"])
    } for r in all_records])

    meta.to_csv(os.path.join(RAW_PATH, "metadata.csv"), index=False)

    print(f"\n--- SUMMARY ---")
    print(f"Planet hosts   (label=1): {len(planet_records)}")
    print(f"Non-planet     (label=0): {len(no_planet_records)}")
    print(f"Total stars             : {len(all_records)}")
    print(meta)

if __name__ == "__main__":
    fetch_all()