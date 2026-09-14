"""Generate realistic synthetic hourly energy consumption data (vectorized)."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

REGIONS = ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load"]
REGION_BASE = {
    "AEP": 17000, "COMED": 12000, "DAYTON": 3000, "DEOK": 2500,
    "DOM": 9000, "DUQ": 2200, "EKPC": 2000, "FE": 4500,
    "NI": 3500, "PJME": 28000, "PJMW": 8000, "PJM_Load": 32000,
}

dates = pd.date_range("2015-01-01", "2024-12-31 23:00:00", freq="h")
print(f"Generating {len(dates)} base hours for {len(REGIONS)} regions...")

hour = dates.hour.to_numpy()
month = dates.month.to_numpy()
dow = dates.dayofweek.to_numpy()

daily = (
    1.0
    + 0.15 * np.exp(-0.5 * ((hour - 9) / 2) ** 2)
    + 0.20 * np.exp(-0.5 * ((hour - 18) / 2.5) ** 2)
    - 0.25 * np.exp(-0.5 * ((hour - 4) / 1.5) ** 2)
)
seasonal = 1.0 + 0.12 * np.abs(np.sin(2 * np.pi * (month - 1) / 12)) ** 1.5
weekend_factor = np.where(dow >= 5, 0.88, 1.0)

noise = np.random.normal(1.0, 0.03, size=len(dates))
anomaly_mask = np.random.random(len(dates)) < 0.008
anomaly_factor = np.random.choice(
    [0.3, 0.5, 1.6, 2.0, 2.5], size=len(dates), p=[0.2, 0.2, 0.2, 0.2, 0.2]
)

frames = []
for region in REGIONS:
    base = REGION_BASE[region]
    consumption = base * daily * seasonal * weekend_factor * noise
    consumption[anomaly_mask] *= anomaly_factor[anomaly_mask]
    consumption = np.maximum(consumption, 100.0)

    frames.append(pd.DataFrame({
        "Datetime": dates,
        "Consumption": np.round(consumption, 2),
        "Region": region,
    }))

df = pd.concat(frames, ignore_index=True)
df = df.sort_values(["Datetime", "Region"]).reset_index(drop=True)

df["Year"] = df["Datetime"].dt.year
df["Month"] = df["Datetime"].dt.month
df["Day"] = df["Datetime"].dt.day
df["Hour"] = df["Datetime"].dt.hour
df["Weekday"] = df["Datetime"].dt.day_name()
df["Weekend"] = (df["Datetime"].dt.dayofweek >= 5)
df["Peak_Hour"] = (df["Datetime"].dt.hour >= 17) & (df["Datetime"].dt.hour <= 21)

out_path = Path(__file__).resolve().parent / "data" / "processed" / "processed_energy_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df.head())