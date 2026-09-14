"""Central configuration for the EnergyIQ platform.

All paths are resolved relative to the project root (the directory that
contains the ``energyiq`` package and the ``api`` / ``web`` / ``data`` folders).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.environ.get("ENERGYIQ_DATA_DIR", ROOT / "data")
RAW_DIR = Path(os.environ.get("ENERGYIQ_RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = Path(os.environ.get("ENERGYIQ_PROCESSED_DIR", DATA_DIR / "processed"))
MODELS_DIR = Path(os.environ.get("ENERGYIQ_MODELS_DIR", ROOT / "models"))

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
REGIONS = [
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
    "EKPC", "FE", "NI", "OKGE", "PJME", "PJMW",
]

# The full pre-processed dataset if the user provides it.
PROCESSED_DATASET_PATH = Path(
    os.environ.get("ENERGYIQ_DATASET", PROCESSED_DIR / "processed_energy_data.csv")
)
# Generated fallback dataset (same schema) produced by scripts/generate_data.py
GENERATED_DATASET_PATH = PROCESSED_DIR / "generated_energy_data.csv"
# Real but truncated sample recovered from the interrupted download.
SAMPLE_DATASET_PATH = PROCESSED_DIR / "sample_real_processed.csv"

FORECAST_HORIZON = int(os.environ.get("ENERGYIQ_HORIZON", 48))  # hours

# --------------------------------------------------------------------------- #
# Electricity tariff  (time-of-use, $ / MWh)
# Baseline rates for the three standard periods used by commercial facilities.
# --------------------------------------------------------------------------- #
TARIFF = {
    "peak": float(os.environ.get("ENERGYIQ_PEAK_RATE", 95.0)),    # $/MWh
    "offpeak": float(os.environ.get("ENERGYIQ_OFFPEAK_RATE", 45.0)),
    "shoulder": float(os.environ.get("ENERGYIQ_SHOULDER_RATE", 70.0)),
    "peak_demand_charge": float(os.environ.get("ENERGYIQ_DEMAND_CHARGE", 12.0)),
}

# Peak windows (hour-of-week index = dow*24 + hour, Monday = 0)
PEAK_DOWS = (0, 1, 2, 3, 4)  # Mon-Fri
PEAK_HOURS = list(range(15, 21))  # 15:00 - 20:59
SHOULDER_HOURS = list(range(7, 15)) + list(range(21, 23))

# --------------------------------------------------------------------------- #
# Optimization settings
# --------------------------------------------------------------------------- #
OPTIMIZATION = {
    "battery_capacity_mwh": float(os.environ.get("ENERGYIQ_BATT_MWH", 40.0)),
    "battery_max_charge_mw": float(os.environ.get("ENERGYIQ_BATT_C_MW", 12.0)),
    "battery_max_discharge_mw": float(os.environ.get("ENERGYIQ_BATT_D_MW", 12.0)),
    "battery_efficiency": float(os.environ.get("ENERGYIQ_BATT_EFF", 0.92)),
    "initial_soc": float(os.environ.get("ENERGYIQ_BATT_SOC", 0.10)),
    "peak_reduction_frac": float(os.environ.get("ENERGYIQ_PEAK_FRAC", 0.10)),
    "shiftable_capacity_mw": float(os.environ.get("ENERGYIQ_SHIFT_MW", 5.0)),
}

# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(os.environ.get("ENERGYIQ_MODEL", MODELS_DIR / "forecast_xgb.joblib"))
SCALER_PATH = Path(os.environ.get("ENERGYIQ_SCALER", MODELS_DIR / "scaler_y.joblib"))
META_PATH = Path(os.environ.get("ENERGYIQ_META", MODELS_DIR / "model_meta.json"))

FEATURE_COLS = [
    "region",
    "anomaly_iqr",
    "hour", "dow", "day", "month", "year",
    "dayofweek", "dayofyear", "weekofyear", "is_weekend", "quarter",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_holiday",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_mean_3h", "rolling_std_3h", "rolling_max_3h", "rolling_min_3h",
    "rolling_mean_6h", "rolling_std_6h", "rolling_max_6h", "rolling_min_6h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_max_24h", "rolling_min_24h",
    "rolling_mean_168h", "rolling_std_168h",
]

TARGET = "consumption_mw"

CATEGORICAL_COLS = ["region"]

# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #
ANOMALY = {
    "zscore_threshold": float(os.environ.get("ENERGYIQ_Z_THRESH", 2.5)),
    "iqr_factor": float(os.environ.get("ENERGYIQ_IQR_FACTOR", 1.5)),
}

# --------------------------------------------------------------------------- #
# US holidays (used by feature engineering). A fixed set of common holidays.
# --------------------------------------------------------------------------- #
HOLIDAYS = {
    (1, 1): "New Year",
    (7, 4): "Independence",
    (12, 25): "Christmas",
    (12, 24): "Christmas Eve",
    (11, 11): "Veterans",
    (2, 14): "Valentines",
    (10, 31): "Halloween",
}

CREATED_BY = "energyiq"