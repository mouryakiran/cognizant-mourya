"""Dataset handling for EnergyIQ.

* ``load_dataset``     - loads whichever dataset is available (real, generated
                         or the real sample) and validates the schema.
* ``generate_dataset`` - produces a full synthetic dataset that follows the exact
                         same schema as the real pre-processed PJM energy data,
                         so the entire platform works end-to-end even when the
                         (incomplete) real download is not available.

Both paths produce a tidy long-format frame with one row per region per hour.
"""

from __future__ import annotations

import logging
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger("energyiq.data")

SCHEMA_COLUMNS = [
    "datetime", "region", "consumption_mw",
    "anomaly_iqr", "hour", "dow",
    "seasonal_zscore", "anomaly_seasonal_zscore", "is_anomaly",
    "day", "month", "year", "dayofweek", "dayofyear", "weekofyear",
    "is_weekend", "quarter",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_holiday",
    "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_48h", "lag_168h",
    "rolling_mean_3h", "rolling_std_3h", "rolling_max_3h", "rolling_min_3h",
    "rolling_mean_6h", "rolling_std_6h", "rolling_max_6h", "rolling_min_6h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_max_24h", "rolling_min_24h",
    "rolling_mean_168h", "rolling_std_168h",
]


# --------------------------------------------------------------------------- #
# Region profiles - base load, seasonal strength, weekday/weekend shape, noise
# --------------------------------------------------------------------------- #
def _region_profiles() -> dict:
    return {
        "AEP":    dict(base=12300, summer=0.28, winter=0.12, weekend=0.88, noise=0.045),
        "COMED":  dict(base=10300, summer=0.26, winter=0.16, weekend=0.90, noise=0.050),
        "DAYTON": dict(base=5800,  summer=0.30, winter=0.14, weekend=0.86, noise=0.055),
        "DEOK":   dict(base=7200,  summer=0.27, winter=0.13, weekend=0.87, noise=0.052),
        "DOM":    dict(base=9800,  summer=0.25, winter=0.20, weekend=0.91, noise=0.048),
        "DUQ":    dict(base=4400,  summer=0.22, winter=0.26, weekend=0.89, noise=0.060),
        "EKPC":   dict(base=5200,  summer=0.30, winter=0.15, weekend=0.85, noise=0.056),
        "FE":     dict(base=8600,  summer=0.26, winter=0.18, weekend=0.88, noise=0.050),
        "NI":     dict(base=4700,  summer=0.25, winter=0.15, weekend=0.90, noise=0.055),
        "OKGE":   dict(base=5900,  summer=0.32, winter=0.10, weekend=0.87, noise=0.060),
        "PJME":   dict(base=12600, summer=0.24, winter=0.22, weekend=0.89, noise=0.042),
        "PJMW":   dict(base=7600,  summer=0.28, winter=0.14, weekend=0.87, noise=0.050),
    }


# --------------------------------------------------------------------------- #
# Hourly shape - typical commercial/residential demand curve
# --------------------------------------------------------------------------- #
def _hour_shape(h: int) -> float:
    """Multiplier by hour of day (0-23). Trough overnight, peak early evening."""
    shape = {
        0: 0.58, 1: 0.55, 2: 0.53, 3: 0.52, 4: 0.54, 5: 0.62,
        6: 0.72, 7: 0.82, 8: 0.88, 9: 0.90, 10: 0.90, 11: 0.90,
        12: 0.88, 13: 0.86, 14: 0.87, 15: 0.90, 16: 0.94, 17: 0.98,
        18: 1.00, 19: 0.98, 20: 0.92, 21: 0.84, 22: 0.75, 23: 0.66,
    }
    return shape.get(h, 0.8)


def _is_us_holiday_ts(ts: pd.Timestamp, holidays: dict) -> bool:
    for (m, d), _name in holidays.items():
        if ts.month == m and ts.day == d:
            return True
    return False


def _is_holiday_ts(ts: pd.Timestamp) -> bool:
    holidays = {tuple(k): v for k, v in config.HOLIDAYS.items()}
    return _is_us_holiday_ts(ts, holidays)


def _seasonality(doy: float, summer: float, winter: float, base: float) -> float:
    """Combined annual (heating+cooling) envelope."""
    cooling = 0.5 * (1 + math.sin(2 * math.pi * (doy / 365.0 - 0.55)))
    heating = 0.5 * (1 + math.sin(2 * math.pi * (doy / 365.0 - 0.15)))
    return base * (1.0 + cooling * summer + heating * winter)


def _build_region_series(region: str, start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Generate one region's hourly consumption series (before derived features)."""
    rng = pd.date_range(start=start, end=end, freq="h")
    n = len(rng)
    r = _region_profiles()[region]
    rng_local = np.random.default_rng(seed)

    doy = rng.dayofyear.to_numpy(dtype=float)
    h = rng.hour.to_numpy()
    dow = rng.dayofweek.to_numpy()

    seasonal = np.array([_seasonality(d, r["summer"], r["winter"], 1.0) for d in doy])
    diurnal = np.array([_hour_shape(x) for x in h])
    weekly = np.where(dow >= 5, r["weekend"], 1.0)

    # Holiday multiplier
    holiday = np.array([0.78 if _is_holiday_ts(x) else 1.0 for x in rng])

    # Slowly varying weather noise with AR(1) persistence + small jumps
    ar = rng_local.normal(0.0, 1.0, n)
    alpha = 0.90
    for i in range(1, n):
        ar[i] = ar[i - 1] * alpha + ar[i] * math.sqrt(1 - alpha**2)
    weather = 0.020 * ar / 3.0

    cons = r["base"] * seasonal * diurnal * weekly * holiday
    cons *= (1.0 + weather + rng_local.normal(0.0, r["noise"], n))

    # Inject a handful of outliers (demand spikes / plant trips) so the anomaly
    # detection component has genuinely anomalous points to find.
    spike_idx = rng_local.choice(n, size=max(5, int(n * 0.0004)), replace=False)
    cons[spike_idx] *= rng_local.uniform(1.25, 2.2, size=len(spike_idx))

    return pd.DataFrame({"datetime": rng, "region": region, "consumption_mw": cons})


# --------------------------------------------------------------------------- #
# Feature engineering - reproduces the exact columns of the real dataset
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotently compute the full feature set on a (region,datetime) frame.

    Expects columns ``datetime``, ``region``, ``consumption_mw``. Returns a frame
    with the exact schema of the real pre-processed dataset.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["region", "datetime"]).reset_index(drop=True)
    if "rolling_mean_168h" not in df.columns:
        df = _add_lags_and_rolling(df)

    ts = df["datetime"]
    df["hour"] = ts.dt.hour.astype(int)
    df["dow"] = ts.dt.dayofweek.astype(int)              # Monday=0
    df["day"] = ts.dt.day.astype(int)
    df["month"] = ts.dt.month.astype(int)
    df["year"] = ts.dt.year.astype(int)
    df["dayofweek"] = ts.dt.dayofweek.astype(int)
    df["dayofyear"] = ts.dt.dayofyear.astype(int)
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["quarter"] = ts.dt.quarter.astype(int)
    df["is_holiday"] = ts.map(_is_holiday_ts).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    # Periodic anomaly features on the raw series (kept simple, per region)
    df["anomaly_iqr"] = _iqr_anomaly(df)
    z = (df["consumption_mw"] - df["rolling_mean_168h"]) / df["rolling_std_168h"].replace(0, np.nan)
    df["seasonal_zscore"] = z.fillna(0.0)
    df["anomaly_seasonal_zscore"] = (df["seasonal_zscore"].abs() > config.ANOMALY["zscore_threshold"]).astype(int)
    df["is_anomaly"] = ((df["anomaly_iqr"] == 1) | (df["anomaly_seasonal_zscore"] == 1)).astype(int)

    df = df[SCHEMA_COLUMNS].copy()
    return df.reset_index(drop=True)


def _iqr_anomaly(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("region")["consumption_mw"]
    q1, q3 = g.transform("quantile", q=0.25), g.transform("quantile", q=0.75)
    iqr = q3 - q1
    lo = q1 - config.ANOMALY["iqr_factor"] * iqr
    hi = q3 + config.ANOMALY["iqr_factor"] * iqr
    return ((df["consumption_mw"] < lo) | (df["consumption_mw"] > hi)).astype(int)


def _add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """lag_1h ... lag_168h and rolling stats (3h/6h/24h/168h)."""
    g = df["consumption_mw"]
    for lag in (1, 2, 3, 24, 48, 168):
        df[f"lag_{lag}h"] = g.groupby(df["region"]).shift(lag)
    for win in (3, 6, 24, 168):
        rolled = g.groupby(df["region"]).rolling(win, min_periods=1)
        df[f"rolling_mean_{win}h"] = rolled.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{win}h"] = rolled.std().reset_index(level=0, drop=True)
        df[f"rolling_max_{win}h"] = rolled.max().reset_index(level=0, drop=True)
        df[f"rolling_min_{win}h"] = rolled.min().reset_index(level=0, drop=True)
    return df


def build_features_with_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: add lags/rolling first, then derive anomaly features."""
    out = _add_lags_and_rolling(df.copy())
    return build_features(out)


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def generate_dataset(
    start: str = "2005-01-01",
    end: str = "2016-12-31",
    regions: list[str] | None = None,
    out_path: str | Path | None = None,
    seed: int = 7,
    progress: bool = True,
) -> Path:
    """Generate a realistic full dataset in the exact pre-processed schema.

    Writes one row per region per hour to ``out_path`` (CSV). Returns the path.
    """
    out_path = Path(out_path or config.GENERATED_DATASET_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regions = regions or config.REGIONS

    wrote_header = False
    for i, region in enumerate(regions):
        raw = _build_region_series(region, start, end, seed=seed + i)
        feats = build_features_with_lags(raw)
        feats = feats[~feats["lag_168h"].isna()].reset_index(drop=True)
        header_row = None if wrote_header else SCHEMA_COLUMNS
        feats.to_csv(out_path, mode="a", index=False, header=header_row)
        wrote_header = True
        if progress:
            log.info("region %s -> %d rows written", region, len(feats))
            print(f"Region {region}: {len(feats)} rows written.")
    log.info("dataset generated at %s (%d regions)", out_path, len(regions))
    return out_path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df = df[SCHEMA_COLUMNS].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    num_cols = [c for c in SCHEMA_COLUMNS if c not in ("datetime", "region")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True)


def _candidate_paths() -> list[Path]:
    paths = [config.PROCESSED_DATASET_PATH, config.GENERATED_DATASET_PATH]
    if config.SAMPLE_DATASET_PATH.exists():
        paths.append(config.SAMPLE_DATASET_PATH)
    return paths


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate a dataset.

    ``path`` is used if given; otherwise the first available of the real dataset,
    the generated dataset, or the real sample is used. Raises FileNotFoundError
    when nothing is available.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        log.info("loading dataset from %s", p)
        df = pd.read_csv(p)
        return _validate(df)

    for cand in _candidate_paths():
        if cand.exists():
            try:
                log.info("loading dataset from %s", cand)
                return _validate(pd.read_csv(cand))
            except (ValueError, pd.errors.ParserError) as e:
                log.warning("skipping unreadable dataset %s: %s", cand, e)

    raise FileNotFoundError(
        "No dataset found. Run `python scripts/generate_data.py` to create one, "
        "or place your pre-processed CSV at data/processed/processed_energy_data.csv"
    )


def ensure_dataset(force_generate: bool = False) -> pd.DataFrame:
    """Ensure a usable dataset exists, generating one if needed."""
    for cand in _candidate_paths():
        if cand.exists() and not force_generate:
            return load_dataset(cand)
    log.warning("No dataset available - generating a full synthetic dataset.")
    generate_dataset()
    return load_dataset()