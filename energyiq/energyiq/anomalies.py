"""Anomaly detection for energy consumption.

Detects statistically unusual demand on both an absolute basis (IQR rule) and a
seasonally-adjusted basis (rolling z-score against the 168h seasonal baseline).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class Anomaly:
    timestamp: str
    region: str
    consumption_mw: float
    seasonal_zscore: float
    iqr_outlier: bool
    magnitude_pct: float
    reason: str = field(default="")
    baseline_mw: float = field(default=0.0)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_anomalies(
    df: pd.DataFrame,
    region: str | None = None,
    start: str | None = None,
    end: str | None = None,
    z_threshold: float | None = None,
) -> list[dict]:
    """Return anomalies in the dataset grid.

    Args:
        df: pre-processed dataset (must contain datetime/region/consumption_mw
            and ideally the pre-computed seasonal_zscore / anomaly flags).
        region: filter by region, otherwise all regions.
        start/end: ISO datetime window filters (inclusive).
    """
    if df.empty:
        return []

    data = df.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    if region is not None:
        data = data[data["region"] == region]
    if start is not None:
        data = data[data["datetime"] >= pd.Timestamp(start)]
    if end is not None:
        data = data[data["datetime"] <= pd.Timestamp(end)]

    z_threshold = z_threshold or config.ANOMALY["zscore_threshold"]

    # Recompute seasonal z-score when missing so the module is self-contained.
    if "seasonal_zscore" not in data.columns or data["seasonal_zscore"].isna().all():
        g = data["consumption_mw"]
        mean168 = g.groupby(data["region"]).transform(lambda s: s.rolling(168, min_periods=1).mean())
        std168 = g.groupby(data["region"]).transform(lambda s: s.rolling(168, min_periods=1).std())
        data["seasonal_zscore"] = ((g - mean168) / std168.replace(0, np.nan)).fillna(0.0)

    if "anomaly_iqr" not in data.columns:
        q1 = data.groupby("region")["consumption_mw"].transform("quantile", q=0.25)
        q3 = data.groupby("region")["consumption_mw"].transform("quantile", q=0.75)
        iqr = q3 - q1
        lo, hi = q1 - config.ANOMALY["iqr_factor"] * iqr, q3 + config.ANOMALY["iqr_factor"] * iqr
        data["anomaly_iqr"] = ((data["consumption_mw"] < lo) | (data["consumption_mw"] > hi)).astype(int)

    if "rolling_mean_168h" in data.columns:
        baseline = data["rolling_mean_168h"]
    else:
        baseline = data.groupby("region")["consumption_mw"].transform(
            lambda s: s.rolling(168, min_periods=1).mean()
        )

    mask = data["seasonal_zscore"].abs() >= z_threshold
    iqr_mask = data["anomaly_iqr"] == 1
    data["__flag"] = (mask | iqr_mask).astype(bool)

    anomalies: list[dict] = []
    flagged = data[data["__flag"]]
    for _, row in flagged.iterrows():
        baseline_v = float(row["rolling_mean_168h"]) if "rolling_mean_168h" in row else float(row["consumption_mw"])
        magnitude = float((row["consumption_mw"] - baseline_v) / baseline_v * 100) if baseline_v else 0.0
        reasons = []
        if abs(float(row["seasonal_zscore"])) >= z_threshold:
            reasons.append(
                f"seasonal z-score {row['seasonal_zscore']:.2f} (threshold +/-{z_threshold})"
            )
        if int(row["anomaly_iqr"]) == 1:
            reasons.append("outside IQR fence")
        a = Anomaly(
            timestamp=str(row["datetime"]),
            region=str(row["region"]),
            consumption_mw=float(row["consumption_mw"]),
            seasonal_zscore=float(row["seasonal_zscore"]),
            iqr_outlier=bool(int(row["anomaly_iqr"])),
            magnitude_pct=round(magnitude, 2),
            reason="; ".join(reasons),
            baseline_mw=float(baseline_v),
        )
        anomalies.append(a.to_dict())

    anomalies.sort(key=lambda a: (a["timestamp"], a["region"]))
    return anomalies


def anomaly_summary(df: pd.DataFrame, region: str | None = None) -> dict:
    """High-level stats about detected anomalies for a region/window."""
    rows = detect_anomalies(df, region=region)
    if not rows:
        return {"count": 0, "avg_magnitude_pct": 0.0, "max_magnitude_pct": 0.0}
    mags = [abs(r["magnitude_pct"]) for r in rows]
    return {
        "count": len(rows),
        "avg_magnitude_pct": float(np.mean(mags)),
        "max_magnitude_pct": float(np.max(mags)),
    }