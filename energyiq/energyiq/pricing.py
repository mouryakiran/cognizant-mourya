"""Time-of-use pricing model and billing calculations.

Standard commercial tariff with three periods:
    peak, shoulder, off-peak  (config.TARIFF)
and an optional peak demand charge on the billing peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


def period_for(hour: int, dow: int) -> str:
    """Return the tariff period for an hour (0-23) and day-of-week (Mon=0)."""
    if dow in config.PEAK_DOWS:
        if hour in config.PEAK_HOURS:
            return "peak"
        if hour in config.SHOULDER_HOURS:
            return "shoulder"
    return "offpeak"


def rate_for(hour: int, dow: int, tariff: dict | None = None) -> float:
    """Energy rate ($/MWh) for a given hour and day-of-week."""
    tariff = tariff or config.TARIFF
    return float(tariff[period_for(hour, dow)])


def hourly_rates(index: pd.DatetimeIndex, tariff: dict | None = None) -> np.ndarray:
    """Array of $/MWh rates aligned to ``index``."""
    return np.array([rate_for(ts.hour, ts.dayofweek, tariff) for ts in index])


def compute_bill(
    consumption_mw: pd.Series,
    index: pd.DatetimeIndex,
    tariff: dict | None = None,
    net_consumption_mw: pd.Series | None = None,
) -> dict:
    """Compute energy + demand-charge bill for a load profile.

    Args:
        consumption_mw: baseline hourly demand (MW).
        index: hourly timestamps.
        tariff: rate table override.
        net_consumption_mw: net load after optimization; when None,
            ``consumption_mw`` is used (baseline bill).
    """
    tariff = tariff or config.TARIFF
    net = net_consumption_mw if net_consumption_mw is not None else consumption_mw
    energy = float(np.sum(hourly_rates(index, tariff) * np.maximum(net, 0)))
    peak = float(np.max(net)) if len(net) else 0.0
    demand_cost = peak * tariff["peak_demand_charge"] * (len(index) / 24.0)
    return {
        "energy_cost_usd": round(energy, 2),
        "peak_mw": round(peak, 2),
        "demand_charge_usd": round(demand_cost, 2),
        "total_cost_usd": round(energy + demand_cost, 2),
    }


@dataclass
class BillComparison:
    baseline: dict
    optimized: dict
    savings_usd: float
    savings_pct: float
    peak_reduction_pct: float

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "optimized": self.optimized,
            "savings_usd": round(self.savings_usd, 2),
            "savings_pct": round(self.savings_pct, 2),
            "peak_reduction_pct": round(self.peak_reduction_pct, 2),
        }