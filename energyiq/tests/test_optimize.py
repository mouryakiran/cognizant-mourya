"""Tests for the LP-based energy optimizer and pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energyiq import config
from energyiq.optimize import OptimizeError, optimize
from energyiq.pricing import compute_bill, period_for, rate_for


def make_forecast(n=48, base=12000.0, amp=2500.0, start="2016-08-01 00:00"):
    start_ts = pd.Timestamp(start)
    vals = base + amp * np.sin(2 * np.pi * np.arange(n) / 24.0) + np.random.default_rng(1).normal(0, 150, n)
    return {
        "region": "TEST",
        "horizon": n,
        "forecast": [
            {"datetime": (start_ts + pd.Timedelta(hours=i)).isoformat(), "forecast_mw": float(v)}
            for i, v in enumerate(vals)
        ],
    }


def test_optimize_returns_schedule_and_savings():
    res = optimize(make_forecast(), region="TEST")
    assert len(res["schedule"]) == 48
    s = res["summary"]
    assert s["savings_usd"] >= 0
    assert s["peak_reduction_pct"] >= 0
    assert s["optimized"]["total_cost_usd"] <= s["baseline"]["total_cost_usd"]


def test_optimize_respects_soc_bounds():
    res = optimize(make_forecast())
    cap = config.OPTIMIZATION["battery_capacity_mwh"]
    for row in res["schedule"]:
        assert 0.0 <= row["soc"] <= 1.0
    # final SOC should not be below the initial SOC
    assert res["battery"]["final_soc"] >= res["battery"]["initial_soc"] - 1e-6


def test_optimize_never_charges_and_discharges_same_hour_wastefully():
    res = optimize(make_forecast())
    for row in res["schedule"]:
        if row["charge_mw"] > 0.01:
            assert row["discharge_mw"] <= 0.01


def test_optimize_raises_on_empty_forecast():
    try:
        optimize({"forecast": []})
        assert False, "expected OptimizeError"
    except OptimizeError:
        pass


def test_bill_strictly_positive():
    s = pd.Series([100.0] * 24)
    idx = pd.date_range("2016-08-01", periods=24, freq="h")
    bill = compute_bill(s, idx)
    assert bill["energy_cost_usd"] > 0
    assert bill["total_cost_usd"] == bill["energy_cost_usd"] + bill["demand_charge_usd"]


def test_tariff_periods():
    assert period_for(2, 0) == "offpeak"      # Mon 02:00
    assert period_for(17, 1) == "peak"        # Tue 17:00
    assert period_for(10, 4) == "shoulder"    # Fri 10:00
    assert period_for(15, 6) == "offpeak"     # Sun 15:00 (weekend)
    assert rate_for(17, 0) == config.TARIFF["peak"]
    assert rate_for(2, 0) == config.TARIFF["offpeak"]