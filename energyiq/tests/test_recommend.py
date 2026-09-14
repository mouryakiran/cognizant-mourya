"""Tests for the recommendations engine."""

from __future__ import annotations

import pandas as pd

from energyiq.optimize import optimize
from energyiq.recommend import build_recommendations


def make_forecast(n=48, base=300.0, amp=120.0, start="2016-07-15 00:00"):
    """A site-scale test profile with a clear weekday peak pattern starting Mon."""
    st = pd.Timestamp(start)  # 2016-07-15 is a Friday
    rows = []
    for i in range(n):
        t = st + pd.Timedelta(hours=i)
        pk = 1.25 if t.dayofweek < 5 and 15 <= t.hour <= 20 else 1.0
        base_amp = base + amp * pk * (0.35 + 0.65 * ((6 - min(abs(t.hour - 18), 6)) / 6))
        rows.append({"datetime": t.isoformat(), "forecast_mw": float(base_amp)})
    return {"region": "TEST", "horizon": n, "forecast": rows}


def test_recommendations_generated():
    fc = make_forecast()
    opt = optimize(fc, region="TEST")
    recs = build_recommendations(fc, optimize_result=opt, region="TEST")
    assert isinstance(recs, list)
    assert len(recs) > 0


def test_recommendations_ordered_by_priority():
    fc = make_forecast()
    recs = build_recommendations(fc, region="TEST")
    prio = {"high": 0, "medium": 1, "low": 2}
    order = [prio[r["priority"]] for r in recs]
    assert order == sorted(order)


def test_recommendations_have_required_keys():
    fc = make_forecast()
    recs = build_recommendations(fc, region="TEST")
    for r in recs:
        for key in ("id", "title", "category", "priority", "description", "steps"):
            assert key in r


def test_empty_forecast_gives_empty_recommendations():
    assert build_recommendations({"forecast": []}) == []


def test_anomaly_aware_recommendation():
    fc = make_forecast()
    anomalies = [{
        "timestamp": fc["forecast"][3]["datetime"],
        "region": "TEST", "consumption_mw": 999, "seasonal_zscore": 9.9,
        "iqr_outlier": True, "magnitude_pct": -40.0, "reason": "spike",
        "baseline_mw": 300.0,
    }]
    recs = build_recommendations(fc, anomalies=anomalies, region="TEST")
    assert any(r["id"] == "anomaly-review" for r in recs)


def test_storage_cycle_recommendation_when_battery_used():
    fc = make_forecast(n=168)
    opt = optimize(fc, region="TEST")
    recs = build_recommendations(fc, optimize_result=opt, region="TEST")
    assert any(r["id"] == "storage-cycle" for r in recs)