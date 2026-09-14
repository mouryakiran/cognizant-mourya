"""Tests for ML forecasting and anomaly detection."""

from __future__ import annotations

import numpy as np
import pytest

from energyiq.anomalies import anomaly_summary, detect_anomalies


def test_train_produces_reasonable_metrics(trained_engine):
    eng, _ = trained_engine
    m = eng.meta["metrics"]
    assert m["mae_mw"] > 0
    assert m["mape_pct"] < 20  # loose bound for tiny dataset


def test_forecast_shape_and_range(trained_engine):
    eng, df = trained_engine
    res = eng.forecast("AEP", horizon=24, df=df)
    assert res["horizon"] == 24
    assert len(res["forecast"]) == 24
    for pt in res["forecast"]:
        assert pt["forecast_mw"] > 0
        assert pt["upper_mw"] >= pt["forecast_mw"] >= pt["lower_mw"]


def test_forecast_unknown_region_raises(trained_engine):
    eng, df = trained_engine
    with pytest.raises(Exception):
        eng.forecast("NOPE", horizon=5, df=df)


def test_forecast_recursive_lags_consistent(trained_engine):
    eng, df = trained_engine
    res = eng.forecast("COMED", horizon=5, df=df)
    vals = [p["forecast_mw"] for p in res["forecast"]]
    assert all(np.isfinite(v) for v in vals)


def test_evaluate_hour_ahead(trained_engine):
    eng, df = trained_engine
    res = eng.evaluate(region="AEP", horizon=6, df=df)
    assert "AEP" in res["results"]
    assert res["results"]["AEP"]["mae_mw"] > 0


def test_anomalies_detected_over_window(tiny_df):
    rows = detect_anomalies(tiny_df, region="AEP")
    assert isinstance(rows, list)
    for r in rows:
        assert r["region"] == "AEP"
        assert "seasonal_zscore" in r


def test_anomaly_summary_counts(tiny_df):
    s = anomaly_summary(tiny_df, region="COMED")
    assert s["count"] >= 0
    assert s["avg_magnitude_pct"] >= 0