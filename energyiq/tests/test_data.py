"""Tests for the dataset generation and loading pipeline."""

from __future__ import annotations

from energyiq.dataset import SCHEMA_COLUMNS, load_dataset


def test_schema_matches_exact_columns(tiny_dataset):
    df = load_dataset(tiny_dataset)
    assert list(df.columns) == SCHEMA_COLUMNS
    assert len(df.columns) == 44


def test_no_nulls_in_numeric_features(tiny_dataset):
    df = load_dataset(tiny_dataset)
    numerics = [c for c in SCHEMA_COLUMNS if c not in ("datetime", "region")]
    assert df[numerics].isnull().sum().sum() == 0


def test_contiguous_hourly_series_per_region(tiny_dataset):
    df = load_dataset(tiny_dataset)
    for region, sub in df.groupby("region"):
        dts = sub["datetime"].diff().dropna().dt.total_seconds() / 3600
        assert (dts == 1).all(), f"region {region} has non-hourly gaps"


def test_cyclical_encodings_within_expected_ranges(tiny_dataset):
    df = load_dataset(tiny_dataset)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"):
        assert df[col].between(-1.01, 1.01).all(), col


def test_daily_pattern_present(tiny_dataset):
    df = load_dataset(tiny_dataset)
    sub = df[df["region"] == "AEP"]
    night = sub[sub["hour"].between(2, 5)]["consumption_mw"].mean()
    day = sub[sub["hour"].between(12, 18)]["consumption_mw"].mean()
    assert day > night, "expected higher daytime than overnight demand"


def test_lag_features_reference_previous_values(tiny_dataset):
    df = load_dataset(tiny_dataset)
    sub = df[df["region"] == "AEP"].sort_values("datetime")
    assert (sub["consumption_mw"].shift(1) - sub["lag_1h"]).abs().max() < 1e-6


def test_anomaly_flags_binary(tiny_dataset):
    df = load_dataset(tiny_dataset)
    assert set(df["is_anomaly"].unique()) <= {0, 1}
    assert set(df["anomaly_iqr"].unique()) <= {0, 1}
    assert set(df["anomaly_seasonal_zscore"].unique()) <= {0, 1}