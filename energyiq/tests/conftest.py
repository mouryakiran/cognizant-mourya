"""Shared fixtures for the EnergyIQ test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyiq.dataset import SCHEMA_COLUMNS, generate_dataset, load_dataset
from energyiq.forecast import ForecastEngine


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory) -> str:
    """A small but valid generated dataset used across tests."""
    out = tmp_path_factory.mktemp("data") / "tiny.csv"
    generate_dataset(
        start="2012-01-01",
        end="2012-03-31",
        regions=["AEP", "COMED"],
        out_path=out,
        progress=False,
    )
    return str(out)


@pytest.fixture(scope="session")
def tiny_df(tiny_dataset):
    return load_dataset(tiny_dataset)


@pytest.fixture(scope="session")
def trained_engine(tiny_dataset, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("models")
    eng = ForecastEngine(model_path=tmp / "m.joblib", meta_path=tmp / "meta.json")
    df = load_dataset(tiny_dataset)
    eng.train(df, test_tail_hours=24 * 20, max_train_rows=None, save=True)
    return eng, df