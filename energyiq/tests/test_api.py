"""End-to-end API tests using FastAPI's TestClient with the tiny dataset."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api


@pytest.fixture(scope="module")
def client(trained_engine):
    eng, df = trained_engine

    def fake_engine():
        return eng

    def fake_dataset():
        return df

    api._engine.cache_clear()
    api._dataset.cache_clear()
    api._engine = fake_engine  # type: ignore[assignment]
    api._dataset = fake_dataset  # type: ignore[assignment]
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_regions(client):
    r = client.get("/api/regions")
    assert r.status_code == 200
    assert "AEP" in r.json()["regions"]


def test_dataset_info(client):
    r = client.get("/api/dataset/info")
    assert r.status_code == 200
    assert r.json()["rows"] > 0
    assert r.json()["schema_ok"] is True


def test_history(client):
    r = client.get("/api/history?region=AEP&days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["region"] == "AEP"
    assert len(body["points"]) > 100  # ~168 hourly points


def test_forecast_endpoint(client):
    r = client.post("/api/forecast", json={"region": "AEP", "horizon": 12})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 12
    assert body["region"] == "AEP"


def test_forecast_unknown_region(client):
    r = client.post("/api/forecast", json={"region": "XYZ", "horizon": 12})
    assert r.status_code == 400


def test_forecast_site_scaling(client):
    r = client.post("/api/forecast", json={"region": "AEP", "horizon": 24, "site_scale_mw": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["scaled_to_mw"] == 5
    peak = max(p["forecast_mw"] for p in body["forecast"])
    assert peak <= 5.0001


def test_optimize_endpoint(client):
    r = client.post("/api/optimize", json={"region": "AEP", "horizon": 48})
    assert r.status_code == 200
    body = r.json()
    assert len(body["schedule"]) == 48
    assert body["summary"]["savings_usd"] >= 0


def test_recommendations_endpoint(client):
    r = client.post("/api/recommendations", json={"region": "AEP", "horizon": 48})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["recommendations"], list)
    assert len(body["recommendations"]) > 0


def test_anomalies_endpoint(client):
    r = client.get("/api/anomalies?region=AEP&days=7&limit=10")
    assert r.status_code == 200
    assert isinstance(r.json()["anomalies"], list)


def test_summary_endpoint(client):
    r = client.get("/api/summary?region=AEP&days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["avg_mw"] > 0
    assert "bill" in body


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "EnergyIQ" in r.text