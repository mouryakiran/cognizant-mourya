"""EnergyIQ FastAPI application.

Run with:  python scripts/run_api.py   (starts uvicorn on :8000)
Test with: pytest tests/test_api.py
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import AnalysisRequest, ForecastRequest, OptimizeRequest
from energyiq import config
from energyiq.anomalies import anomaly_summary, detect_anomalies
from energyiq.dataset import load_dataset
from energyiq.forecast import ForecastEngine, ForecastError
from energyiq.optimize import OptimizeError, optimize
from energyiq.pricing import compute_bill
from energyiq.recommend import build_recommendations

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("energyiq.api")

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

app = FastAPI(title="EnergyIQ - Smart Energy Optimisation Platform", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@functools.lru_cache(maxsize=1)
def _dataset():
    return load_dataset()


@functools.lru_cache(maxsize=1)
def _engine() -> ForecastEngine:
    return ForecastEngine().load()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scale(forecast: dict, site_scale_mw: float | None) -> dict:
    if not site_scale_mw:
        return forecast
    pts = forecast.get("forecast", [])
    if not pts:
        return forecast
    peak = max(p["forecast_mw"] for p in pts)
    if peak <= 0:
        return forecast
    k = site_scale_mw / peak
    out = []
    for p in pts:
        out.append({**p, "forecast_mw": round(p["forecast_mw"] * k, 3),
                    "upper_mw": round(p["upper_mw"] * k, 3),
                    "lower_mw": round(p["lower_mw"] * k, 3)})
    forecast["forecast"] = out
    forecast["scaled_to_mw"] = site_scale_mw
    forecast["scale_factor"] = round(k, 6)
    return forecast


def _analysis(region: str, horizon: int, start: str | None, site_scale_mw: float | None) -> dict:
    df = _dataset()
    eng = _engine()
    fc = eng.forecast(region, horizon=horizon, df=df, start=start)
    fc = _scale(fc, site_scale_mw)
    opt = optimize(fc, region=region)
    anoms = detect_anomalies(df, region=region,
                             start=fc["anchor"][:10],
                             end=fc["forecast"][-1]["datetime"][:10])
    recs = build_recommendations(fc, optimize_result=opt, anomalies=anoms,
                                 history=df, region=region)
    return {"forecast": fc, "optimization": opt, "recommendations": recs, "anomalies": anoms[:20]}


# --------------------------------------------------------------------------- #
# UI + health
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# --------------------------------------------------------------------------- #
# Data endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/regions")
def api_regions():
    df = _dataset()
    return {
        "regions": sorted(df["region"].unique().tolist()),
        "train_regions": _engine().meta.get("regions", []),
        "model": _engine().meta.get("metrics", {}),
        "source": None,
    }


@app.get("/api/dataset/info")
def api_dataset_info():
    df = _dataset()
    return {
        "rows": int(len(df)),
        "regions": sorted(df["region"].unique().tolist()),
        "start": str(df["datetime"].min()),
        "end": str(df["datetime"].max()),
        "columns": list(df.columns),
        "schema_ok": list(df.columns) == _schema_cols(),
    }


def _schema_cols():
    from energyiq.dataset import SCHEMA_COLUMNS
    return SCHEMA_COLUMNS


@app.get("/api/history")
def api_history(region: str = Query("AEP"), days: int = Query(7, ge=1, le=120)):
    df = _dataset()
    sub = df[df["region"] == region].copy()
    if sub.empty:
        raise HTTPException(404, f"region {region} not found")
    sub = sub.sort_values("datetime")
    cutoff = sub["datetime"].max() - pd.Timedelta(days=days)
    sub = sub[sub["datetime"] >= cutoff]
    return {
        "region": region,
        "points": [
            {"datetime": str(ts), "consumption_mw": round(v, 3)}
            for ts, v in zip(sub["datetime"], sub["consumption_mw"])
        ],
    }


# --------------------------------------------------------------------------- #
# Analysis endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/forecast", response_model=None)
def api_forecast(req: ForecastRequest = ForecastRequest()):
    return _analysis(req.region, req.horizon, req.start, req.site_scale_mw)["forecast"]


@app.post("/api/forecast")
def api_forecast_post(req: ForecastRequest):
    df = _dataset()
    eng = _engine()
    try:
        fc = eng.forecast(req.region, horizon=req.horizon, df=df, start=req.start)
    except ForecastError as e:
        raise HTTPException(400, str(e))
    return _scale(fc, req.site_scale_mw)


@app.get("/api/optimize")
def api_optimize(region: str = Query("AEP"), horizon: int = Query(48),
                 start: str | None = Query(None), site_scale_mw: float | None = Query(None)):
    df = _dataset()
    eng = _engine()
    try:
        fc = eng.forecast(region, horizon=horizon, df=df, start=start)
    except ForecastError as e:
        raise HTTPException(400, str(e))
    fc = _scale(fc, site_scale_mw)
    try:
        return optimize(fc, region=region)
    except OptimizeError as e:
        raise HTTPException(400, str(e))


@app.post("/api/optimize")
def api_optimize_post(req: OptimizeRequest):
    df = _dataset()
    eng = _engine()
    try:
        fc = eng.forecast(req.region, horizon=req.horizon, df=df, start=req.start)
    except ForecastError as e:
        raise HTTPException(400, str(e))
    fc = _scale(fc, req.site_scale_mw)
    try:
        return optimize(fc, region=req.region)
    except OptimizeError as e:
        raise HTTPException(400, str(e))


@app.get("/api/recommendations")
def api_recommendations(region: str = Query("AEP"), horizon: int = Query(48),
                        site_scale_mw: float | None = Query(None)):
    try:
        res = _analysis(region, horizon, None, site_scale_mw)
    except (ForecastError, OptimizeError) as e:
        raise HTTPException(400, str(e))
    return {"region": region, "recommendations": res["recommendations"]}


@app.post("/api/recommendations")
def api_recommendations_post(req: OptimizeRequest):
    try:
        res = _analysis(req.region, req.horizon, req.start, req.site_scale_mw)
    except (ForecastError, OptimizeError) as e:
        raise HTTPException(400, str(e))
    return {"region": req.region, "recommendations": res["recommendations"]}


@app.get("/api/anomalies")
def api_anomalies(region: str | None = Query(None), days: int = Query(7, ge=1, le=180),
                  limit: int = Query(50, ge=1, le=500)):
    df = _dataset()
    end = df["datetime"].max()
    start = end - pd.Timedelta(days=days)
    return {"anomalies": detect_anomalies(df, region=region, start=str(start), end=str(end))[:limit]}


@app.get("/api/summary")
def api_summary(region: str = Query("AEP"), days: int = Query(30, ge=1, le=365)):
    df = _dataset()
    sub = df[df["region"] == region]
    if sub.empty:
        raise HTTPException(404, f"region {region} not found")
    sub = sub.copy()
    sub["datetime"] = pd.to_datetime(sub["datetime"])
    cutoff = sub["datetime"].max() - pd.Timedelta(days=days)
    sub = sub[sub["datetime"] >= cutoff]

    bill = compute_bill(sub["consumption_mw"], sub["datetime"])

    return {
        "region": region,
        "days": days,
        "avg_mw": round(float(sub["consumption_mw"].mean()), 2),
        "peak_mw": round(float(sub["consumption_mw"].max()), 2),
        "min_mw": round(float(sub["consumption_mw"].min()), 2),
        "consumed_mwh": round(float(sub["consumption_mw"].sum()), 2),
        "bill": bill,
        "anomalies": anomaly_summary(sub, region),
        "last_updated": str(sub["datetime"].max()),
    }


import pandas as pd  # noqa: E402