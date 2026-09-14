"""Pydantic schemas for the EnergyIQ API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    region: str = Field("AEP", description="Region code, e.g. AEP, PJME, ...")
    horizon: int = Field(48, ge=1, le=336, description="Forecast horizon in hours")
    start: str | None = Field(None, description="ISO timestamp of the last actual observation")
    site_scale_mw: float | None = Field(None, gt=0, description="Scale the profile to this site peak (MW)")


class OptimizeRequest(ForecastRequest):
    pass


class AnalysisRequest(ForecastRequest):
    include_history_days: int = Field(7, ge=0, le=60)


class RecommendationsRequest(ForecastRequest):
    pass


class AnomaliesRequest(BaseModel):
    region: str | None = Field(None)
    days: int = Field(7, ge=1, le=180)
    limit: int = Field(50, ge=1, le=500)


class HistoryRequest(BaseModel):
    region: str = Field("AEP")
    days: int = Field(7, ge=1, le=120)


class SummaryRequest(BaseModel):
    region: str = Field("AEP")
    days: int = Field(30, ge=1, le=365)