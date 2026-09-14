"""Recommendation engine.

Translates forecasts, tariffs, anomalies and optimisation outputs into concrete,
actionable energy-saving recommendations with estimated savings and priorities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config
from .pricing import rate_for


@dataclass
class Recommendation:
    id: str
    title: str
    category: str
    priority: str
    description: str
    steps: list[str] = field(default_factory=list)
    potential_kwh: float = 0.0
    potential_usd: float = 0.0
    confidence: str = field(default="medium")

    def as_dict(self) -> dict:
        return asdict(self)


def _period_summary(f_list: list[dict], tariff: dict) -> dict:
    """Hourly-rate stats of the forecast by period."""
    pk = sh = op = 0.0
    for f in f_list:
        ts = pd.Timestamp(f["datetime"])
        r = rate_for(ts.hour, ts.dayofweek, tariff)
        v = f["forecast_mw"]
        if r >= tariff["peak"] - 1e-9:
            pk += v
        elif r >= tariff["shoulder"] - 1e-9:
            sh += v
        else:
            op += v
    return {
        "peak_mwh": round(pk, 1),
        "shoulder_mwh": round(sh, 1),
        "offpeak_mwh": round(op, 1),
        "peak_rate": tariff["peak"],
        "offpeak_rate": tariff["offpeak"],
    }


def build_recommendations(
    forecast: dict,
    optimize_result: dict | None = None,
    anomalies: list[dict] | None = None,
    history: pd.DataFrame | None = None,
    region: str | None = None,
) -> list[dict]:
    """Generate a ranked list of recommendations.

    Args:
        forecast: output of ForecastEngine.forecast.
        optimize_result: optional output of energyiq.optimize.optimize.
        anomalies: optional anomaly rows from anomaly.detect_anomalies.
        history: historical dataset (used for percentile comparisons).
        region: region label used in the UI.
    """
    f_list = forecast.get("forecast", [])
    if not f_list:
        return []
    tariff = config.TARIFF
    recs: list[Recommendation] = []
    vals = np.array([f["forecast_mw"] for f in f_list])
    peak_val = float(np.max(vals))

    # -------------------------------------------------------------- 1. arbitrage
    ps = _period_summary(f_list, tariff)
    per = config.OPTIMIZATION
    if ps["peak_mwh"] > 0 and ps["offpeak_mwh"] > 0:
        shiftable_energy = per["shiftable_capacity_mw"] * 6
        usd = shiftable_energy * (ps["peak_rate"] - ps["offpeak_rate"])
        recs.append(Recommendation(
            id="arbitrage",
            title="Shift load from peak to off-peak hours",
            category="Load shifting",
            priority="high",
            description=(
                f"Forecast shows {ps['peak_mwh']:,.0f} MWh during on-peak hours. "
                f"Moving deferrable loads {ps['peak_rate']:.0f}/MWh -> "
                f"{ps['offpeak_rate']:.0f}/MWh captures the price spread."
            ),
            steps=[
                "Sideline non-critical cooling/EV/charging workloads to 00:00-06:00",
                "Pre-cool building thermal mass during the last off-peak hour",
                "Stage industrial floor washing / air handling outside peak",
            ],
            potential_kwh=round(shiftable_energy * 1000),
            potential_usd=round(usd, 2),
            confidence="high",
        ))

    # -------------------------------------------------------------- 2. demand cap
    if optimize_result:
        summ = optimize_result["summary"]["optimized"]
        savings = optimize_result["summary"]["savings_usd"]
        peak_red = optimize_result["summary"]["peak_reduction_pct"]
    else:
        summ = None
        savings = 0.0
        peak_red = 0.0
    if peak_red > 0.5:
        recs.append(Recommendation(
            id="demand-charge",
            title="Shave the billing peak to cut demand charges",
            category="Demand response",
            priority="high",
            description=(
                f"Optimising with the {per['battery_capacity_mwh']:.0f} MWh storage "
                f"cuts the forecast peak by {peak_red:.1f}% (peak now "
                f"{summ['peak_mw']:,.1f} MW vs {optimize_result['summary']['baseline']['peak_mw']:,.1f} MW)."
            ),
            steps=[
                "Discharge storage on the highest-demand forecast windows",
                "Synchronise big non-critical loads away from the regional peak hour",
            ],
            potential_usd=round(savings, 2),
            confidence="high",
        ))

    # -------------------------------------------------------------- 3. anomalies
    if anomalies:
        recent = anomalies[:5]
        times = ", ".join(a["timestamp"][:16] for a in recent)
        recs.append(Recommendation(
            id="anomaly-review",
            title="Investigate unusual demand events",
            category="Operations",
            priority="high" if any(abs(a["magnitude_pct"]) > 25 for a in recent) else "medium",
            description=(
                f"{len(anomalies)} anomalous hour(s) flagged (max deviation "
                f"{max((abs(a['magnitude_pct']) for a in recent), default=0):.1f}% from "
                f"seasonal baseline). Recent: {times}."
            ),
            steps=[
                "Review meter logs / equipment faults at flagged timestamps",
                "Verify HPWH/HVAC schedules did not overlap at those hours",
                "Check for unoccupied-zone heating/cooling (thermostat overrides)",
            ],
            confidence="medium",
        ))

    # -------------------------------------------------------------- 4. season / HVAC
    d = pd.to_datetime(f_list[0]["datetime"])
    month = d.month
    avg = float(np.mean(vals))
    if month in (5, 6, 7, 8):
        recs.append(Recommendation(
            id="precooling",
            title="Pre-cool before the on-peak window during cooling season",
            category="HVAC",
            priority="medium",
            description=(
                f"Current month has sustained cooling load. Charging the building's "
                f"thermal mass before {', '.join(f'{h:02d}:00' for h in config.PEAK_HOURS[:2])} "
                f"reduces compressor starts during peak rates."
            ),
            steps=[
                "Run economy-mode pre-cooling 05:00-08:00 (off-peak)",
                "Raise the dehumidifier setpoint by 1F during shoulder hours",
            ],
            potential_usd=round(avg * 0.01 * 24 * (ps["peak_rate"] - ps["offpeak_rate"]) * (d.days_in_month / 30.0) / 1000, 0),
            confidence="medium",
        ))
    else:
        recs.append(Recommendation(
            id="setback",
            title="Apply night/weekend temperature setbacks",
            category="HVAC",
            priority="medium",
            description=(
                f"Heating season: a 1C night setback on unoccupied zones ~21:00-06:00 "
                f"reduces baseline heat loss while keeping frost protection."
            ),
            steps=[
                "Set schedule 19:00-05:00 setback to 18C for offices",
                "Keep warehouse setpoint -2C below occupied setting",
            ],
            confidence="medium",
        ))

    # -------------------------------------------------------------- 5. weekend
    is_wknd = [pd.Timestamp(f["datetime"]).dayofweek >= 5 for f in f_list]
    if any(is_wknd):
        wknd_avg = float(np.mean([v for v, w in zip(vals, is_wknd) if w]))
        wday_avg = float(np.mean([v for v, w in zip(vals, is_wknd) if not w])) if len(vals) > sum(is_wknd) else wknd_avg
        if wday_avg > 0 and wknd_avg / wday_avg < 0.97:
            recs.append(Recommendation(
                id="weekend-setback",
                title="Deepen weekend unoccupied setpoint",
                category="HVAC",
                priority="low",
                description=(
                    f"Weekend forecast demand is {(1 - wknd_avg / wday_avg) * 100:.0f}% below "
                    f"weekday levels - occupancy is low, so setback can be more aggressive."
                ),
                steps=[
                    "Extend weekend setpoint by 2C on non-production zones",
                    "Run ASHRAE G3.1 scheduling for holiday-style mode",
                ],
                confidence="low",
            ))

    # -------------------------------------------------------------- 6. holiday
    hol = [(i, f) for i, f in enumerate(f_list) if pd.Timestamp(f["datetime"]).month in (12,) and pd.Timestamp(f["datetime"]).day >= 24]
    if hol:
        recs.append(Recommendation(
            id="holiday-mode",
            title="Activate holiday operating mode",
            category="Scheduling",
            priority="low",
            description="Seasonal holiday window detected in the forecast - use holiday mode to curtail non-essential loads.",
            steps=[
                "Disable non-essential equipment schedules during holiday days",
                "Run only frost-protection and security loads",
            ],
            potential_kwh=round(avg * 0.1 * len(hol) * 24 * 1000),
            confidence="medium",
        ))

    # -------------------------------------------------------------- 7. storage cycle
    if optimize_result:
        chg_hours = [s for s in optimize_result["schedule"] if s["charge_mw"] > 0.05]
        dis_hours = [s for s in optimize_result["schedule"] if s["discharge_mw"] > 0.05]
        if chg_hours and dis_hours:
            recs.append(Recommendation(
                id="storage-cycle",
                title="Run the recommended battery charge/discharge cycle",
                category="Storage",
                priority="medium",
                description=(
                    f"Optimal cycle: charge {len(chg_hours)}h (mostly off-peak), "
                    f"discharge {len(dis_hours)}h on-peak. Final SOC target "
                    f"{optimize_result['battery']['final_soc']:.0%}."
                ),
                steps=[
                    f"Charge up to {optimize_result['battery']['capacity_mwh']:.0f} MWh before {chg_hours[0]['datetime'][11:16]}",
                    "Enable discharge only inside the peak window to protect cycle life",
                ],
                potential_usd=round(savings, 2),
                confidence="high",
            ))
        curt = [s for s in optimize_result["schedule"] if s["curtailment_mw"] > 0.05]
        if curt:
            recs.append(Recommendation(
                id="curtail",
                title="Defer shiftable loads on constrained hours",
                category="Demand response",
                priority="medium",
                description=(
                    f"{len(curt)} hour(s) with recommended load deferral "
                    f"(max {max(s['curtailment_mw'] for s in curt):.1f} MW) to avoid peak overlap."
                ),
                steps=[
                    "Schedule batch industrial processes outside the flagged hours",
                    "Hold EV fleet charging until off-peak",
                ],
                confidence="high",
            ))

    # -------------------------------------------------------------- 8. context
    if history is not None and len(history):
        try:
            sub = history[history["region"] == forecast.get("region")].copy()
            if len(sub):
                p95 = sub["consumption_mw"].quantile(0.95)
                if peak_val > p95:
                    recs.append(Recommendation(
                        id="record-peak",
                        title="Forecast exceeds historical 95th percentile",
                        category="Operations",
                        priority="high",
                        description=(
                            f"Forecast peak {peak_val:,.0f} MW exceeds the local 95th "
                            f"percentile ({p95:,.0f} MW). Consider demand-response alerts."
                        ),
                        steps=[
                            "Notify the energy manager via alerting channel",
                            "Pre-verify generators/battery availability",
                        ],
                        confidence="medium",
                    ))
        except Exception:
            pass

    # order by priority
    prio = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: prio[r.priority])
    out = [r.as_dict() for r in recs]
    for i, r in enumerate(out):
        r["potential_usd"] = round(r.get("potential_usd", 0.0), 2)
    return out