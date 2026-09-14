"""Energy cost optimisation via load-shifting and battery storage.

A linear programme (scipy linprog / HiGHS) minimises the total electricity bill
(energy cost + peak demand-charge penalty) over a forecast horizon by choosing:

* Battery charge / discharge schedule (efficiency losses, SOC bounds)
* Curtailment of shiftable non-critical loads (penalised, bounded)

The output is a per-hour schedule plus baseline-vs-optimised bill comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from . import config
from .pricing import BillComparison, compute_bill, rate_for


class OptimizeError(Exception):
    """Raised when the LP fails or produces an infeasible problem."""


@dataclass
class ScheduleRow:
    datetime: str
    forecast_mw: float
    charge_mw: float
    discharge_mw: float
    curtailment_mw: float
    net_mw: float
    soc: float
    rate: float
    cost_usd: float

    def as_dict(self) -> dict:
        return asdict(self)


def _rate_from_dt(dt_str: str, tariff: dict) -> float:
    ts = pd.Timestamp(dt_str)
    return rate_for(ts.hour, ts.dayofweek, tariff)


def optimize(
    forecast: dict,
    tariff: dict | None = None,
    region: str | None = None,
) -> dict:
    """Run the LP optimiser on a forecast result; return schedule + savings.

    Args:
        forecast: output of ``ForecastEngine.forecast`` with ``forecast`` list
            of per-hour dicts containing ``datetime`` and ``forecast_mw``.
        tariff: optional tariff override (see config.TARIFF).
    """
    tariff = tariff or config.TARIFF
    opt = config.OPTIMIZATION

    f_list = forecast.get("forecast", [])
    if not f_list:
        raise OptimizeError("forecast list is empty")

    n = len(f_list)
    dts = [f["datetime"] for f in f_list]
    y = np.array([f["forecast_mw"] for f in f_list], dtype=float)
    rates = np.array([rate_for(pd.Timestamp(dt).hour, pd.Timestamp(dt).dayofweek, tariff) for dt in dts])

    cap = opt["battery_capacity_mwh"]
    soc0 = opt["initial_soc"] * cap
    eff = opt["battery_efficiency"]
    max_chg = opt["battery_max_charge_mw"]
    max_dis = opt["battery_max_discharge_mw"]
    max_curt = opt["shiftable_capacity_mw"]
    demand_coeff = tariff["peak_demand_charge"]

    max_rate = float(np.max(rates))
    curt_pen = max_rate * 1.5

    # ---- variables: [discharge(n), charge(n), curt(n), peak, soc(n)] -----
    nd, nc, nk, npk = n, n, n, 1
    n_soc = n
    n_vars = nd + nc + nk + npk + n_soc
    o = nd + nc + nk            # offset of peak
    s = nd + nc + nk + npk      # offset of soc vars

    obj = np.zeros(n_vars)
    obj[:nd] = -rates          # discharge saves energy
    obj[nd:nd + nc] = rates    # charge costs energy
    obj[nd + nc:nd + nc + nk] = rates + curt_pen  # curtailment penalised
    obj[o] = demand_coeff      # peak demand charge

    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    A_eq: list[np.ndarray] = []
    b_eq: list[float] = []

    # -- peak dominance: peak >= net_i = y_i - dis_i + chg_i - curt_i
    #    -> -peak + dis_i - chg_i + curt_i <= -y_i
    for i in range(n):
        row = np.zeros(n_vars)
        row[o] = -1.0
        row[i] = 1.0
        row[nd + i] = -1.0
        row[nd + nc + i] = 1.0
        A_ub.append(row)
        b_ub.append(-y[i])

    # -- SOC dynamics: soc_{i+1} = soc_i + eff*chg_i - dis_i
    for i in range(n):
        row = np.zeros(n_vars)
        row[s + i] = 1.0
        if i > 0:
            row[s + i - 1] = -1.0
        row[nd + i] = eff
        row[i] = -1.0
        A_eq.append(row)
        b_eq.append(soc0 if i == 0 else 0.0)

    # -- SOC bounds 0 <= soc_i <= cap; final >= soc0 (never deplete)
    for i in range(n):
        row_ub = np.zeros(n_vars); row_ub[s + i] = 1.0
        row_lb = np.zeros(n_vars); row_lb[s + i] = -1.0
        A_ub.append(row_ub); b_ub.append(cap)
        A_ub.append(row_lb); b_ub.append(0.0)
    row_final = np.zeros(n_vars)
    row_final[s + n - 1] = -1.0
    A_ub.append(row_final)
    b_ub.append(-soc0)

    # -- variable bounds
    bounds: list[tuple[float | None, float | None]] = [(0.0, min(max_dis, v)) for v in y]  # discharge (no export)
    bounds += [(0.0, max_chg)] * n                                                          # charge
    bounds += [(0.0, v) for v in np.minimum(max_curt, y)]                                    # curtailment
    bounds += [(float(np.max(y)), None)]                                                     # peak
    bounds += [(0.0, cap)] * n                                                               # soc

    res = linprog(
        obj,
        A_ub=np.array(A_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        A_eq=np.array(A_eq, dtype=float),
        b_eq=np.array(b_eq, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise OptimizeError(f"LP infeasible or solver error: {res.message}")

    x = res.x
    dis = x[:nd]
    chg = x[nd:nd + nc]
    cur = x[nd + nc:nd + nc + nk]
    soc_mwh = x[s:]
    net = y - dis + chg - cur

    schedule = []
    for i in range(n):
        schedule.append(ScheduleRow(
            datetime=dts[i],
            forecast_mw=round(float(y[i]), 2),
            charge_mw=round(float(chg[i]), 2),
            discharge_mw=round(float(dis[i]), 2),
            curtailment_mw=round(float(cur[i]), 2),
            net_mw=round(float(net[i]), 2),
            soc=round(float(soc_mwh[i]) / cap, 4),
            rate=float(rates[i]),
            cost_usd=round(float(rates[i] * max(net[i], 0.0)), 2),
        ).as_dict())

    idx = pd.to_datetime(dts)
    baseline_bill = compute_bill(pd.Series(y), idx, tariff)
    optimized_bill = compute_bill(pd.Series(y), idx, tariff, net_consumption_mw=pd.Series(net))
    total0 = baseline_bill["total_cost_usd"]
    savings_usd = total0 - optimized_bill["total_cost_usd"]
    savings_pct = savings_usd / total0 * 100 if total0 else 0.0
    peak0 = baseline_bill["peak_mw"]
    peak_red = (peak0 - optimized_bill["peak_mw"]) / peak0 * 100 if peak0 else 0.0
    peak_red = max(peak_red, 0.0)

    return {
        "region": region or "unknown",
        "horizon": n,
        "battery": {
            "capacity_mwh": cap,
            "initial_soc": round(soc0 / cap, 4),
            "final_soc": round(float(soc_mwh[-1]) / cap, 4),
            "efficiency": eff,
        },
        "summary": BillComparison(
            baseline_bill, optimized_bill, savings_usd, savings_pct, peak_red
        ).as_dict(),
        "schedule": schedule,
    }