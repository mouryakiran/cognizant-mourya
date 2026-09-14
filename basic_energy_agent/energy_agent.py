"""
energy_agent.py — A basic Energy Optimization Agent.

Runs four tasks on historical hourly energy-consumption data:
  1. Forecasts peak energy consumption (next N days)
  2. Detects energy-consumption anomalies (statistical outliers)
  3. Recommends energy-saving actions (rule based)
  4. Estimates the kWh / cost impact of those actions

No ML libraries are required — only pandas and numpy. This is the
basic-level implementation of the AI Energy Optimization Agent.

Usage:
    python energy_agent.py [path/to/data.csv] [--region PJME] [--days 7]
                           [--rate 0.12] [--currency $]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Default dataset (12 regional utilities, hourly, 2015-2024)
DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "AI-Energy-Optimization-Agent" / "data" / "processed" / "processed_energy_data.csv"
)

# Default electricity rate in $/kWh
DEFAULT_RATE = 0.12

# Hour blocks used by the rules engine
PEAK_HOURS = set(range(17, 22))      # evening peak window
OFFICE_HOURS = set(range(8, 18))     # business window
NIGHT_HOURS = set(range(0, 6))       # overnight / idle window


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------

def load_data(path=DEFAULT_DATA, region=None):
    """Read a processed CSV into a clean, sorted, datetime-indexed frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    needed = {"Datetime", "Consumption"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df["Consumption"] = pd.to_numeric(df["Consumption"], errors="coerce")

    if "Region" in df.columns and region:
        df = df[df["Region"] == region]

    df = (
        df.dropna(subset=["Consumption"])
        .sort_values("Datetime")
        .set_index("Datetime")
    )

    return df


# ----------------------------------------------------------------------
# 2. Forecast of peak energy consumption
# ----------------------------------------------------------------------

def hourly_profile(df):
    """
    Build a basic seasonal profile: expected consumption for every
    (weekday, hour) combination, computed from the trailing window of data.
    """
    prof = (
        df.assign(weekday=df.index.dayofweek, hour=df.index.hour)
        .groupby(["weekday", "hour"])["Consumption"]
        .mean()
    )
    return prof


def forecast_peaks(df, days=7):
    """
    Predict the next `days` days hour-by-hour using the same weekday/hour
    average of the most recent known values, then report the daily peaks.
    Returns a dict with the full predicted series and the peak summary.
    """
    if df.empty:
        return {"forecast": [], "daily_peaks": []}

    freq = pd.infer_freq(df.index[:200])
    if not freq or freq == "D":
        freq = "h"

    # Last observed timestamp in data
    last_ts = df.index.max()
    if isinstance(last_ts, pd.Timestamp):
        start = last_ts + pd.Timedelta(hours=1)
    else:
        start = last_ts + pd.tseries.timedeltas.to_timedelta("1h")

    horizon = pd.date_range(start=start, periods=days * 24, freq=freq)

    prof = hourly_profile(df)

    pred = pd.Series(
        [prof.get((ts.dayofweek, ts.hour), np.nan) for ts in horizon],
        index=horizon,
        name="predicted_consumption",
    )
    pred = pred.fillna(df["Consumption"].mean())

    daily_peaks = (
        pred.groupby(pred.index.date)
        .max()
        .rename("predicted_peak")
        .round(2)
        .to_dict()
    )

    return {
        "forecast": pred,
        "daily_peaks": daily_peaks,
    }


# ----------------------------------------------------------------------
# 3. Anomaly detection
# ----------------------------------------------------------------------

def detect_anomalies(df, z_threshold=3.0, bottom_top=None):
    """
    Flags rows whose consumption deviates strongly from the trailing 96h
    rolling mean (scored as a z-score). Returns the anomalies sorted by
    extremity, plus summary counts.
    """
    if df.empty:
        return {"total_anomalies": 0, "anomalies": []}

    roll_mean = df["Consumption"].rolling(96, min_periods=48).mean()
    roll_std = df["Consumption"].rolling(96, min_periods=48).std()

    zscores = (df["Consumption"] - roll_mean) / roll_std.replace(0, np.nan)

    mask = zscores.abs() > z_threshold

    anomalies = pd.DataFrame(
        {
            "datetime": df.index[mask],
            "consumption": df["Consumption"].values[mask],
            "zscore": zscores.values[mask],
        }
    )

    top = anomalies.reindex(anomalies["zscore"].abs().sort_values().index[::-1])
    if bottom_top:
        top = top.head(bottom_top)

    return {
        "total_anomalies": int(len(anomalies)),
        "anomalies": top.to_dict("records"),
    }


# ----------------------------------------------------------------------
# 4. Recommendations + impact estimation
# ----------------------------------------------------------------------

def _hour_stats(df):
    """Per-hour mean consumption used by the rules engine."""
    return df.groupby(df.index.hour)["Consumption"].mean()


def estimate(kwh, rate):
    """Return cost saved for a given number of kWh."""
    return round(kwh * rate, 2)


def recommend_actions(df, rate=DEFAULT_RATE, anomalies=None):
    """
    Rule-based recommendations with estimated savings. Each action report
    contains the reasoning plus the kWh / cost saved vs. status quo.
    """
    if df.empty:
        return {"actions": []}

    mean_all = float(df["Consumption"].mean())
    hour_means = _hour_stats(df)

    peak_mean = float(hour_means[list(PEAK_HOURS)].mean()) if PEAK_HOURS.issubset(hour_means.index) else mean_all
    night_mean = float(hour_means[list(NIGHT_HOURS)].mean()) if NIGHT_HOURS.issubset(hour_means.index) else mean_all
    office_mean = float(hour_means[list(OFFICE_HOURS)].mean()) if OFFICE_HOURS.issubset(hour_means.index) else mean_all

    # Total consumption from each window over the whole history
    peak_kwh_total = float(df[df.index.hour.isin(PEAK_HOURS)]["Consumption"].sum())
    night_kwh_total = float(df[df.index.hour.isin(NIGHT_HOURS)]["Consumption"].sum())
    office_kwh_total = float(df[df.index.hour.isin(OFFICE_HOURS)]["Consumption"].sum())

    actions = []

    # -- Action A: shift non-essential load out of the evening peak ------
    if peak_mean > 1.15 * mean_all:
        shiftable = peak_kwh_total * 0.10          # assume 10% is shiftable
        actions.append(
            {
                "action": "Shift non-essential loads (EV charging, water heating, heavy machinery) out of the 17:00-21:00 peak window.",
                "why": (
                    f"Peak-window average ({peak_mean:,.0f} kWh/h) is more than 15% above the "
                    f"overall average ({mean_all:,.0f} kWh/h)."
                ),
                "estimated_kwh_saved": round(shiftable, 0),
                "estimated_cost_saved": estimate(shiftable, rate),
                "units": "kWh (over analysed history)",
            }
        )

    # -- Action B: cut idle (overnight / weekend) loads ------------------
    # Idle load is 'wasted' only if a big share of the night-time mean is
    # unavoidable; we assume 10% could be shed with scheduling.
    if night_mean > 0.35 * office_mean:
        idle = night_kwh_total * 0.05
        actions.append(
            {
                "action": "Power down idle equipment overnight (HVAC setback, servers, lighting) and schedule appliances off during 00:00-05:00.",
                "why": (
                    f"Overnight average ({night_mean:,.0f} kWh/h) is "
                    f"{night_mean / office_mean:.0%} of daytime average "
                    f"({office_mean:,.0f} kWh/h), indicating high idle load."
                ),
                "estimated_kwh_saved": round(idle, 0),
                "estimated_cost_saved": estimate(idle, rate),
                "units": "kWh (over analysed history)",
            }
        )

    # -- Action C: chase consumption spikes (uses anomaly output) ---------
    if anomalies and anomalies.get("total_anomalies", 0) > 0 and anomalies["anomalies"]:
        worst = max(anomalies["anomalies"], key=lambda a: a["zscore"])
        # Rough cost of repeating that spike once per month
        spike_cost = estimate(worst["consumption"], rate)
        actions.append(
            {
                "action": "Investigate and eliminate the cause of abnormal consumption spikes.",
                "why": (
                    f"Detected {anomalies['total_anomalies']:,} anomalies; the worst spike peaks at "
                    f"{worst['consumption']:,.0f} kWh on {worst['datetime']}."
                ),
                "estimated_kwh_saved": round(worst["consumption"], 0),
                "estimated_cost_saved": estimate(worst["consumption"], rate),
                "units": "kWh per avoided spike (one-time)",
            }
        )

    # -- Action D: trim the general office-hours baseline -----------------
    if office_mean > mean_all:
        trim = office_kwh_total * 0.05
        actions.append(
            {
                "action": "Trim the weekday daytime baseline (lighting, HVAC scheduling, occupancy controls).",
                "why": (
                    f"Business-hours average ({office_mean:,.0f} kWh/h) runs above the overall "
                    f"average ({mean_all:,.0f} kWh/h); a 5% trim is realistic with occupancy controls."
                ),
                "estimated_kwh_saved": round(trim, 0),
                "estimated_cost_saved": estimate(trim, rate),
                "units": "kWh (over analysed history)",
            }
        )

    if not actions:
        actions.append(
            {
                "action": "No major saving opportunities found; usage looks balanced.",
                "why": "All demand windows are close to the overall average.",
                "estimated_kwh_saved": 0,
                "estimated_cost_saved": 0.0,
                "units": "kWh",
            }
        )

    return {"actions": actions}


# ----------------------------------------------------------------------
# 5. Report
# ----------------------------------------------------------------------

def build_report(df, days=7, rate=DEFAULT_RATE, z_threshold=3.0, top=10):
    """Run the whole pipeline and return a human-readable report string."""
    lines = []
    add = lines.append

    add("=" * 72)
    add("  BASIC ENERGY OPTIMIZATION AGENT REPORT")
    add("=" * 72)
    add(f"  Data rows        : {len(df):,}")
    add(f"  Time range       : {df.index.min()}  ->  {df.index.max()}")
    add(f"  Regions          : {df['Region'].nunique() if 'Region' in df.columns else 1}")
    add(f"  Electricity rate : ${rate:.2f}/kWh")
    add("")

    # --- Summary stats --------------------------------------------------
    add("-- CONSUMPTION SUMMARY --")
    add(f"  Average hourly load : {df['Consumption'].mean():,.1f} kWh")
    add(f"  Peak hour ever      : {df['Consumption'].max():,.1f} kWh  ({df['Consumption'].idxmax()})")
    add("")

    # --- 1. Forecast ----------------------------------------------------
    add("-- 1. PEAK CONSUMPTION FORECAST (next {} days) --".format(days))
    forecast = forecast_peaks(df, days=days)
    hourly_entry = []

    for day, peak in list(forecast["daily_peaks"].items())[:days]:
        add(f"     {day}   predicted peak {peak:,.1f} kWh")
    add("")

    # --- 2. Anomalies ----------------------------------------------------
    add("-- 2. ANOMALY DETECTION (z-score > {:.1f}) --".format(z_threshold))
    anoms = detect_anomalies(df, z_threshold=z_threshold, bottom_top=top)
    add(f"     Total anomalies detected : {anoms['total_anomalies']:,} "
        f"({100 * anoms['total_anomalies'] / len(df):.2f}% of rows)")
    add("     Top outliers:")
    for a in anoms["anomalies"]:
        add(
            f"     - {a['datetime']}  {a['consumption']:>12,.1f} kWh  "
            f"(z={a['zscore']:+.1f})"
        )
    add("")

    # --- 3 & 4. Recommendations + impact --------------------------------
    add("-- 3 & 4. RECOMMENDATIONS & ESTIMATED IMPACT --")
    recs = recommend_actions(df, rate=rate, anomalies=anoms)
    for r in recs["actions"]:
        add(f"  [{'!' if r['action'] != 'No major saving opportunities found; usage looks balanced.' else ' '}] {r['action']}")
        add(f"      Why     : {r['why']}")
        add(f"      Impact  : ~{r['estimated_kwh_saved']:,.0f} {r['units']} "
            f"~ ${r['estimated_cost_saved']:,.2f}")
        add("")

    total_save = sum(a["estimated_cost_saved"] for a in recs["actions"])
    add(f"  TOTAL ESTIMATED SAVINGS (all actions combined):  ${total_save:,.2f}")
    add("=" * 72)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Basic Energy Optimization Agent")
    parser.add_argument(
        "data",
        nargs="?",
        default=str(DEFAULT_DATA),
        help="Path to processed CSV (default: project processed dataset)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Analyse a single region only, e.g. PJME (default: all regions)",
    )
    parser.add_argument("--days", type=int, default=7, help="Forecast horizon in days")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Electricity rate $/kWh")
    parser.add_argument("--z", type=float, default=3.0, help="Anomaly z-score threshold")
    parser.add_argument("--top", type=int, default=10, help="Max anomalies to print")

    args = parser.parse_args(argv)

    try:
        df = load_data(args.data, region=args.region)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(build_report(df, days=args.days, rate=args.rate, z_threshold=args.z, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())