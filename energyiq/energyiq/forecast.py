"""ML demand forecasting.

A gradient-boosting regressor (XGBoost, ``hist`` tree method) predicts hourly
load from calendar, lag and rolling features. Multi-step forecasts are produced
recursively: each step uses previously generated values where an actual lag or
rolling statistic would need a future point.

Artifacts (model + metadata) are persisted with joblib so the API can run
without retraining.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from . import config
from .dataset import SCHEMA_COLUMNS, load_dataset

log = logging.getLogger("energyiq.forecast")


class ForecastError(Exception):
    """Raised when a forecast cannot be produced."""


def _regions_from_df(df: pd.DataFrame) -> list[str]:
    if "region" in df.columns:
        return sorted(df["region"].unique().tolist())
    return list(config.REGIONS)


def _one_hot_regions(df: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Return df with ``region`` replaced by one-hot columns in stable order."""
    known = {r for r in regions}
    missing = sorted(set(df["region"].unique()) - known)
    if missing:
        raise ForecastError(f"Unknown region(s): {missing}. Known: {known}")
    cat = pd.Categorical(df["region"], categories=[r for r in regions])
    dummies = pd.get_dummies(cat, prefix="region")
    out = df.drop(columns=["region"]).reset_index(drop=True)
    return pd.concat([out, dummies.astype(float)], axis=1)


class ForecastEngine:
    """Trains, persists and runs the demand forecasting model."""

    def __init__(self, model_path: Path | None = None, meta_path: Path | None = None) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.meta_path = Path(meta_path or config.META_PATH)
        self.model: XGBRegressor | None = None
        self.meta: dict[str, Any] = {}
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(
        self,
        df: pd.DataFrame | None = None,
        test_tail_hours: int = 30 * 24,
        max_train_rows: int | None = 300_000,
        seed: int = 7,
        save: bool = True,
    ) -> dict[str, Any]:
        data = df if df is not None else load_dataset()
        data = data.copy()
        data["datetime"] = pd.to_datetime(data["datetime"])

        # ---- build training matrix ----------------------------------- #
        regions = _regions_from_df(data)
        feat_cols = [c for c in config.FEATURE_COLS if c != "region"]
        data = data.dropna(subset=feat_cols + [config.TARGET])

        # split by time at the last test_tail_hours
        max_ts = data["datetime"].max()
        split_ts = max_ts - pd.Timedelta(hours=test_tail_hours)
        test: pd.DataFrame = data[data["datetime"] > split_ts]
        train: pd.DataFrame = data[data["datetime"] <= split_ts]

        if max_train_rows and len(train) > max_train_rows:
            train = train.sample(n=max_train_rows, random_state=seed).sort_values(
                ["region", "datetime"]
            )
        log.info("training rows=%s test rows=%s regions=%d", len(train), len(test), len(regions))

        X_train = _one_hot_regions(train[config.FEATURE_COLS], regions)
        y_train = train[config.TARGET].to_numpy(dtype=float)
        X_test = _one_hot_regions(test[config.FEATURE_COLS], regions)
        y_test = test[config.TARGET].to_numpy(dtype=float)

        model = XGBRegressor(
            n_estimators=900,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.8,
            tree_method="hist",
            early_stopping_rounds=40,
            random_state=seed,
            n_jobs=-1,
        )
        t0 = time.time()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        elapsed = time.time() - t0

        y_pred = model.predict(X_test)
        metrics = {
            "mae_mw": float(mean_absolute_error(y_test, y_pred)),
            "rmse_mw": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mape_pct": float(
                np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-6))) * 100
            ),
            "test_rows": int(len(y_test)),
            "train_rows": int(len(train)),
            "train_seconds": round(elapsed, 2),
        }
        log.info("forecast metrics: %s", metrics)

        # per-region error (for confidence bands / UI)
        per_region: dict[str, dict] = {}
        for r in regions:
            mask = test["region"] == r
            if mask.sum() == 0:
                continue
            within = _one_hot_regions(test[mask][config.FEATURE_COLS], regions)
            p = model.predict(within)
            per_region[r] = {
                "mae_mw": float(mean_absolute_error(test[mask][config.TARGET], p)),
                "rmse_mw": float(np.sqrt(mean_squared_error(test[mask][config.TARGET], p))),
            }

        self.meta = {
            "regions": regions,
            "feature_columns": config.FEATURE_COLS,
            "one_hot_columns": list(X_train.columns),
            "target": config.TARGET,
            "metrics": metrics,
            "per_region_metrics": per_region,
            "trained_at": pd.Timestamp.now().isoformat(),
            "best_iteration": int(model.best_iteration or model.n_estimators),
        }
        self.model = model

        if save:
            self.save()
        return metrics

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, model_path: Path | None = None, meta_path: Path | None = None) -> None:
        model_path = Path(model_path or self.model_path)
        meta_path = Path(meta_path or self.meta_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is None:
            raise ForecastError("No trained model to save.")
        joblib.dump(self.model, model_path)
        meta_path.write_text(json.dumps(self.meta, indent=2))
        log.info("saved model -> %s, meta -> %s", model_path, meta_path)

    def load(self, model_path: Path | None = None, meta_path: Path | None = None) -> "ForecastEngine":
        model_path = Path(model_path or self.model_path)
        meta_path = Path(meta_path or self.meta_path)
        if not model_path.exists() or not meta_path.exists():
            raise ForecastError(
                "Forecast model not found. Run `python scripts/train.py` first "
                f"(expected {model_path} and {meta_path})."
            )
        self.model = joblib.load(model_path)
        self.meta = json.loads(meta_path.read_text())
        log.info("loaded model %s (best_iteration=%s)", model_path, self.meta.get("best_iteration"))
        return self

    # ------------------------------------------------------------------ #
    # Forecasting
    # ------------------------------------------------------------------ #
    def forecast(
        self,
        region: str,
        horizon: int | None = None,
        df: pd.DataFrame | None = None,
        start: str | None = None,
    ) -> dict[str, Any]:
        """Recursive multi-step forecast for one region.

        Args:
            region: region code (see config.REGIONS).
            horizon: number of hourly steps (default config.FORECAST_HORIZON).
            df: dataset; if None the engine uses its cached/last-loaded dataset.
            start: ISO timestamp for the last *actual* observation. When omitted,
                the most recent data point is used as the anchor.
        """
        if self.model is None:
            self.load()
        horizon = horizon or config.FORECAST_HORIZON
        if horizon < 1:
            raise ForecastError("horizon must be >= 1")
        if horizon > 24 * 14:
            raise ForecastError("horizon too large (max 336h)")

        data = df if df is not None else self._ensure_data()
        regions = self.meta.get("regions", config.REGIONS)
        if region not in regions:
            raise ForecastError(
                f"Region '{region}' not in model. Available: {sorted(regions)}"
            )

        sub = data[data["region"] == region].copy()
        sub["datetime"] = pd.to_datetime(sub["datetime"])
        sub = sub.sort_values("datetime").drop_duplicates("datetime", keep="last")

        if start is not None:
            anchor = pd.Timestamp(start)
            if anchor < sub["datetime"].min() or anchor > sub["datetime"].max():
                raise ForecastError(f"start {start} out of range for region {region}")
            history = sub[sub["datetime"] <= anchor]
        else:
            anchor = sub["datetime"].max()
            history = sub[sub["datetime"] <= anchor]

        return self._recursive_forecast(region, horizon, history, anchor)

    def _ensure_data(self) -> pd.DataFrame:
        if self._df is None:
            self._df = load_dataset()
        return self._df

    def _recursive_forecast(
        self,
        region: str,
        horizon: int,
        history: pd.DataFrame,
        anchor: pd.Timestamp,
    ) -> dict[str, Any]:
        """Predict horizon points after ``anchor`` using effective series Y."""
        h_feat = [c for c in config.FEATURE_COLS if c != "region"]
        regions = self.meta.get("regions", config.REGIONS)

        cons = history["consumption_mw"].to_numpy(dtype=float)
        regional_err = self.meta.get("per_region_metrics", {}).get(region, {}).get("rmse_mw")
        base_err = float(regional_err or self.meta.get("metrics", {}).get("rmse_mw", 1.0))

        hist_len = len(cons)
        hist_lookup: dict[pd.Timestamp, float] = {
            ts: float(c) for ts, c in zip(history["datetime"], cons)
        }

        forecast_ts: list[pd.Timestamp] = []
        forecast_vals: list[float] = []
        eff: list[float] = list(cons)

        for step in range(1, horizon + 1):
            t = anchor + pd.Timedelta(hours=step)
            if len(eff) < 168:
                raise ForecastError("history too short for forecast (need >= 168h)")

            known_ts = hist_lookup
            known_vals = dict(zip(forecast_ts, forecast_vals))
            feats = self._flat_features_for(region, t, eff, hist_len, known_ts, known_vals)
            row_df = pd.DataFrame([{c: feats.get(c, 0.0) for c in config.FEATURE_COLS}])
            X = _one_hot_regions(row_df, regions)
            pred = float(max(self.model.predict(X)[0], 0.0))
            forecast_ts.append(t)
            forecast_vals.append(pred)
            eff.append(pred)

        # confidence band ~ residual std growing with sqrt(step)
        ser = pd.Series(forecast_vals)
        steps = np.arange(1, horizon + 1)
        std = base_err * np.sqrt(steps)
        dates = [t.isoformat() for t in forecast_ts]
        result = {
            "region": region,
            "horizon": horizon,
            "anchor": anchor.isoformat(),
            "model_rmse_mw": round(base_err, 2),
            "forecast": [
                {
                    "datetime": d,
                    "forecast_mw": round(v, 2),
                    "upper_mw": round(v + 1.96 * s, 2),
                    "lower_mw": round(max(v - 1.96 * s, 0.0), 2),
                }
                for d, v, s in zip(dates, ser.round(2), std)
            ],
        }
        return result

    def _flat_features_for(
        self,
        region: str,
        t: pd.Timestamp,
        eff: list[float],
        hist_len: int,
        hist_lookup: dict[pd.Timestamp, float],
        forecast_lookup: dict[pd.Timestamp, float],
    ) -> dict[str, float]:
        """Compute a single feature row for timestamp ``t`` using effective series.

        ``eff`` contains consumption for all instants at-or-before t-1h
        (actual history + forecast values). Lookups resolve lag values from
        actual history first, then from generated forecast values.
        """

        def val(t_delta_hours: int) -> float:
            target = t - pd.Timedelta(hours=t_delta_hours)
            if target in hist_lookup:
                return hist_lookup[target]
            if target in forecast_lookup:
                return forecast_lookup[target]
            raise ForecastError(f"missing lag value for {target}")

        # ---- rolling statistics from effective series ------------------ #
        tail3 = eff[-3:]
        tail6 = eff[-6:]
        tail24 = eff[-24:]
        tail168 = eff[-168:]

        base = {
            "hour": float(t.hour),
            "dow": float(t.dayofweek),
            "day": float(t.day),
            "month": float(t.month),
            "year": float(t.year),
            "dayofweek": float(t.dayofweek),
            "dayofyear": float(t.dayofyear),
            "weekofyear": float(t.isocalendar()[1]),
            "is_weekend": float(t.dayofweek >= 5),
            "quarter": float(t.quarter),
            "hour_sin": float(np.sin(2 * np.pi * t.hour / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * t.hour / 24.0)),
            "dow_sin": float(np.sin(2 * np.pi * t.dayofweek / 7.0)),
            "dow_cos": float(np.cos(2 * np.pi * t.dayofweek / 7.0)),
            "month_sin": float(np.sin(2 * np.pi * (t.month - 1) / 12.0)),
            "month_cos": float(np.cos(2 * np.pi * (t.month - 1) / 12.0)),
            "is_holiday": float((t.month, t.day) in config.HOLIDAYS),
            "lag_1h": val(1), "lag_2h": val(2), "lag_3h": val(3),
            "lag_24h": val(24), "lag_48h": val(48), "lag_168h": val(168),
            "rolling_mean_3h": float(np.mean(tail3)),
            "rolling_std_3h": float(np.std(tail3)),
            "rolling_max_3h": float(np.max(tail3)),
            "rolling_min_3h": float(np.min(tail3)),
            "rolling_mean_6h": float(np.mean(tail6)),
            "rolling_std_6h": float(np.std(tail6)),
            "rolling_max_6h": float(np.max(tail6)),
            "rolling_min_6h": float(np.min(tail6)),
            "rolling_mean_24h": float(np.mean(tail24)),
            "rolling_std_24h": float(np.std(tail24)),
            "rolling_max_24h": float(np.max(tail24)),
            "rolling_min_24h": float(np.min(tail24)),
            "rolling_mean_168h": float(np.mean(tail168)),
            "rolling_std_168h": float(np.std(tail168)),
            # Future anomaly features are unknown: default to 0.
            "anomaly_iqr": 0.0,
            "seasonal_zscore": float(
                (eff[-1] - np.mean(tail168)) / np.std(tail168) if np.std(tail168) > 0 else 0.0
            ),
            "anomaly_seasonal_zscore": 0.0,
            "is_anomaly": 0.0,
        }
        return {**base, "region": region}

    def evaluate(
        self,
        region: str | None = None,
        horizon: int = 24,
        df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Hour-ahead style evaluation on held-out data."""
        data = df if df is not None else self._ensure_data()
        data = data.copy()
        data["datetime"] = pd.to_datetime(data["datetime"])

        regions = [region] if region else self.meta.get("regions", config.REGIONS)
        outs: dict[str, dict] = {}
        for r in regions:
            sub = data[data["region"] == r].sort_values("datetime")
            anchor = sub["datetime"].max() - pd.Timedelta(hours=horizon)
            try:
                res = self.forecast(r, horizon=horizon, df=data, start=anchor)
                actuals = sub[sub["datetime"] > anchor]["consumption_mw"].to_numpy()[:horizon]
                preds = np.array([f["forecast_mw"] for f in res["forecast"]])

                n = min(len(actuals), len(preds))
                if n == 0:
                    continue
                a, p = actuals[:n], preds[:n]
                outs[r] = {
                    "mae_mw": round(float(np.mean(np.abs(a - p))), 2),
                    "rmse_mw": round(float(np.sqrt(np.mean((a - p) ** 2))), 2),
                    "mape_pct": round(float(np.mean(np.abs((a - p) / np.maximum(a, 1e-6)))) * 100, 2),
                    "samples": int(n),
                }
            except ForecastError:
                continue
        return {"horizon": horizon, "results": outs}