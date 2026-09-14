# EnergyIQ - Smart Energy Consumption Forecasting & Optimisation Platform

Buildings, campuses, and factories need to reduce energy consumption and optimise
costs without disrupting critical operations. **EnergyIQ** predicts demand
(which varies with time, occupancy, weather and usage patterns), detects
anomalies, and computes an **optimal battery + load-shifting schedule** that
cuts energy cost and peak demand — surfacing everything in a live dashboard.

```
                       ┌────────────  pre-processed dataset (44-col PJM schema)
 raw hourly demand ──►│ feature engineering (calendar + cyclical + lags +
                       │   rolling stats + anomaly flags)
                       ▼                        ┌──────────────────────────┐
              XGBoost forecaster  ────────────► │ Forecast (0-336h ahead)   │
              recursive multi-step               │  with confidence bands    │
                       │                          └────────────┬─────────────┘
                       ▼                                         ▼
             scipy LP optimiser  ────────────►  battery charge/discharge +
                  (cost + demand                  load-shift schedule + $
                       │                          savings vs baseline
                       ▼
          Rule-based recommendation engine ─► actionable, prioritised steps
                       ▼
        FastAPI (/api/...)  ◄──  Dashboard UI (Chart.js)
```

## Features

| Component | Detail |
| --- | --- |
| **Dataset pipeline** | Loader + validator for the exact 44-column schema of the pre-processed PJM energy dataset; self-contained generator produces a full 12-region hourly dataset when the (large) real file is unavailable |
| **ML forecasting** | XGBoost (`hist` trees) trained on 1.26M rows; calendar, lag (1/2/3/24/48/168h) and rolling (3/6/24/168h) features; **MAPE ~0.89%**, RMSE ~84 MW; recursive multi-step forecasts with growing confidence bands |
| **Anomaly detection** | Seasonal z-score + IQR rules over the 168h baseline |
| **Optimisation** | scipy LP (HiGHS) minimises energy cost + peak-demand charge: battery charge/discharge (SOC bounds, efficiency), penalised curtailment of shiftable loads → per-hour schedule + baseline-vs-optimised bill |
| **Recommendations** | Prioritised rule engine (arbitrage, demand response, storage cycling, precooling/setback, holiday/weekend mode, anomaly review) with estimated savings |
| **API** | FastAPI at `/api/...` with OpenAPI docs at `/docs` |
| **UI** | Self-contained dashboard (Chart.js): forecast, optimised schedule, KPIs, recommendations, anomalies |
| **Tests** | 38 pytest tests covering data, forecasting, optimisation, recommendations and the live API |

## Project layout

```
energyiq/
├── energyiq/            # core package
│   ├── config.py        # paths, tariff, battery, model settings
│   ├── dataset.py       # loader, validator, feature engineering, generator
│   ├── anomalies.py     # anomaly detection + summary
│   ├── forecast.py      # XGBoost training + recursive forecasting
│   ├── optimize.py      # LP cost/peak optimisation
│   ├── pricing.py       # time-of-use tariff + billing
│   └── recommend.py     # recommendation engine
├── api/
│   ├── main.py          # FastAPI application
│   └── schemas.py       # request models
├── web/                 # dashboard (index.html, app.js, style.css)
├── scripts/
│   ├── generate_data.py # build the full dataset
│   ├── train.py         # train + persist the forecast model
│   └── run_api.py       # start uvicorn
├── tests/               # pytests (data, forecast, optimize, recommend, api)
├── data/processed/      # dataset (generated / real / sample)
├── models/              # trained model artifacts
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. (optional, ~2 min) generate the full 12-region dataset (2005-2016, 1.26M rows)
python scripts/generate_data.py

# 2. train the forecast model (~40 s)
python scripts/train.py

# 3. start the server → http://127.0.0.1:8000
python scripts/run_api.py

# 4. run the test-suite
python -m pytest tests -q
```

A real dataset from the interrupted download (`data/processed/processed_energy_data.csv`)
is used automatically when present; the recovered real sample is at
`data/processed/sample_real_processed.csv`.

### Key API endpoints

| Endpoint | Description |
| --- | --- |
| `GET /health` | liveness |
| `GET /api/regions` | regions + model metrics |
| `GET /api/dataset/info` | dataset stats + schema check |
| `POST /api/forecast` | recursive multi-step forecast (`region`, `horizon`, `site_scale_mw`) |
| `POST /api/optimize` | battery/load-shift schedule + savings |
| `POST /api/recommendations` | prioritised energy-saving recommendations |
| `GET /api/anomalies` | detected anomaly events |
| `GET /api/summary` | KPI + bill for a region/window |

`site_scale_mw` rescales the grid-level profile to a single site (e.g. an 8 MW
factory or 2 MW store) so optimisation results reflect facility economics.

## How optimisation works

The linear programme chooses, hour by hour, battery discharge/charge and load
curtailment to minimise `Σ rate(t)·net(t) + demand_charge·peak`. Battery state
of charge follows `SOC(t+1) = SOC(t) + eff·charge(t) − discharge(t)` with
capacity, rate and final-SOC bounds; curtailment is penalised above the top
energy rate so shiftable loads are only deferred when truly valuable. Example
output on a 15 MW campus over 48 h: **~5% bill reduction** with SOC 10% → 100%.

## Notes

* The four `Unconfirmed *.crdownload.zip` files in the parent folder are but
  interrupted Chrome downloads of the same pre-processed dataset (475 MB CSV,
  only ~5% was captured). The schema was recovered from the partial stream and
  a real sample is bundled; the generator reproduces the exact schema so the
  platform is fully functional without the original file.