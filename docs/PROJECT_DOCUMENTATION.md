# Cognizant Mourya — Project Documentation

**AI Energy Optimization Agent · EnergyIQ · Dataset & Agents**

> Repository: `https://github.com/mouryakiran/cognizant-mourya.git`

---

## 1. Overview

**Cognizant Mourya** is a collection of energy-analytics projects built for the
**Smart Energy Consumption Optimization** use case (Smart India Hackathon,
proposed by Cognizant). The aim is to help buildings, campuses, factories, and
grids **reduce energy usage and cost without disrupting critical operations**,
because energy demand varies by time, occupancy, weather, and usage pattern.

The repository contains three working projects plus a reusable real-world dataset:

| Item | Description |
| --- | --- |
| `AI-Energy-Optimization-Agent/` | Full-stack platform — FastAPI backend, ML models (forecast / anomaly / recommendation), React dashboard, SQLite |
| `energyiq/` | End-to-end energy intelligence platform — XGBoost forecasting, LP battery/load optimisation, fast rule-based recommendations, Chart.js dashboard |
| `basic_energy_agent/` | Minimal single-file CLI agent (pandas + numpy only) that forecasts peaks, finds anomalies, and recommends actions |
| `processed_energy_data.csv` | 68 MB real pre-processed PJM energy dataset (12 regions, hourly) |

---

## 2. The Problem

Energy usage in buildings and facilities is **time-varying and hard to predict**.
Uncontrolled peak demand leads to:

- High peak-demand / capacity charges
- Inefficient time-of-use tariffs
- Overtime / weekend and holiday waste
- Undersized or oversized storage decisions
- Missed shiftable-load opportunities

**Goal:** an agent that forecasts peak consumption, detects anomalies, and
recommends (with estimated impact) energy-saving actions — usable by both
end-users and facility operators.

---

## 3. Architecture (high level)

```
Raw / generated hourly energy data
                │
                ▼
        Feature engineering
   (calendar / cyclical / lags / rolling / anomaly flags)
                │
        ┌───────┴────────┐
        ▼                ▼
  Forecast model     Anomaly detector
  (recursive,        (z-score / IQR /
   confidence bands)  seasonal)
        │                │
        └───────┬────────┘
                ▼
      Optimisation (LP/scipy)
   battery charge/discharge +
   shiftable load curtailment
   → schedule + $ savings
                │
                ▼
      Recommendation engine
   (prioritised, with savings)
                │
                ▼
       FastAPI backend (/api/...)
                │
                ▼
       React / Chart.js dashboard
```

Both `AI-Energy-Optimization-Agent` and `energyiq` follow this same pipeline
shape; they differ in stack depth and feature richness.

---

## 4. AI-Energy-Optimization-Agent (full-stack)

An end-to-end analytics platform combining ML, agentic workflows, a FastAPI
service layer, and a responsive React dashboard, storing data in SQLite
(`energy.db`).

### 4.1 Technology stack

| Layer | Technology |
| --- | --- |
| Backend | Python, FastAPI, Uvicorn, SQLAlchemy, SQLite |
| ML | Pandas, NumPy, Scikit-learn, Joblib |
| Frontend | React (Vite), Tailwind CSS, React Router, Axios, Recharts, Framer Motion, React Hook Form, Lucide |

### 4.2 Key capabilities

- Interactive energy dashboard
- 24-hour energy-consumption forecasting
- ML anomaly detection
- AI-generated energy optimisation recommendations
- Historical analytics with search, filters, pagination, and CSV export
- Automated energy reports
- Conversational AI assistant (natural-language queries via the agent)

### 4.3 API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | API health |
| `GET` | `/dashboard/summary` | Consumption totals, average, peak, count |
| `GET` | `/forecast/` | Next 24 predicted consumption values |
| `GET` | `/anomaly/` | Detected anomalies + scores |
| `GET` | `/recommendation/` | Ranked optimisation recommendations |
| `GET` | `/history/` | Latest 100 consumption records |
| `GET` | `/report/` | Dataset + operational summary |
| `POST` | `/agent/` | Natural-language query interface |

### 4.4 ML pipelines

- **Forecast** — recursive multi-step forecasting with time-based features.
- **Anomaly** — scaled Isolation Forest over the data.
- **Recommendation** — scaled Random Forest with region encoding for savings
  estimates, savings, and priority.

Trained artifacts are saved under `ml/trained_models/`.

---

## 5. EnergyIQ (end-to-end optimisation platform)

EnergyIQ predicts demand, detects anomalies, and computes an **optimal battery +
load-shifting schedule** that cuts energy cost and peak demand, surfaced via a
live dashboard.

### 5.1 Highlights

- **Dataset pipeline** — loader/validator for the exact 44-column PJM schema;
  self-contained generator reproduces a full 12-region hourly dataset when the
  large real file is unavailable.
- **ML forecasting** — XGBoost (`hist` trees) on 1.26M rows; calendar, lag
  (1/2/3/24/48/168h) and rolling (3/6/24/168h) features; **MAPE ≈ 0.89%**,
  RMSE ≈ 84 MW; recursive multi-step forecasts with growing confidence bands.
- **Anomaly detection** — seasonal z-score + IQR rules over a 168h baseline.
- **Optimisation** — scipy LP (HiGHS) minimises energy cost + peak-demand
  charge: battery charge/discharge (SOC bounds, efficiency), penalised
  curtailment of shiftable loads → per-hour schedule + baseline-vs-optimised
  bill (example: **~5% bill reduction** on a 15 MW campus over 48h, SOC 10% →
  100%).
- **Recommendations** — prioritised rule engine (arbitrage, demand response,
  storage cycling, precooling/setback, holiday/weekend mode, anomaly review)
  with estimated savings.
- **API** — FastAPI at `/api/...` with OpenAPI docs at `/docs`.
- **UI** — self-contained Chart.js dashboard: forecast, optimised schedule,
  KPIs, recommendations, anomalies.
- **Tests** — 38 pytest tests across data, forecasting, optimisation,
  recommendations, and the live API.

### 5.2 Optimisation formulation

The linear programme chooses, hour by hour, battery discharge/charge and load
curtailment to minimise:

```
Σ rate(t)·net(t) + demand_charge·peak
```

with battery SOC:

```
SOC(t+1) = SOC(t) + eff·charge(t) − discharge(t)
```

subject to capacity, rate, and final-SOC bounds. Curtailment is penalised above
the top energy rate so shiftable loads are only deferred when truly valuable.

`site_scale_mw` rescales the grid-level profile to a single site (e.g. an 8 MW
factory or 2 MW store) so optimisation results reflect facility economics.

### 5.3 Layout

```
energyiq/
├── energyiq/            # core package (config, dataset, anomalies,
│                        #   forecast, optimize, pricing, recommend)
├── api/                 # FastAPI app + schemas
├── web/                 # dashboard (index.html, app.js, style.css)
├── scripts/             # generate_data.py, train.py, run_api.py
├── tests/               # pytest suite
├── data/processed/      # dataset
├── models/              # trained model artifacts
└── requirements.txt
```

### 5.4 Key endpoints

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

---

## 6. Basic Energy Agent (lightweight CLI)

`basic_energy_agent/energy_agent.py` is a **dependency-light** implementation
(only pandas + numpy) of the same idea — a good on-boarding / minimal version.

It performs, in one report:

1. **Peak consumption forecast** — next-N-days hourly prediction using a
   seasonal (weekday × hour) profile of recent values, plus daily peaks.
2. **Anomaly detection** — statistical outlier detection (z-score) against a
   trailing 96-hour window, with top anomalies.
3. **Recommendations** — rule-based actions (shift evening-peak loads, cut idle
   overnight loads, investigate spikes, trim weekday baseline) with kWh and
   cost-savings estimates.
4. **Impact estimates** — per-action and total $ savings at a configurable
   electricity rate (default `$0.12/kWh`).

### CLI usage

```bash
python energy_agent.py [data.csv] [--region PJME] [--days 7] [--rate 0.12]
```

### 6.1 Example recommendations produced

- **Shift non-essential loads out of the 17:00–21:00 peak window** — when the
  peak-window average is >15% above the overall average.
- **Power down idle equipment overnight** — when overnight load is high relative
  to daytime (HVAC setback, servers, lighting scheduling).
- **Investigate anomaly spikes** — the worst detected z-score event, with
  estimated one-time cost.
- **Trim the weekday daytime baseline** — 5% target via occupancy controls when
  business-hours usage runs high.

---

## 7. Dataset — `processed_energy_data.csv`

- **Size:** 68 MB real pre-processed data (with duplicates of the interrupted
  PJM download present in parent folders).
- **Granularity:** hourly consumption.\n
- **Scope:** multiple PJM regions (regional utilities).
- **Use:** loaded by the energy agent and platforms to demonstrate the
  forecasting / anomaly / recommendation / optimisation pipelines on real data.

The EnergyIQ generator also ships a self-contained path that reproduces the
exact 44-column PJM schema (12 regions, 2005–2016, ~1.26M rows) so the platform
works even without the original large file.

---

## 8. How to run

### 8.1 AI-Energy-Optimization-Agent

```powershell
# backend
python -m venv venv
pip install -r backend/requirements.txt
python -m backend.app.database.load_data    # build energy.db
uvicorn backend.main:app --reload --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev    # http://localhost:5173
```

API docs: `http://localhost:8000/docs`

### 8.2 EnergyIQ

```bash
pip install -r requirements.txt
python scripts/generate_data.py    # optional full 12-region dataset
python scripts/train.py            # train forecast model (~40 s)
python scripts/run_api.py          # server → http://127.0.0.1:8000
python -m pytest tests -q          # tests
```

### 8.3 Basic Energy Agent

```bash
python basic_energy_agent/energy_agent.py processed_energy_data.csv --region PJME
```

---

## 9. Scope & limitations

- The applications are **analytics / decision-support**; they do not directly
  control building equipment or execute energy-saving actions.
- Recommendation savings are **model estimates** and should be sanity-checked
  against operational constraints before implementation.
- Frontend dashboards display only what backend contracts expose; advanced
  metrics (cost impact, carbon, confidence intervals, PDF reports, server-side
  history filters) require the corresponding backend fields/endpoints to be
  surfaced as authoritative.

---

## 10. What was pushed

All three projects, documentation, and the real dataset were committed and
pushed to `github.com/mouryakiran/cognizant-mourya` (branch `main`).

Excluded from the repository (large/junk, per `.gitignore`):

- `venv/`, `frontend/node_modules/` (147 MB), `ml/trained_models/` (775 MB)
- `energy.db` (82 MB), `data/raw` + `data/processed`
- `energyiq.zip` (181 MB — over GitHub's 100 MB file limit) and the interrupted
  `.crdownload` downloads

---

## 11. Tech-stack summary

| Area | Tools |
| --- | --- |
| Languages | Python, JavaScript |
| Backend | FastAPI, Uvicorn, SQLAlchemy, SQLite |
| ML/DL | Scikit-learn, XGBoost, Pandas, NumPy, scipy (HiGHS/LP), Joblib |
| Frontend | React (Vite), Chart.js, Recharts, Tailwind CSS, Axios |
| Testing | pytest (38 tests in EnergyIQ; plus backend/agent suites) |
| Delivery | Git + GitHub, Markdown docs, live dashboards |
