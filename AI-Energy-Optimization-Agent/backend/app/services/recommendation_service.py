from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import text

from backend.app.database.database import SessionLocal
from backend.app.services.cache import cached

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "ml" / "trained_models"

_MODEL_PATH = MODEL_DIR / "recommendation_model.joblib"
_SCALER_PATH = MODEL_DIR / "recommendation_scaler.joblib"
_ENCODER_PATH = MODEL_DIR / "region_encoder.joblib"

FEATURES = [
    "consumption",
    "region",
    "year",
    "month",
    "day",
    "hour",
    "weekend",
    "peak_hour",
]

# Every 10th row gives ~105k evenly-spread records instead of scoring all 1M.
# SQLite still scans the table once (fast C scan) but only ~10% transfers.
_SQL = text(
    """
    SELECT
        datetime,
        region,
        consumption,
        year,
        month,
        day,
        hour,
        CAST(weekend  AS INTEGER) AS weekend,
        CAST(peak_hour AS INTEGER) AS peak_hour
    FROM energy_consumption
    WHERE id % 10 = 0
    """
)

if not _MODEL_PATH.exists():
    raise FileNotFoundError(f"Recommendation model not found: {_MODEL_PATH}")
if not _SCALER_PATH.exists():
    raise FileNotFoundError(f"Recommendation scaler not found: {_SCALER_PATH}")
if not _ENCODER_PATH.exists():
    raise FileNotFoundError(f"Region encoder not found: {_ENCODER_PATH}")

model = joblib.load(_MODEL_PATH)
scaler = joblib.load(_SCALER_PATH)
encoder = joblib.load(_ENCODER_PATH)


def _compute_recommendations(top_n: int = 100):
    db = SessionLocal()
    try:
        with db.connection() as conn:
            df = pd.read_sql(_SQL, con=conn)
    finally:
        db.close()

    if df.empty:
        return {"total_records": 0, "recommendations": []}

    df["weekend"] = df["weekend"].astype(int)
    df["peak_hour"] = df["peak_hour"].astype(int)
    df["datetime"] = pd.to_datetime(df["datetime"])

    df.dropna(subset=FEATURES, inplace=True)
    if df.empty:
        return {"total_records": 0, "recommendations": []}

    df["region_encoded"] = encoder.transform(df["region"])

    X = df.copy()
    X["region"] = X["region_encoded"]
    X_scaled = scaler.transform(X[FEATURES])

    predictions = model.predict(X_scaled)

    df["recommended_consumption"] = predictions

    df["saving"] = df["consumption"] - df["recommended_consumption"]

    df["saving_percent"] = (df["saving"] / df["consumption"]) * 100

    sample_size = len(df)

    def priority(x):
        if x >= 15:
            return "Critical"
        elif x >= 10:
            return "High"
        elif x >= 5:
            return "Medium"
        else:
            return "Low"

    df = (
        df
        .nlargest(top_n, "saving_percent")
        .sort_values("saving_percent", ascending=False)
    )

    df["priority"] = df["saving_percent"].apply(priority)

    recommendations = [
        {
            "datetime": str(row.datetime),
            "region": str(row.region),
            "current_consumption": round(float(row.consumption), 2),
            "recommended_consumption": round(float(row.recommended_consumption), 2),
            "saving": round(float(row.saving), 2),
            "saving_percent": round(float(row.saving_percent), 2),
            "priority": row.priority,
            "recommendation":
                "Reduce HVAC and shift non-essential load during peak hours."
                if row.priority in ["Critical", "High"]
                else
                "Current energy usage is acceptable.",
        }
        for row in df.itertuples(index=False)
    ]

    return {
        "total_records": sample_size,
        "recommendations": recommendations,
    }


def detect_recommendations(db, top_n: int = 100):
    return cached(
        "recommendations",
        lambda: _compute_recommendations(top_n),
    )