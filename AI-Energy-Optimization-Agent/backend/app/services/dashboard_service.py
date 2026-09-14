from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.app.database.models import EnergyConsumption
from backend.app.services.cache import cached

ELECTRICITY_RATE = 8.0


def _compute_dashboard_summary(db: Session):

    total_records, total_consumption, average_consumption, peak_consumption = db.query(
        func.count(EnergyConsumption.id),
        func.sum(EnergyConsumption.consumption),
        func.avg(EnergyConsumption.consumption),
        func.max(EnergyConsumption.consumption),
    ).one()

    estimated_cost = (total_consumption or 0) * ELECTRICITY_RATE

    return {
        "total_consumption": round(total_consumption or 0, 2),
        "average_consumption": round(average_consumption or 0, 2),
        "peak_consumption": round(peak_consumption or 0, 2),
        "total_records": total_records,
        "estimated_cost": round(estimated_cost, 2)
    }


def get_dashboard_summary(db: Session):
    return cached(
        "dashboard_summary",
        lambda: _compute_dashboard_summary(db),
    )


def get_dashboard(db: Session):
    return get_dashboard_summary(db)