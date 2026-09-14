from backend.app.services.anomaly_service import detect_anomalies


def test_detect_anomalies_returns_summary_and_anomaly_rows():
    result = detect_anomalies()

    assert set(result) == {"total_records", "total_anomalies", "anomalies"}
    assert result["total_records"] > 0
    assert 0 <= result["total_anomalies"] <= result["total_records"]

    for record in result["anomalies"]:
        assert set(record) == {"datetime", "region", "consumption", "anomaly_score"}
        assert isinstance(record["datetime"], str)
        assert isinstance(record["consumption"], float)
        assert isinstance(record["anomaly_score"], float)