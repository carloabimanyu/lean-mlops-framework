import os
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

log = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline",
]


def _load_reference_data() -> pd.DataFrame:
    wine = load_wine()
    return pd.DataFrame(wine.data, columns=FEATURE_COLUMNS)


def _load_current_data(outputs_base: Path) -> pd.DataFrame | None:
    outputs_dir = outputs_base / "wine_clustering" / "outputs"
    csvs = sorted(outputs_dir.glob("batch_*.csv")) if outputs_dir.exists() else []
    if not csvs:
        log.warning("No batch output found for monitoring — skipping")
        return None
    return pd.read_csv(csvs[-1])[FEATURE_COLUMNS]


def run_monitoring() -> dict | None:
    outputs_base = Path(os.getenv("OUTPUTS_BASE", "/models"))
    reports_dir = Path(os.getenv("REPORTS_DIR", "/reports"))
    drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.5"))

    reference = _load_reference_data()
    current = _load_current_data(outputs_base)

    if current is None:
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))
    log.info(f"Drift report saved: {report_path}")

    result = report.as_dict()
    drift_score = result["metrics"][0]["result"].get("dataset_drift_share")
    if drift_score is None:
        log.warning("dataset_drift_share not found in Evidently result — check Evidently version")
        return {"drift_detected": False, "drift_score": 0.0}
    drift_score = float(drift_score)
    drift_detected = drift_score >= drift_threshold

    log.info(f"Drift score: {drift_score:.3f} (threshold: {drift_threshold}) — detected: {drift_detected}")

    if drift_detected:
        try:
            from alert import send_alert
            send_alert(drift_score, str(report_path))
        except Exception as e:
            log.error(f"Alert failed: {e}", exc_info=True)
            raise

    return {"drift_detected": drift_detected, "drift_score": drift_score}


if __name__ == "__main__":
    run_monitoring()
