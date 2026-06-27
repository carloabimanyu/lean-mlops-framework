import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.datasets import load_wine

sys.path.insert(0, "/models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INTERVAL_MINUTES = int(os.getenv("BATCH_SCHEDULE_INTERVAL_MINUTES", "60"))
OUTPUTS_BASE = Path("/models")


def run_wine_clustering_batch():
    log.info("Starting wine_clustering batch inference")
    from models.wine_clustering.inference import predict

    wine = load_wine()
    # Simulate new production data: add small noise to reference data
    rng = np.random.default_rng(seed=int(datetime.now().timestamp()))
    data = wine.data + rng.normal(0, 0.1, wine.data.size).reshape(wine.data.shape)

    feature_names = wine.feature_names
    results = []
    for row in data:
        sample = dict(zip(feature_names, row.tolist()))
        # wine feature names use spaces — map to inference schema names
        mapped = {
            "alcohol": sample["alcohol"],
            "malic_acid": sample["malic_acid"],
            "ash": sample["ash"],
            "alcalinity_of_ash": sample["alcalinity_of_ash"],
            "magnesium": sample["magnesium"],
            "total_phenols": sample["total_phenols"],
            "flavanoids": sample["flavanoids"],
            "nonflavanoid_phenols": sample["nonflavanoid_phenols"],
            "proanthocyanins": sample["proanthocyanins"],
            "color_intensity": sample["color_intensity"],
            "hue": sample["hue"],
            "od280_od315": sample["od280/od315_of_diluted_wines"],
            "proline": sample["proline"],
        }
        result = predict(mapped)
        results.append({**mapped, "cluster": result["cluster"]})

    output_path = OUTPUTS_BASE / "wine_clustering" / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    df.to_csv(output_path / f"batch_{timestamp}.csv", index=False)
    log.info(f"wine_clustering batch complete: {len(results)} rows → {output_path}")
    return df


def run_rule_engine_batch():
    log.info("Starting rule_engine batch inference")
    from models.rule_engine.inference import predict

    rng = np.random.default_rng(seed=int(datetime.now().timestamp()))
    n = 100
    samples = [
        {
            "age": int(rng.integers(18, 80)),
            "income": float(rng.integers(500, 150000)),
            "credit_score": int(rng.integers(300, 850)),
        }
        for _ in range(n)
    ]

    results = []
    for sample in samples:
        result = predict(sample)
        results.append({**sample, **result})

    output_path = OUTPUTS_BASE / "rule_engine" / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    df.to_csv(output_path / f"batch_{timestamp}.csv", index=False)
    log.info(f"rule_engine batch complete: {len(results)} rows → {output_path}")
    return df


def run_all():
    log.info("=== Batch run started ===")
    try:
        run_wine_clustering_batch()
        run_rule_engine_batch()
    except Exception as e:
        log.error(f"Batch run failed: {e}", exc_info=True)

    try:
        from monitor import run_monitoring
        run_monitoring()
    except Exception as e:
        log.error(f"Monitoring failed: {e}", exc_info=True)

    log.info("=== Batch run complete ===")


if __name__ == "__main__":
    log.info(f"Batch runner starting — interval: {INTERVAL_MINUTES} minutes")
    run_all()  # Run once immediately on startup

    scheduler = BlockingScheduler()
    scheduler.add_job(run_all, "interval", minutes=INTERVAL_MINUTES)
    scheduler.start()
