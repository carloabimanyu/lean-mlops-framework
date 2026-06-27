import os
import pytest
import mlflow

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")


def test_train_registers_model():
    from models.iris_classifier.train import train
    run_id = train()
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='iris-classifier'")
    assert len(versions) > 0
    assert run_id is not None


def test_train_logs_metrics():
    from models.iris_classifier.train import train
    run_id = train()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert "accuracy" in run.data.metrics
    assert run.data.metrics["accuracy"] > 0.8
