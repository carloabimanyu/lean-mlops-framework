import os
import pytest
import mlflow

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")


@pytest.fixture(scope="session")
def trained_run_id():
    from models.iris_classifier.train import train
    return train()


def test_train_registers_model(trained_run_id):
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='iris-classifier'")
    assert len(versions) > 0
    assert trained_run_id is not None


def test_train_logs_metrics(trained_run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(trained_run_id)
    assert "accuracy" in run.data.metrics
    assert run.data.metrics["accuracy"] > 0.8
