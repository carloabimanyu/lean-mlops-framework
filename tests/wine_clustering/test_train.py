import os
import mlflow

os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")


def test_train_registers_model():
    from models.wine_clustering.train import train
    run_id = train()
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='wine-clustering'")
    assert len(versions) > 0


def test_train_logs_inertia():
    from models.wine_clustering.train import train
    run_id = train()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    assert "inertia" in run.data.metrics
    assert run.data.metrics["inertia"] > 0
