import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

MODEL_NAME = "wine-clustering"


def train() -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("wine-clustering")

    wine = load_wine()
    X = wine.data

    params = {"n_clusters": 3, "random_state": 42, "n_init": 10}
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(**params)),
    ])
    pipeline.fit(X)

    inertia = pipeline.named_steps["kmeans"].inertia_

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("n_samples", len(X))
        mlflow.sklearn.log_model(pipeline, name="model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)

        return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    print(f"Training complete. Run ID: {run_id}")
