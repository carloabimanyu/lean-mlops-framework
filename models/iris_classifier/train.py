import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_NAME = "iris-classifier"


def train() -> str:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("iris-classifier")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.sklearn.log_model(model, name="model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, MODEL_NAME)

        return run.info.run_id


if __name__ == "__main__":
    run_id = train()
    print(f"Training complete. Run ID: {run_id}")
