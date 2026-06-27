import os
import mlflow
import mlflow.sklearn
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "iris-classifier"
LABELS = ["setosa", "versicolor", "virginica"]

_model = None


class InputSchema(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def load_model():
    global _model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")


def predict(data: dict) -> dict:
    if _model is None:
        load_model()
    validated = InputSchema(**data)
    features = [[
        validated.sepal_length,
        validated.sepal_width,
        validated.petal_length,
        validated.petal_width,
    ]]
    prediction = int(_model.predict(features)[0])
    return {"prediction": prediction, "label": LABELS[prediction]}
