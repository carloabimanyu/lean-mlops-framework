import os
import mlflow.pyfunc
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MODEL_NAME = "rule-engine"

_model = None


class InputSchema(BaseModel):
    age: int
    income: float
    credit_score: int


def load_model():
    global _model
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")


def predict(data: dict) -> dict:
    if _model is None:
        load_model()
    validated = InputSchema(**data)
    results = _model.predict([validated.model_dump()])
    return results[0]
