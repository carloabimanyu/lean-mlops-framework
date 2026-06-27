import os
import pytest
from fastapi.testclient import TestClient

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"

@pytest.fixture
def client():
    from services.api.main import app
    return TestClient(app)

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_list_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_iris(client):
    response = client.post("/predict/iris-classifier", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "label" in response.json()

def test_predict_unknown_model(client):
    response = client.post("/predict/does-not-exist", json={"foo": 1})
    assert response.status_code == 404
