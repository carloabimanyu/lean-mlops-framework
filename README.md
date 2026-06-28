# Lean MLOps Framework

Minimalist MLOps framework with experiment tracking, model registry, real-time & batch inference, drift monitoring, and email alerting.

## Stack

| Layer | Technology |
|---|---|
| Experiment tracking + registry | MLflow 3.14 + SQLite |
| Real-time inference | FastAPI + Uvicorn |
| Batch inference | APScheduler |
| Drift monitoring | Evidently AI |
| Email alerting | Gmail SMTP |
| Containerization | Docker Compose |
| Model library | scikit-learn |
| Package manager | uv |

## Architecture

```
models/                   # Data Scientist workspace — one folder per model
  iris_classifier/        # Supervised — real-time inference (FastAPI)
  wine_clustering/        # Unsupervised — batch inference
  rule_engine/            # Deterministic — batch inference
services/
  api/                    # FastAPI server — auto-discovers all real-time models
  batch_runner/           # Runs batch inference + drift monitoring + alert
mlflow/                   # MLflow tracking server
reports/                  # Evidently HTML drift reports (auto-generated)
```

Three Docker containers communicate over a shared network. Trains models locally pointing at the MLflow container; the API and batch runner pick up registered models automatically.

## Quick Start

**Prerequisites:** Docker Desktop, Python 3.12, uv

**1. Clone and configure**

```bash
git clone https://github.com/carloabimanyu/lean-mlops-framework.git
cd lean-mlops-framework
cp .env.example .env       # Fill in GMAIL_SENDER, GMAIL_APP_PASSWORD, GMAIL_RECIPIENT
uv sync                    # Install local dependencies
```

**2. Start the stack**

```bash
docker compose up -d
```

MLflow UI: `http://localhost:5001`
Inference API: `http://localhost:8000`

**3. Train models**

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python models/iris_classifier/train.py
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python models/wine_clustering/train.py
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python models/rule_engine/setup.py
```

**4. Run inference**

```bash
# Real-time (iris classifier)
curl -X POST http://localhost:8000/predict/iris-classifier \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# List all available models
curl http://localhost:8000/models
```

Batch inference runs automatically on the schedule defined in `BATCH_SCHEDULE_INTERVAL_MINUTES`.

**5. Run tests**

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 uv run pytest tests/ -v
```

## Model Convention

Every model must expose two entry points:

```python
# train.py (or setup.py for rule-based models)
with mlflow.start_run():
    mlflow.log_params({...})
    mlflow.log_metrics({...})
    mlflow.register_model(...)   # registers to MLflow Model Registry

# inference.py
def predict(data: dict) -> dict:
    ...  # called by FastAPI and batch_runner — this is the only interface
```

Internal structure of each model folder is free — subfolders, preprocessing modules, notebooks, and tests are all welcome.

## Adding a New Model

1. Create a folder under `models/`
2. Write `train.py` following the convention above
3. Write `inference.py` with a `predict(data: dict) -> dict` function
4. Run `train.py` — model appears in MLflow registry and API endpoint is live automatically

No changes needed in `services/`.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow:5000` |
| `GMAIL_SENDER` | Gmail address for alerts | — |
| `GMAIL_APP_PASSWORD` | Gmail App Password (16 chars) | — |
| `GMAIL_RECIPIENT` | Alert recipient email | — |
| `DRIFT_THRESHOLD` | Drift score threshold (0.0–1.0) | `0.5` |
| `BATCH_SCHEDULE_INTERVAL_MINUTES` | Batch run interval | `60` |

> Note: `MLFLOW_TRACKING_URI` inside containers uses `http://mlflow:5000`. For local scripts (training), use `http://localhost:5001`.
