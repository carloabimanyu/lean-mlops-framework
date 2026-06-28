FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install --system "setuptools<71" skops apscheduler==3.10.4 mlflow==3.14.0 scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4 pydantic==2.7.4 evidently==0.4.33

COPY runner.py ./
# monitor.py and alert.py are Phase 5 files — copied in when they exist
# Runner handles ImportError for monitor gracefully at runtime

CMD ["python", "runner.py"]
