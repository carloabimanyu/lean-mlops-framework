FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

RUN pip install --no-cache-dir "setuptools==70.3.0" && \
    uv pip install --system fastapi==0.111.0 uvicorn==0.30.1 mlflow==3.14.0 scikit-learn==1.4.2 pydantic==2.7.4

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
