# MLflow Tracking Server - HuggingFace Spaces
# Simple & Working version

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose HF Spaces port
EXPOSE 7860

# Set environment variables (will be overridden by HF Spaces)
ENV MLFLOW_BACKEND_STORE_URI=""
ENV MLFLOW_ARTIFACT_ROOT=""
ENV MLFLOW_S3_ENDPOINT_URL=""
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV PORT=7860

# Run MLflow server directly (no app.py)
CMD mlflow server \
    --host 0.0.0.0 \
    --port "${PORT} "\
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}"
