# MLflow Tracking Server - HuggingFace Spaces
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose HF Spaces port
EXPOSE 7860

# Environment variables (set in HF Spaces settings)
ENV MLFLOW_BACKEND_STORE_URI=""
ENV MLFLOW_ARTIFACT_ROOT=""
ENV MLFLOW_S3_ENDPOINT_URL=""
ENV R2_ACCESS_KEY_ID=""
ENV R2_SECRET_ACCESS_KEY=""
ENV PORT=7860

# Run MLflow server
CMD ["python", "app.py"]
