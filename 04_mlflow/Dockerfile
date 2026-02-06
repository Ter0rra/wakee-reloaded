# # MLflow Tracking Server - HuggingFace Spaces
# FROM python:3.11-slim

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy app
# COPY app.py .

# # Expose HF Spaces port
# EXPOSE 7860

# # Environment variables (set in HF Spaces settings)
# ENV MLFLOW_BACKEND_STORE_URI=""
# ENV MLFLOW_ARTIFACT_ROOT=""
# ENV MLFLOW_S3_ENDPOINT_URL=""
# ENV R2_ACCESS_KEY_ID=""
# ENV R2_SECRET_ACCESS_KEY=""
# ENV PORT=7860

# # Run MLflow server
# CMD ["python", "app.py"]

FROM continuumio/miniconda3

RUN apt-get update
RUN apt-get install nano unzip curl -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN rm -rf awscliv2.zip aws

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

CMD mlflow server -p 7860 \
    --host 0.0.0.0 \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT