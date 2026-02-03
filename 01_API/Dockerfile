FROM python:3.11-slim

WORKDIR /app

# Minimal system dependencies pour ONNX Runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages (SANS PyTorch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV ORT_DISABLE_TELEMETRY=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]