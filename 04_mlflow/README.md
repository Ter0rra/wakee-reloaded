---
title: Wakee MLflow
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🚀 Wakee MLflow Tracking Server

MLflow tracking server for the Wakee emotion detection project.

## 🔧 Configuration

This Space requires the following environment variables in **Settings**:

### Backend Store (PostgreSQL)
```
MLFLOW_BACKEND_STORE_URI=postgresql://user:password@host/mlflow_backend?sslmode=require
```

### Artifact Storage (Cloudflare R2)
```
MLFLOW_ARTIFACT_ROOT=s3://bucket-name/mlflow-artifacts/
MLFLOW_S3_ENDPOINT_URL=https://account_id.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=your_r2_access_key
AWS_SECRET_ACCESS_KEY=your_r2_secret_key
```

## 📊 Usage

Once configured, access MLflow UI at:
```
https://huggingface.co/spaces/your-username/wakee-mlflow
```

### From Python

```python
import mlflow

mlflow.set_tracking_uri("https://your-username-wakee-mlflow.hf.space")
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("param1", "value1")
    mlflow.log_metric("metric1", 0.85)
```

## 🐛 Troubleshooting

### Space won't start

1. Check all environment variables are set in Settings
2. Verify PostgreSQL connection string is correct
3. Verify R2 credentials are valid
4. Check logs for specific error messages

### Database migration errors

If you see Alembic errors, reset the database:

```sql
DROP DATABASE mlflow_backend;
CREATE DATABASE mlflow_backend;
```

### Artifacts not saving to R2

1. Verify `MLFLOW_ARTIFACT_ROOT` starts with `s3://`
2. Verify `MLFLOW_S3_ENDPOINT_URL` is correct
3. Test R2 credentials with boto3

## 📚 Documentation

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Wakee Project](https://github.com/your-username/wakee-reloaded)

## 📝 License

MIT
