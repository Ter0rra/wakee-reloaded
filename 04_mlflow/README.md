---
title: Wakee MLflow
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🚀 MLflow Tracking Server - HuggingFace Spaces

MLflow tracking server déployé sur HuggingFace Spaces pour le projet Wakee Reloaded.

## 📊 Architecture

```
GitHub Repository (04_mlflow/)
    ↓
GitHub Actions
    ↓
HuggingFace Spaces
    ↓
MLflow UI: https://your-username-wakee-mlflow.hf.space
```

## 🎯 Fonctionnalités

- ✅ Tracking des expériences ML
- ✅ Logging des métriques (accuracy, MAE, etc.)
- ✅ Versioning des modèles
- ✅ Storage des artifacts (R2)
- ✅ Backend PostgreSQL (NeonDB)
- ✅ Déploiement automatique via GitHub Actions

## 📁 Structure

```
04_mlflow/
├── app.py                      # MLflow server
├── Dockerfile                  # HF Spaces config
├── requirements.txt            # Dependencies
├── .github/
│   └── workflows/
│       └── deploy-mlflow.yml   # CI/CD
└── README.md                   # Ce fichier
```

## 🚀 Setup

### 1. Créer le Space sur HuggingFace

```bash
# Va sur https://huggingface.co/new-space
# - Nom : wakee-mlflow
# - SDK : Docker
# - Visibility : Public ou Private
```

### 2. Configurer les secrets GitHub

Dans ton repository GitHub :

```
Settings → Secrets and variables → Actions → New repository secret
```

Ajoute :
- `HF_TOKEN` : Ton HuggingFace token
- `HF_USERNAME` : Ton username HuggingFace

### 3. Configurer les variables HF Spaces

Dans HuggingFace Spaces Settings :

```bash
# Backend Store (NeonDB)
MLFLOW_BACKEND_STORE_URI=postgresql://user:password@host/database

# Artifact Root (Cloudflare R2)
MLFLOW_ARTIFACT_ROOT=s3://wakee-bucket/mlflow-artifacts/

# R2 Configuration
MLFLOW_S3_ENDPOINT_URL=https://account_id.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=your_r2_access_key
AWS_SECRET_ACCESS_KEY=your_r2_secret_key
```

### 4. Deploy

```bash
# Push vers GitHub
git add 04_mlflow/
git commit -m "Add MLflow tracking server"
git push origin main

# GitHub Actions se déclenche automatiquement
# → Deploy sur HF Spaces
```

### 5. Vérifier

```bash
# Accède à ton Space
https://huggingface.co/spaces/your-username/wakee-mlflow

# Tu dois voir l'interface MLflow
```

## 🔧 Utilisation depuis Airflow

### Dans DAG 3 (model_retrain.py)

```python
import mlflow

# Configure l'URL du MLflow sur HF Spaces
MLFLOW_TRACKING_URI = "https://your-username-wakee-mlflow.hf.space"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("wakee-model-retrain")

# Log params
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.pytorch.log_model(model, "model")
```

### Variables d'environnement Airflow

```yaml
# docker-compose.yml
environment:
  MLFLOW_TRACKING_URI: https://your-username-wakee-mlflow.hf.space
```

## 📊 Backend Storage

### NeonDB (Metadata)
```sql
-- MLflow crée automatiquement ses tables
experiments
runs
metrics
params
tags
...
```

### Cloudflare R2 (Artifacts)
```
wakee-bucket/
└── mlflow-artifacts/
    ├── 0/
    │   └── run_id_xxx/
    │       └── artifacts/
    └── 1/
        └── run_id_yyy/
            └── artifacts/
```

## 🧪 Test local

```bash
cd 04_mlflow

# Build
docker build -t wakee-mlflow .

# Run (avec tes variables)
docker run -p 7860:7860 \
  -e MLFLOW_BACKEND_STORE_URI="postgresql://..." \
  -e MLFLOW_ARTIFACT_ROOT="s3://..." \
  -e MLFLOW_S3_ENDPOINT_URL="https://..." \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  wakee-mlflow

# Accède à http://localhost:7860
```

## 🔄 Workflow CI/CD

```
1. Modifie du code dans 04_mlflow/
   ↓
2. Push vers GitHub
   ↓
3. GitHub Actions détecte les changements
   ↓
4. Build & Deploy vers HF Spaces
   ↓
5. MLflow accessible sur HF Spaces
```

## 📈 Métriques trackées

### DAG 3 (Model Retrain)
```python
# Hyperparamètres
- learning_rate
- batch_size
- num_epochs

# Métriques training
- train_loss (par epoch)
- val_loss (par epoch)

# Métriques évaluation
- accuracy
- f1_score
- mae_boredom
- mae_confusion
- mae_engagement
- mae_frustration
- mae_global
```

## 🐛 Troubleshooting

### Space ne démarre pas
```bash
# Vérifie les logs dans HF Spaces
# Vérifie que MLFLOW_BACKEND_STORE_URI est configuré
```

### Cannot connect to PostgreSQL
```bash
# Vérifie que NeonDB est accessible depuis internet
# Vérifie les credentials
```

### Artifacts not saved
```bash
# Vérifie MLFLOW_S3_ENDPOINT_URL
# Vérifie AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY
# Vérifie que le bucket R2 existe
```

## 🎯 Avantages de cette architecture

```python
✅ MLflow accessible depuis n'importe où (pas local)
✅ Déploiement automatique (GitHub Actions)
✅ Séparé d'Airflow (indépendant)
✅ Gratuit (HF Spaces)
✅ Production-ready
✅ Versioning via Git
```

## 📚 Documentation

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces)
- [GitHub Actions](https://docs.github.com/en/actions)

## 🎉 Résultat

**Tu as maintenant un MLflow tracking server en production sur HF Spaces ! 🚀**

**URL finale :**
```
https://your-username-wakee-mlflow.hf.space
```
