---
title: Wakee API
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# 🧠 Wakee Emotion Detection API

API FastAPI pour la détection multi-label d'émotions (Boredom, Confusion, Engagement, Frustration).

Développé par **Terorra** avec Python 3.11.

## 🚀 Endpoints

### `/predict` - Prédiction d'émotions

```bash
curl -X POST https://terorra-wakee-api.hf.space/predict \
  -F "file=@face.jpg"
```

**Response:**
```json
{
  "boredom": 1.25,
  "confusion": 0.80,
  "engagement": 2.15,
  "frustration": 0.35,
  "timestamp": "2025-02-04T10:30:00"
}
```

### `/insert` - Insérer annotation utilisateur

```bash
curl -X POST https://terorra-wakee-api.hf.space/insert \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "...",
    "predicted_boredom": 1.2,
    "predicted_confusion": 0.8,
    "predicted_engagement": 2.1,
    "predicted_frustration": 0.3,
    "user_boredom": 1.5,
    "user_confusion": 1.0,
    "user_engagement": 2.0,
    "user_frustration": 0.5
  }'
```

### `/load` - Charger données et statistiques

```bash
curl https://terorra-wakee-api.hf.space/load?limit=10
```

### `/health` - Health check

```bash
curl https://terorra-wakee-api.hf.space/health
```

### `/docs` - Documentation Swagger

Accédez à la documentation interactive : [https://terorra-wakee-api.hf.space/docs](https://terorra-wakee-api.hf.space/docs)

## 🏗️ Architecture

- **Modèle** : EfficientNet B4 (ONNX)
- **Source du modèle** : [Terorra/wakee-reloaded](https://huggingface.co/Terorra/wakee-reloaded)
- **Dataset** : DAiSEE
- **Framework** : FastAPI + ONNX Runtime
- **Python** : 3.11
- **Stockage** : Cloudflare R2 + NeonDB

## 🔄 Workflow

1. **Prédiction** : L'utilisateur envoie une image → API retourne les 4 scores
2. **Validation** : L'utilisateur corrige les scores si nécessaire
3. **Insertion** : L'image est uploadée vers R2 et les labels vers NeonDB
4. **Collecte** : Les données validées servent au réentraînement du modèle

## 🔐 Secrets requis

L'API nécessite les secrets suivants (configurés dans les Settings du Space) :

- `NEON_DATABASE_URL` : Connection string PostgreSQL
- `R2_ACCOUNT_ID` : Cloudflare account ID
- `R2_ACCESS_KEY_ID` : Cloudflare access key
- `R2_SECRET_ACCESS_KEY` : Cloudflare secret key
- `R2_BUCKET_NAME` : Nom du bucket R2

## 🔗 Liens

- [Code source GitHub](https://github.com/Terorra/wakee-reloaded)
- [Modèle HuggingFace](https://huggingface.co/Terorra/wakee-reloaded)
- [App Sourcing](https://huggingface.co/spaces/Terorra/wakee-sourcing)

## 📄 License

MIT

---

**Développé avec 💙 par Terorra - Certification AIA Lead MLOps**
