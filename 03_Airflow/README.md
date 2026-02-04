# 🚀 Wakee Airflow - Pipeline MLOps

Pipeline d'orchestration MLOps pour le projet Wakee Reloaded.

## 📊 Architecture

```
Airflow Scheduler
    ├── DAG 1: health_check_weekly (Dimanche 3h)
    │   ├── Test API
    │   ├── Test Database
    │   ├── Test Storage (R2)
    │   ├── Test App Sourcing
    │   ├── Test Model Hub
    │   ├── Run Pytest suite
    │   └── Generate Summary
    │
    ├── DAG 2: drift_detection_daily (Quotidien 2h)
    │   ├── Fetch annotations from NeonDB
    │   ├── Load predictions
    │   ├── Calculate drift (Evidently AI)
    │   ├── Generate drift report
    │   ├── Save report to NeonDB
    │   └── Trigger retrain if drift > threshold
    │
    └── DAG 3: model_retrain (Manuel / Triggered)
        ├── Fetch training data (NeonDB + R2)
        ├── Split train/val/test
        ├── Fine-tune PyTorch model
        ├── Evaluate on test set
        ├── Export to ONNX
        ├── Upload to HF Model Hub
        ├── Log metrics to MLflow
        └── Update model_versions table
```

## 🛠️ Installation

### Prérequis

- Docker & Docker Compose
- 4GB+ RAM disponible
- 10GB+ espace disque

### Setup

1. **Clone le repository**
```bash
cd wakee_reloaded/03_Airflow
```

2. **Configure les variables d'environnement**
```bash
cp .env.example .env
# Édite .env avec tes credentials
```

3. **Définis l'UID Airflow (Linux seulement)**
```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```

4. **Build l'image Docker**
```bash
docker-compose build
```

5. **Initialize Airflow**
```bash
docker-compose up airflow-init
```

6. **Lance Airflow**
```bash
docker-compose up -d
```

7. **Accède à l'interface Web**
```
URL: http://localhost:8080
Username: airflow
Password: airflow
```

## 📁 Structure

```
03_Airflow/
├── dags/
│   ├── dag_health_check.py      # Health checks hebdomadaires
│   ├── dag_drifting.py          # Détection drift quotidienne
│   └── dag_retrain.py           # Réentraînement modèle
│
├── tests/
│   ├── test_api_health.py
│   ├── test_database.py
│   ├── test_storage.py
│   └── test_model.py
│
├── docker-compose.yml           # Configuration services
├── Dockerfile                   # Image Airflow custom
├── requirements.txt             # Dépendances Python
├── .env.example                 # Template variables
└── README.md                    # Ce fichier
```

## 🔧 Configuration

### Variables d'environnement requises

```bash
# NeonDB
NEON_DATABASE_URL=postgresql://user:pass@host/db

# Cloudflare R2
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=wakee-bucket

# HuggingFace
HF_TOKEN=hf_xxxxx
```

## 📊 DAGs

### DAG 1: Health Check (Hebdomadaire)

**Schedule:** Dimanche 3h du matin

**Tests:**
- ✅ API endpoints (/health, /predict)
- ✅ Database (NeonDB connexion, tables)
- ✅ Storage (R2 upload/download)
- ✅ App Sourcing (accessibilité)
- ✅ Model Hub (download modèle)
- ✅ Pytest suite

**Output:** Rapport de santé complet

### DAG 2: Drift Detection (Quotidien)

**Schedule:** Tous les jours à 2h

**Process:**
1. Récupère annotations validées (NeonDB)
2. Compare avec prédictions initiales
3. Calcule métriques de drift (Evidently AI)
4. Génère rapport de drift
5. Sauvegarde dans drift_reports table
6. Déclenche réentraînement si drift > seuil

**Seuil de drift:** 0.15 (configurable)

### DAG 3: Model Retrain (Manuel/Triggered)

**Triggers:**
- Manuel (via UI Airflow)
- Automatique si drift détecté

**Process:**
1. Download données (R2 + NeonDB)
2. Preprocessing & split
3. Fine-tune EfficientNet B4
4. Évaluation (accuracy, F1, confusion matrix)
5. Export ONNX
6. Upload HF Model Hub
7. Log MLflow
8. Update model_versions table

**Durée:** ~30-60 minutes

## 🧪 Tests

### Exécuter les tests manuellement

```bash
docker-compose exec airflow-scheduler pytest /opt/airflow/tests -v
```

### Tests inclus

- `test_api_health.py`: Tests endpoints API
- `test_database.py`: Tests connexion/tables NeonDB
- `test_storage.py`: Tests upload/download R2
- `test_model.py`: Tests inférence ONNX

## 📝 Logs

### Accéder aux logs

```bash
# Logs Airflow webserver
docker-compose logs airflow-webserver

# Logs scheduler
docker-compose logs airflow-scheduler

# Logs spécifiques à un DAG
# Via UI: http://localhost:8080 → DAGs → [Nom du DAG] → Logs
```

### Emplacement des logs

```
03_Airflow/logs/
├── dag_id=health_check_weekly/
├── dag_id=drift_detection_daily/
└── dag_id=model_retrain/
```

## 🔄 Maintenance

### Arrêter Airflow

```bash
docker-compose down
```

### Redémarrer Airflow

```bash
docker-compose restart
```

### Nettoyer les volumes

```bash
docker-compose down -v
```

### Rebuild après changement requirements

```bash
docker-compose build --no-cache
docker-compose up -d
```

## 📊 Monitoring

### Interface Airflow

- **URL:** http://localhost:8080
- **DAGs:** Liste des pipelines
- **Graph View:** Visualisation du flow
- **Task Logs:** Logs détaillés de chaque tâche
- **XCom:** Variables partagées entre tâches

### Métriques

Airflow expose des métriques sur:
- Durée d'exécution des DAGs
- Taux de succès/échec
- Temps d'attente des tâches

## 🐛 Troubleshooting

### Erreur: "Permission denied"

```bash
# Linux: Définis AIRFLOW_UID
echo "AIRFLOW_UID=$(id -u)" >> .env
docker-compose down
docker-compose up -d
```

### Erreur: "Database not found"

```bash
# Réinitialise la DB Airflow
docker-compose down -v
docker-compose up airflow-init
docker-compose up -d
```

### DAG ne s'affiche pas

```bash
# Vérifie les erreurs de syntaxe
docker-compose exec airflow-scheduler python /opt/airflow/dags/dag_health_check.py

# Redémarre le scheduler
docker-compose restart airflow-scheduler
```

### Variables d'environnement non chargées

```bash
# Vérifie le .env
cat .env

# Rebuild avec nouvelles variables
docker-compose down
docker-compose up -d
```

## 📚 Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

## 🎯 Next Steps

1. ✅ Setup Airflow
2. ⏳ Créer DAG drift detection
3. ⏳ Créer DAG retrain
4. ⏳ Intégrer MLflow
5. ⏳ Configurer alertes email

## 📧 Support

Pour toute question, ouvre une issue sur le repository GitHub.

---

**Développé avec 💙 pour la certification AIA Lead MLOps**
