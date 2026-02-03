"""
Script d'initialisation : Upload du modèle Wakee vers HuggingFace Hub
À exécuter UNE SEULE FOIS pour créer le repo model
"""

from huggingface_hub import HfApi, create_repo, login
import json
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION - À MODIFIER SELON TON SETUP
# ============================================================================

HF_USERNAME = "Terorra"  # 👈 TON username HuggingFace
MODEL_NAME = "wakee-reloaded"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# Chemins vers tes fichiers (depuis wakee_reloaded/)
ONNX_MODEL_PATH = "../00_wakee/model_legacy/daisee_model.onnx"  # 👈 Ton ONNX existant

# ============================================================================
# 1. LOGIN HUGGINGFACE
# ============================================================================

print("🔐 Connexion à HuggingFace...")
print("\n⚠️  Tu vas avoir besoin d'un token HuggingFace !")
print("👉 Va sur : https://huggingface.co/settings/tokens")
print("👉 Crée un token avec permissions 'write'")
print("👉 Copie-le et colle-le ci-dessous\n")

# Option 1 : Login interactif (recommandé première fois)
try:
    login()
    print("✅ Connexion réussie !\n")
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")
    print("\n💡 Alternative : définis la variable d'environnement")
    print("   export HF_TOKEN='ton_token_ici'")
    exit(1)

# Option 2 : Si tu as déjà le token en variable d'environnement
# from huggingface_hub import login
# login(token=os.getenv("HF_TOKEN"))

api = HfApi()

# ============================================================================
# 2. CRÉER LE REPO MODEL
# ============================================================================

print(f"📦 Création du repo : {REPO_ID}")

# try:
#     create_repo(
#         repo_id=REPO_ID,
#         repo_type="model",
#         private=False,  # Public pour que l'API puisse y accéder
#         exist_ok=True   # Ne plante pas si existe déjà
#     )
#     print(f"✅ Repo créé : https://huggingface.co/{REPO_ID}\n")
# except Exception as e:
#     print(f"⚠️  Repo existe déjà ou erreur : {e}\n")

# ============================================================================
# 3. CRÉER LE CONFIG.JSON
# ============================================================================

print("📝 Création du fichier config.json...")

config = {
    "model_type": "efficientnet-b4",
    "architecture": "EfficientNet B4 fine-tuned on DAiSEE",
    "task": "multi-label-regression",
    "num_labels": 4,
    "label_names": ["boredom", "confusion", "engagement", "frustration"],
    "label_ranges": {
        "boredom": [0, 3],
        "confusion": [0, 3],
        "engagement": [0, 3],
        "frustration": [0, 3]
    },
    "input_size": [224, 224],
    "preprocessing": {
        "resize": 256,
        "center_crop": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    },
    "framework": "onnx",
    "onnx_opset": 11,
    "dataset": "DAiSEE",
    "baseline_metrics": {
        "val_mae": 0.5665,
        "val_rmse": 0.7016,
        "val_r2": -0.0014,
        "boredom_accuracy": 0.39,
        "confusion_accuracy": 0.46,
        "engagement_accuracy": 0.54,
        "frustration_accuracy": 0.72
    },
    "version": "1.0.0",
    "created_by": "Terorra",
    "license": "apache-2.0"
}

# Sauvegarde temporaire
config_path = "/tmp/config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print("✅ config.json créé\n")

# ============================================================================
# 4. CRÉER LE README.md
# ============================================================================

print("📄 Création du README.md...")

readme_content = f"""---
license: apache-2.0
tags:
- emotion-detection
- tdah
- adhd
- computer-vision
- multi-label-regression
library_name: onnxruntime
pipeline_tag: image-classification
---

# 🧠 Wakee Emotion Detector

**Modèle de détection d'émotions pour accompagnement TDAH**

## 📊 Description

Modèle EfficientNet B4 fine-tuné sur le dataset DAiSEE pour détecter 4 états émotionnels simultanés :

- **Boredom** (Ennui) : 0-3
- **Confusion** : 0-3  
- **Engagement** (Concentration) : 0-3
- **Frustration** : 0-3

Ce modèle est conçu pour l'application **Wakee** (Work Assistant with Kindness & Emotional Empathy), 
destinée à aider les personnes atteintes de TDAH pendant leurs sessions de travail.

## 🎯 Usage

### Avec ONNX Runtime (recommandé pour production)
```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Load model
session = ort.InferenceSession("model.onnx")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inference
image = Image.open("face.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).numpy()
outputs = session.run(['output'], {{'input': input_tensor}})
scores = outputs[0][0]  # [boredom, confusion, engagement, frustration]

print(f"Boredom: {{scores[0]:.2f}}/3")
print(f"Confusion: {{scores[1]:.2f}}/3")
print(f"Engagement: {{scores[2]:.2f}}/3")
print(f"Frustration: {{scores[3]:.2f}}/3")
```

### Avec HuggingFace Hub
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="model.onnx"
)
# Ensuite utilise model_path avec onnxruntime
```

## 📈 Performances (Baseline)

| Métrique | Valeur |
|----------|--------|
| **MAE globale** | 0.57 |
| **RMSE** | 0.70 |
| Boredom Accuracy | 39% |
| Confusion Accuracy | 46% |
| Engagement Accuracy | 54% |
| Frustration Accuracy | **72%** ✅ |

## 🏗️ Architecture

- **Base model** : EfficientNet B4 (pré-entraîné sur ImageNet)
- **Fine-tuning** : DAiSEE dataset
- **Output** : 4 scores de régression (0-3)
- **Loss** : Smooth L1 Loss
- **Framework** : PyTorch → ONNX export

## 📦 Dataset

Entraîné sur **DAiSEE** (Dataset for Affective States in E-Environments) :
- 9,068 vidéos
- 112 sujets
- 4 labels : Boredom, Engagement, Confusion, Frustration
- Échelle 0-3 pour chaque label

## 🔄 MLOps Pipeline

Ce modèle fait partie d'un pipeline MLOps complet :

1. **Collecte continue** : Images d'utilisateurs réels via app de sourcing
2. **Drift detection** : Evidently AI (hebdomadaire)
3. **Réentraînement automatique** : Airflow orchestration
4. **Versioning** : MLflow model registry

## 👨‍💻 Auteur

Développé par **Terorra** dans le cadre du projet Wakee (certification AIA Lead).

## 📄 License

Apache 2.0

## 🔗 Liens

- [Wakee App Repository](https://github.com/{HF_USERNAME}/wakee-reloaded)
- [API Endpoint](https://huggingface.co/spaces/{HF_USERNAME}/wakee-api)
- [Annotation App](https://huggingface.co/spaces/{HF_USERNAME}/wakee-sourcing)
"""

readme_path = "/tmp/README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print("✅ README.md créé\n")

# ============================================================================
# 5. UPLOAD VERS HUGGINGFACE HUB
# ============================================================================

print("🚀 Upload des fichiers vers HuggingFace Hub...\n")

# 5.1 Upload ONNX model
print("📤 Upload du modèle ONNX...")
if not Path(ONNX_MODEL_PATH).exists():
    print(f"❌ ERREUR : Fichier introuvable : {ONNX_MODEL_PATH}")
    print("👉 Vérifie le chemin vers ton daisee_model.onnx")
    exit(1)

try:
    api.upload_file(
        path_or_fileobj=ONNX_MODEL_PATH,
        path_in_repo="model.onnx",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Initial upload: ONNX model from DAiSEE training"
    )
    print("✅ model.onnx uploadé\n")
except Exception as e:
    print(f"❌ Erreur upload ONNX : {e}\n")
    exit(1)

# 5.2 Upload config.json
print("📤 Upload du config.json...")
try:
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add model configuration"
    )
    print("✅ config.json uploadé\n")
except Exception as e:
    print(f"❌ Erreur upload config : {e}\n")

# 5.3 Upload README.md
print("📤 Upload du README.md...")
try:
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add comprehensive README"
    )
    print("✅ README.md uploadé\n")
except Exception as e:
    print(f"❌ Erreur upload README : {e}\n")

# ============================================================================
# 6. VÉRIFICATION
# ============================================================================

print("=" * 70)
print("🎉 UPLOAD TERMINÉ !")
print("=" * 70)
print(f"\n✅ Ton modèle est disponible sur :")
print(f"   👉 https://huggingface.co/{REPO_ID}\n")

print("🔍 Vérifications à faire :")
print("   1. Visite le lien ci-dessus")
print("   2. Vérifie que model.onnx est bien là (devrait faire ~50-100 MB)")
print("   3. Lis le README généré")
print("   4. Teste le download :\n")

print("=" * 70)
print("📝 CODE DE TEST (à exécuter séparément) :")
print("=" * 70)

test_code = f"""
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# Download
model_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="model.onnx"
)

# Test load
session = ort.InferenceSession(model_path)
print(f"✅ Modèle chargé : {{model_path}}")
print(f"   Input : {{session.get_inputs()[0].name}}")
print(f"   Output : {{session.get_outputs()[0].name}}")
print(f"   Shape : {{session.get_inputs()[0].shape}}")
"""

print(test_code)
print("=" * 70)

# Cleanup
os.remove(config_path)
os.remove(readme_path)

print("\n✅ Script terminé ! Passe au script suivant : onnx_to_pytorch.py\n")