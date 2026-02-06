"""
HuggingFace Uploader
Upload model.bin et model.onnx vers HF Model Hub
"""

from huggingface_hub import HfApi, create_repo, upload_file
import os
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Terorra/wakee-reloaded")

# ============================================================================
# UPLOAD TO HF HUB
# ============================================================================

def upload_model_to_hf(
    model_bin_path: str,
    model_onnx_path: str,
    version_name: str,
    commit_message: Optional[str] = None
) -> dict:
    """
    Upload model.bin et model.onnx vers HuggingFace Hub
    
    Args:
        model_bin_path (str): Chemin local du model.bin
        model_onnx_path (str): Chemin local du model.onnx
        version_name (str): Version du modèle (ex: 'v1.1.0')
        commit_message (str): Message du commit
    
    Returns:
        dict: URLs des fichiers uploadés
    """
    print("🚀 Uploading models to HuggingFace Hub...")
    print(f"   Repository: {HF_MODEL_REPO}")
    print(f"   Version: {version_name}")
    
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not configured")
    
    # Créer l'API client
    api = HfApi(token=HF_TOKEN)
    
    # Vérifier que le repo existe (sinon créer)
    try:
        api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        print(f"✅ Repository exists: {HF_MODEL_REPO}")
    except Exception:
        print(f"⚠️  Repository not found, have to create: {HF_MODEL_REPO}")
        # create_repo(repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN)
    
    # Message de commit par défaut
    if commit_message is None:
        commit_message = f"Upload {version_name} - Automated retrain from drift detection"
    
    uploaded_files = {}
    
    # 1. Upload model.bin
    print("\n📦 Uploading pytorch_model.bin...")
    try:
        url_bin = api.upload_file(
            path_or_fileobj=model_bin_path,
            path_in_repo="pytorch_model.bin",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=commit_message
        )
        print(f"✅ model.bin uploaded: {url_bin}")
        uploaded_files['model_bin_url'] = url_bin
    except Exception as e:
        print(f"❌ Failed to upload model.bin: {e}")
        raise
    
    # 2. Upload model.onnx
    print("\n📦 Uploading model.onnx...")
    try:
        url_onnx = api.upload_file(
            path_or_fileobj=model_onnx_path,
            path_in_repo="model.onnx",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=commit_message
        )
        print(f"✅ model.onnx uploaded: {url_onnx}")
        uploaded_files['model_onnx_url'] = url_onnx
    except Exception as e:
        print(f"❌ Failed to upload model.onnx: {e}")
        raise
    
    # 3. Upload README avec version info (optionnel)
    print("\n📄 Updating README...")
    try:
        readme_content = generate_readme(version_name)
        readme_path = "/tmp/README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        url_readme = upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=f"Update README for {version_name}"
        )
        print(f"✅ README.md updated")
        uploaded_files['readme_url'] = url_readme
    except Exception as e:
        print(f"⚠️  Failed to update README: {e}")
    
    print("\n🎉 All files uploaded successfully!")
    
    return uploaded_files

# ============================================================================
# GENERATE README
# ============================================================================

def generate_readme(version_name: str) -> str:
    """Génère un README pour le Model Hub"""
    return f"""---
license: mit
tags:
- emotion-detection
- daisee
- efficientnet
- pytorch
datasets:
- daisee
---

# Wakee - Emotion Detection Model

Version: **{version_name}**

## Model Description

EfficientNet B4 fine-tuned for emotion detection in educational settings.

Predicts 4 emotion intensities (0-3 scale):
- Boredom
- Confusion  
- Engagement
- Frustration

## Training Data

- Base: DAiSEE dataset
- Fine-tuned: User-validated annotations from Wakee app

## Usage

### ONNX (Production)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("model.onnx")

# Preprocess image (224x224)
image = Image.open("image.jpg").resize((224, 224))
input_array = np.array(image).transpose(2, 0, 1).astype(np.float32)
input_array = np.expand_dims(input_array, axis=0) / 255.0

# Predict
outputs = session.run(['output'], {{'input': input_array}})
boredom, confusion, engagement, frustration = outputs[0][0]
```

### PyTorch (Fine-tuning)

```python
import torch
from torchvision import models

# Load checkpoint
model = models.efficientnet_b4()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("model.bin"))
```

## Model Card

- **Architecture**: EfficientNet B4
- **Framework**: PyTorch 2.1.2
- **Input**: RGB images (224x224)
- **Output**: 4 emotion scores (regression)
- **License**: MIT

## Metrics

See model_versions table in database for evaluation metrics.

## Citation

```bibtex
@software{{wakee_emotion_detection,
  author = {{Terorra}},
  title = {{Wakee Emotion Detection Model}},
  year = {{2025}},
  version = {{{version_name}}},
}}
```
"""

# ============================================================================
# DOWNLOAD FROM HF HUB
# ============================================================================

def download_model_from_hf(
    filename: str = "model.bin",
    cache_dir: str = "/tmp/wakee_models"
) -> str:
    """
    Télécharge un modèle depuis HF Hub
    
    Args:
        filename (str): Nom du fichier ('model.bin' ou 'model.onnx')
        cache_dir (str): Répertoire de cache
    
    Returns:
        str: Chemin local du fichier téléchargé
    """
    from huggingface_hub import hf_hub_download
    
    print(f"📥 Downloading {filename} from HuggingFace Hub...")
    
    model_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=filename,
        cache_dir=cache_dir,
        token=HF_TOKEN
    )
    
    print(f"✅ Downloaded to: {model_path}")
    
    return model_path
