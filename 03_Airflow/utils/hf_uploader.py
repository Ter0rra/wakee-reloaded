"""
HuggingFace Uploader
Upload model.bin et model.onnx vers HF Model Hub
"""

from huggingface_hub import HfApi, create_repo, upload_file
from github import Github

import os
from typing import Optional, Dict
import shutil
from pathlib import Path

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
    github_repo: str,
    hf_repo: str,
    github_token: str,
    hf_token: str,
    commit_message: Optional[str] = None
) -> Dict:
    """
    Pipeline:
    1) Copie les modèles dans ./models/
    2) Push ce dossier vers GitHub via API
    3) Upload le dossier vers HuggingFace Hub (auto LFS)

    Returns:
        dict contenant urls GitHub + HF
    """

    if commit_message is None:
        commit_message = f"Upload model {version_name}"

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("📦 Step 1 — Copy local models")

    bin_dst = models_dir / "pytorch_model.bin"
    onnx_dst = models_dir / "model.onnx"

    shutil.copy(model_bin_path, bin_dst)
    shutil.copy(model_onnx_path, onnx_dst)

    print("✅ Files copied to ./models")

    # =========================================================
    # 2️⃣ PUSH TO GITHUB (sans git)
    # =========================================================
    print("\n🚀 Step 2 — Push to GitHub")

    gh = Github(github_token)
    repo = gh.get_repo(github_repo)

    github_urls = {}

    for file_path in models_dir.iterdir():
        with open(file_path, "rb") as f:
            content = f.read()

        try:
            repo.create_file(
                path=f"models/{file_path.name}",
                message=commit_message,
                content=content,
                branch="main"
            )
            print(f"✅ Created {file_path.name}")
        except Exception:
            # update si déjà existant
            existing = repo.get_contents(f"models/{file_path.name}")
            repo.update_file(
                path=existing.path,
                message=commit_message,
                content=content,
                sha=existing.sha,
                branch="main"
            )
            print(f"♻️ Updated {file_path.name}")

        github_urls[file_path.name] = f"https://github.com/{github_repo}/blob/main/models/{file_path.name}"

    # =========================================================
    # 3️⃣ UPLOAD TO HUGGINGFACE (auto LFS côté serveur)
    # =========================================================
    print("\n🤗 Step 3 — Upload to HuggingFace")

    api = HfApi(token=hf_token)

    api.upload_folder(
        folder_path=str(models_dir),
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=commit_message
    )

    hf_url = f"https://huggingface.co/{hf_repo}"

    print("✅ Upload HF terminé")

    return {
        "github_files": github_urls,
        "hf_repo": hf_url
    }

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
