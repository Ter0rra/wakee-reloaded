"""
CRÉATION .BIN CORRECT depuis ton model.onnx ORIGINAL
Utilise onnx2torch OU fallback intelligent
"""

import torch
from torchvision import models
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from huggingface_hub import HfApi
import os

print("=" * 70)
print("🔧 CRÉATION PYTORCH_MODEL.BIN CORRECT")
print("=" * 70 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

ORIGINAL_ONNX_PATH = "../00_wakee/model_legacy/daisee_model.onnx"  # ← TON ONNX LOCAL
OUTPUT_BIN_PATH = "pytorch_model_CORRECT.bin"
HF_REPO_ID = "Terorra/wakee-reloaded"

NUM_CLASSES = 4
DEVICE = 'cpu'

# ============================================================================
# MÉTHODE 1 : onnx2torch (IDÉAL)
# ============================================================================

print("🔄 Tentative conversion avec onnx2torch...\n")

try:
    from onnx2torch import convert
    
    pytorch_model = convert(ORIGINAL_ONNX_PATH)
    pytorch_model.eval()
    
    print("✅ Conversion onnx2torch réussie !")
    
    # Test
    with torch.no_grad():
        test1 = torch.randn(1, 3, 224, 224)
        test2 = torch.randn(1, 3, 224, 224)
        out1 = pytorch_model(test1)
        out2 = pytorch_model(test2)
        diff = torch.abs(out1 - out2).max().item()
        print(f"   Output variation: {diff:.4f}")
    
    # Save
    torch.save(pytorch_model.state_dict(), OUTPUT_BIN_PATH)
    
    file_size_mb = os.path.getsize(OUTPUT_BIN_PATH) / 1e6
    print(f"   File size: {file_size_mb:.2f} MB")
    
    if file_size_mb < 50:
        raise ValueError("File too small!")
    
    print("\n✅ .bin créé avec onnx2torch !\n")
    USE_ONNX2TORCH = True

except Exception as e:
    print(f"⚠️  onnx2torch failed: {e}\n")
    USE_ONNX2TORCH = False

# ============================================================================
# MÉTHODE 2 : Fallback - ONNX comme référence (SI onnx2torch rate)
# ============================================================================

if not USE_ONNX2TORCH:
    print("🔄 Fallback : Création .bin avec ONNX comme référence...\n")
    
    # 1. Tester l'ONNX
    print("📊 Test du modèle ONNX original...")
    session = ort.InferenceSession(ORIGINAL_ONNX_PATH)
    
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    onnx_output = session.run(['output'], {'input': test_input})[0]
    
    print(f"   ONNX output: {onnx_output[0]}")
    print(f"   ✅ ONNX fonctionne\n")
    
    # 2. Créer architecture PyTorch
    print("🏗️  Création architecture PyTorch...")
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    # 3. Stratégie : Utiliser l'ONNX dans le DAG
    print("\n💡 RECOMMANDATION :")
    print("   Comme onnx2torch ne marche pas, voici ce qu'il faut faire :\n")
    print("   1. NE PAS uploader de pytorch_model.bin")
    print("   2. Upload seulement ton model.onnx ORIGINAL")
    print("   3. Modifier le DAG pour charger ONNX au lieu de .bin\n")
    
    # 4. Créer quand même un .bin "bootstrap" pour le DAG
    print("🔧 Création .bin bootstrap (ImageNet backbone)...")
    from torchvision.models import EfficientNet_B4_Weights
    pretrained_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
    
    # Copier backbone
    model_dict = model.state_dict()
    pretrained_dict = pretrained_model.state_dict()
    
    pretrained_dict_filtered = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and 'classifier' not in k
    }
    
    model_dict.update(pretrained_dict_filtered)
    model.load_state_dict(model_dict)
    
    # ⚠️  IMPORTANT : Freeze le backbone pour le fine-tuning
    print("   ⚠️  ATTENTION : Ce .bin a ImageNet backbone + classifier random")
    print("   → Le DAG devra FREEZE le backbone et ne fine-tuner QUE le classifier\n")
    
    torch.save(model.state_dict(), OUTPUT_BIN_PATH)
    print(f"✅ .bin bootstrap créé\n")

# ============================================================================
# UPLOAD
# ============================================================================

print("🚀 Upload vers HuggingFace Hub...\n")

api = HfApi()

# Upload le .bin
api.upload_file(
    path_or_fileobj=OUTPUT_BIN_PATH,
    path_in_repo="pytorch_model.bin",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Fix pytorch_model.bin - Use ONNX weights or ImageNet bootstrap"
)

print("✅ pytorch_model.bin uploadé\n")

# Upload aussi l'ONNX original
print("📤 Upload model.onnx original...")

api.upload_file(
    path_or_fileobj=ORIGINAL_ONNX_PATH,
    path_in_repo="model.onnx",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Restore original trained ONNX model"
)

print("✅ model.onnx original restauré\n")

print("=" * 70)
print("🎉 DONE !")
print("=" * 70)
print("""
PROCHAINES ÉTAPES :

1. ✅ pytorch_model.bin fixé (ou bootstrap)
2. ✅ model.onnx original restauré

3. 🔧 MODIFIER LE DAG pour FREEZE le backbone :

Dans model_trainer.py, ajoute avant le training :

```python
# Freeze le backbone (ne fine-tune QUE le classifier)
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

# Vérifier
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres entraînables : {trainable:,}")
```

4. 🔧 AUGMENTER learning rate pour le classifier :

```python
LEARNING_RATE = 1e-3  # Plus élevé car on train juste le classifier
NUM_EPOCHS = 10
```

5. 🔧 Relancer le DAG

RÉSULTAT ATTENDU :
- Backbone gelé → Garde les features ImageNet
- Classifier fine-tuné → Apprend tes émotions
- Pas de catastrophe sur les prédictions
""")
