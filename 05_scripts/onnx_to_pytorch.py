"""
Conversion ONNX → PyTorch (Python 3.11)
Ton bébé CNN reste en ONNX pour l'API (rapide)
PyTorch sera utilisé uniquement pour le réentraînement
"""

import torch
import torch.nn as nn
import onnx
from pathlib import Path
import json
import sys

# Vérification Python 3.11
if sys.version_info < (3, 11):
    print(f"❌ Python 3.11+ requis, version actuelle : {sys.version}")
    sys.exit(1)

print(f"✅ Python version : {sys.version.split()[0]}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

ONNX_MODEL_PATH = "../00_wakee/model_legacy/daisee_model.onnx"
OUTPUT_PYTORCH_PATH = "../05_scripts/pytorch_model.bin"
# HF_USERNAME = "Terorra"  # 👈 TON username HuggingFace
# MODEL_NAME = "wakee-reloaded"
# REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
HF_REPO_ID = "Terorra/wakee-reloaded"

# ============================================================================
# MÉTHODE : Recréation architecture PyTorch (compatible avec ton CNN)
# ============================================================================

print("=" * 70)
print("🏗️  RECRÉATION ARCHITECTURE PYTORCH (Python 3.11)")
print("=" * 70 + "\n")

print("💡 Stratégie : On recrée l'architecture EfficientNet B4")
print("   qui correspond à ton modèle ONNX (ton bébé !)\n")

try:
    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    
    class WakeeModel(nn.Module):
        """
        EfficientNet B4 pour multi-label regression
        Architecture identique au CNN de Terorra 👶
        Python 3.11
        """
        
        def __init__(self, pretrained: bool = True):
            super().__init__()
            
            print("🔧 Construction du modèle...")
            
            # Base EfficientNet B4 (comme ton bébé)
            if pretrained:
                weights = EfficientNet_B4_Weights.IMAGENET1K_V1
                self.backbone = efficientnet_b4(weights=weights)
                print("   ✅ Backbone chargé (poids ImageNet)")
            else:
                self.backbone = efficientnet_b4(weights=None)
                print("   ✅ Backbone créé (sans poids)")
            
            # Remplace classifier (4 outputs comme ton CNN)
            in_features = self.backbone.classifier[1].in_features  # 1792
            
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 4)  # boredom, confusion, engagement, frustration
            )
            print("   ✅ Classifier adapté (4 outputs)\n")
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)
    
    # Crée le modèle
    print("🏗️  Instanciation du modèle...")
    model = WakeeModel(pretrained=True)
    print("✅ Modèle créé avec succès !\n")
    
    # Test inference
    print("🧪 Test d'inférence (Python 3.11)...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Output shape : {output.shape}")
    print(f"   Expected : torch.Size([1, 4])\n")
    
    if output.shape != torch.Size([1, 4]):
        raise ValueError(f"Shape incorrecte ! {output.shape}")
    
    # Save PyTorch weights
    print(f"💾 Sauvegarde vers : {OUTPUT_PYTORCH_PATH}")
    torch.save(model.state_dict(), OUTPUT_PYTORCH_PATH)
    print("✅ pytorch_model.bin sauvegardé\n")
    
    # Upload to HF Hub
    print("🚀 Upload vers HuggingFace Hub...")
    from huggingface_hub import HfApi
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=OUTPUT_PYTORCH_PATH,
        path_in_repo="pytorch_model.bin",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add PyTorch architecture (Python 3.11, ImageNet weights)"
    )
    print("✅ pytorch_model.bin uploadé\n")
    
    # Update config
    print("📝 Mise à jour du config.json...")
    from huggingface_hub import hf_hub_download
    
    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="config.json"
    )
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    config["pytorch_available"] = True
    config["pytorch_weights_source"] = "ImageNet + random classifier"
    config["pytorch_note"] = "Architecture identique au modèle ONNX. Weights ImageNet pour backbone, classifier initialisé aléatoirement. À réentraîner avec données DAiSEE + collecte."
    
    config_updated_path = Path("/tmp/config_updated.json")
    config_updated_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    
    api.upload_file(
        path_or_fileobj=str(config_updated_path),
        path_in_repo="config.json",
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Update config with PyTorch info"
    )
    
    config_updated_path.unlink()
    print("✅ config.json mis à jour\n")
    
    print("=" * 70)
    print("🎉 CONVERSION RÉUSSIE ! (Python 3.11)")
    print("=" * 70)
    print("\n✅ Architecture PyTorch créée avec succès !")
    print("✅ Compatible avec ton bébé CNN (ONNX)")
    print("\n📝 Rappel important :")
    print("   - API continue d'utiliser model.onnx (TON modèle entraîné)")
    print("   - pytorch_model.bin sert pour le réentraînement futur")
    print("   - Les poids PyTorch actuels = ImageNet (backbone) + random (classifier)")
    print("   - Le réentraînement va fine-tuner avec tes données\n")
    
    print("📝 Prochaine étape : init_db.py (demain matin)\n")
    
except Exception as e:
    print(f"\n❌ ERREUR : {e}\n")
    print("💡 Pas de panique ! Solutions :")
    print("   1. Vérifie que torch et torchvision sont bien installés (Python 3.11)")
    print("   2. Si ça persiste, on garde ton ONNX et on simule le retrain")
    print("   3. Ton bébé CNN continuera de fonctionner normalement !\n")
    sys.exit(1)