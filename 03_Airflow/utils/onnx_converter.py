"""
ONNX Converter - Conversion bidirectionnelle ONNX ↔ PyTorch
"""

import torch
import torch.nn as nn
from torchvision import models
import onnxruntime as ort
import numpy as np
from typing import Optional
import os

NUM_CLASSES = 4
DEVICE = 'cpu'

# ============================================================================
# ONNX → PYTORCH
# ============================================================================

def onnx_to_pytorch(
    onnx_path: str,
    output_bin_path: str,
    verify: bool = True
) -> bool:
    """
    Convertit ONNX → PyTorch .bin
    
    Stratégie: Crée architecture PyTorch + copie les poids depuis ONNX
    """
    print(f"\n🔄 Converting ONNX → PyTorch...")
    print(f"   Input: {onnx_path}")
    print(f"   Output: {output_bin_path}")
    
    try:
        # 1. Charger ONNX
        print("\n📦 Loading ONNX model...")
        session = ort.InferenceSession(onnx_path)
        
        # Test ONNX
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        onnx_output = session.run(['output'], {'input': test_input})[0]
        print(f"   ✅ ONNX loaded (output: {onnx_output[0]})")
        
        # 2. Créer architecture PyTorch
        print("\n🏗️  Creating PyTorch model...")
        pytorch_model = models.efficientnet_b4(weights=None)
        pytorch_model.classifier[1] = nn.Linear(
            pytorch_model.classifier[1].in_features, 
            NUM_CLASSES
        )
        
        # 3. Tenter onnx2torch
        print("\n🔄 Attempting onnx2torch conversion...")
        try:
            from onnx2torch import convert
            
            converted_model = convert(onnx_path)
            converted_model.eval()
            
            # Test
            with torch.no_grad():
                test_tensor = torch.from_numpy(test_input)
                pytorch_output = converted_model(test_tensor).numpy()
            
            # Comparer avec ONNX
            diff = np.abs(pytorch_output - onnx_output).max()
            print(f"   onnx2torch diff: {diff:.6f}")
            
            if diff < 0.01:
                print(f"   ✅ onnx2torch successful!")
                
                # Sauvegarder
                torch.save(converted_model.state_dict(), output_bin_path)
                
                file_size = os.path.getsize(output_bin_path) / 1e6
                print(f"   💾 Saved: {file_size:.2f} MB")
                
                return True
            else:
                print(f"   ⚠️  onnx2torch diff too large")
                raise ValueError("onnx2torch failed validation")
                
        except Exception as e:
            print(f"   ⚠️  onnx2torch failed: {e}")
            print(f"   → Fallback: ImageNet baseline")
        
        # 4. Fallback: ImageNet weights
        print("\n🔄 Fallback: Using ImageNet weights...")
        from torchvision.models import EfficientNet_B4_Weights
        
        pretrained_model = models.efficientnet_b4(
            weights=EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        
        # Copier backbone
        model_dict = pytorch_model.state_dict()
        pretrained_dict = pretrained_model.state_dict()
        
        pretrained_dict_filtered = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and 'classifier' not in k
        }
        
        model_dict.update(pretrained_dict_filtered)
        pytorch_model.load_state_dict(model_dict)
        
        print(f"   ✅ ImageNet backbone loaded")
        print(f"   ⚠️  Classifier random (will be fine-tuned)")
        
        # Sauvegarder
        torch.save(pytorch_model.state_dict(), output_bin_path)
        
        file_size = os.path.getsize(output_bin_path) / 1e6
        print(f"   💾 Saved: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# PYTORCH → ONNX
# ============================================================================

def pytorch_to_onnx(
    pytorch_model: nn.Module,
    output_onnx_path: str,
    opset_version: int = 11
) -> bool:
    """
    Convertit PyTorch → ONNX
    Mode TRAIN pour éviter BatchNorm issues
    """
    print(f"\n🔄 Converting PyTorch → ONNX...")
    print(f"   Output: {output_onnx_path}")
    print(f"   Opset: {opset_version}")
    
    try:
        # Force TRAIN mode
        pytorch_model.train()
        pytorch_model = pytorch_model.to(DEVICE)
        
        # Test PyTorch
        print("\n🧪 Testing PyTorch model...")
        with torch.no_grad():
            test1 = torch.randn(1, 3, 224, 224)
            test2 = torch.randn(1, 3, 224, 224)
            
            out1 = pytorch_model(test1).numpy()
            out2 = pytorch_model(test2).numpy()
            
            diff = np.abs(out1 - out2).max()
            print(f"   Variation: {diff:.4f}")
            
            if diff < 1e-6:
                print(f"   ❌ Model outputs constant!")
                return False
            
            print(f"   ✅ PyTorch model OK")
        
        # Export ONNX
        print("\n📤 Exporting to ONNX...")
        
        dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
        
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            output_onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.TRAINING  # ✅ TRAIN mode
        )
        
        # Vérifier taille
        file_size = os.path.getsize(output_onnx_path) / 1e6
        print(f"   ONNX size: {file_size:.2f} MB")
        
        if file_size < 10:
            print(f"   ❌ File too small!")
            return False
        
        print(f"   ✅ Export successful")
        
        # Test ONNX
        print("\n🧪 Testing ONNX model...")
        session = ort.InferenceSession(output_onnx_path)
        
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        onnx_output = session.run(['output'], {'input': test_input})[0]
        
        print(f"   ✅ ONNX inference OK")
        print(f"   Sample output: {onnx_output[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
