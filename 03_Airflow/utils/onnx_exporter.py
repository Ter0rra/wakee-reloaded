"""
ONNX Exporter - VERSION ULTIME
Utilise l'ANCIEN exporteur PyTorch pour éviter les optimisations
"""

import os
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = 'cpu'
INPUT_SHAPE = (1, 3, 224, 224)

# ============================================================================
# EXPORT TO ONNX (ANCIEN EXPORTEUR)
# ============================================================================

def export_to_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    opset_version: int = 11
) -> bool:
    """
    Exporte le modèle PyTorch vers ONNX
    UTILISE L'ANCIEN EXPORTEUR pour éviter les optimisations
    """
    print(f"🔄 Exporting PyTorch model to ONNX...")
    print(f"   Output: {onnx_path}")
    print(f"   Opset version: {opset_version}")
    print(f"   Using LEGACY ONNX exporter (no optimizations)")
    
    try:
        # Mode eval
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(DEVICE)
        
        # Créer un input dummy
        dummy_input = torch.randn(INPUT_SHAPE, device=DEVICE)
        
        # ✅ EXPORT AVEC L'ANCIEN EXPORTEUR
        # En mettant dynamo=False, on force l'ancien exporteur
        with torch.onnx.select_model_mode_for_export(
            pytorch_model, torch.onnx.TrainingMode.EVAL
        ):
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                onnx_path,
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
                # ✅ CRITIQUE : Forcer l'ancien exporteur
                dynamo=False,
                # ✅ CRITIQUE : Garder les formes originales
                keep_initializers_as_inputs=False,
            )

        # Vérifier la taille du fichier
        file_size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"   ONNX file size: {file_size_mb:.2f} MB")
        
        if file_size_mb < 10:
            print(f"   ⚠️  File size too small ({file_size_mb:.2f} MB)")
            print(f"   Expected: ~70-75 MB")
            
            # Vérifier le nombre d'initializers
            onnx_model = onnx.load(onnx_path)
            num_init = len(onnx_model.graph.initializer)
            print(f"   Initializers: {num_init}")
            print(f"   Expected: ~400+")
            
            return False
        
        print("✅ ONNX export successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# [Reste du code identique à avant...]

def verify_onnx_model(onnx_path: str) -> bool:
    """Vérifie que le modèle ONNX est valide"""
    print(f"🔍 Verifying ONNX model...")
    
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print("✅ ONNX model is valid")
        print(f"   Input: {onnx_model.graph.input[0].name}")
        print(f"   Output: {onnx_model.graph.output[0].name}")
        
        num_initializers = len(onnx_model.graph.initializer)
        print(f"   Initializers: {num_initializers}")
        
        if num_initializers < 100:
            print(f"   ⚠️  Very few initializers ({num_initializers})")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return False

def test_onnx_inference(onnx_path: str) -> bool:
    """Teste l'inférence avec ONNX Runtime"""
    print(f"🧪 Testing ONNX inference...")
    
    try:
        session = ort.InferenceSession(onnx_path)
        
        dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        outputs = session.run(['output'], {'input': dummy_input})
        
        assert outputs[0].shape == (1, 4), f"Unexpected output shape: {outputs[0].shape}"
        
        print("✅ ONNX inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Sample output: {outputs[0][0]}")
        
        dummy_input2 = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        outputs2 = session.run(['output'], {'input': dummy_input2})
        
        diff = np.abs(outputs[0] - outputs2[0]).max()
        print(f"   Output variation: {diff:.6f}")
        
        if diff < 1e-6:
            print(f"   ❌ Outputs are constant!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        return False

def compare_pytorch_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    num_tests: int = 10
) -> Tuple[bool, float]:
    """Compare les sorties PyTorch vs ONNX"""
    print(f"🔬 Comparing PyTorch vs ONNX outputs...")
    
    try:
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(DEVICE)
        
        onnx_session = ort.InferenceSession(onnx_path)
        
        max_diff = 0.0
        
        for i in range(num_tests):
            test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            
            with torch.no_grad():
                pytorch_input = torch.from_numpy(test_input).to(DEVICE)
                pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
            
            onnx_output = onnx_session.run(['output'], {'input': test_input})[0]
            
            diff = np.abs(pytorch_output - onnx_output).max()
            max_diff = max(max_diff, diff)
        
        tolerance = 1e-4
        match = max_diff < tolerance
        
        if match:
            print(f"✅ PyTorch and ONNX outputs match (max diff: {max_diff:.2e})")
        else:
            print(f"⚠️  PyTorch and ONNX outputs differ (max diff: {max_diff:.2e})")
        
        return match, max_diff
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False, float('inf')

def export_and_verify(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    verify: bool = True,
    test_inference: bool = True,
    compare: bool = True
) -> bool:
    """Pipeline complet d'export et vérification"""
    print("\n" + "="*70)
    print("🚀 ONNX EXPORT PIPELINE (LEGACY EXPORTER)")
    print("="*70)
    
    if not export_to_onnx(pytorch_model, onnx_path):
        return False
    
    if verify and not verify_onnx_model(onnx_path):
        return False
    
    if test_inference and not test_onnx_inference(onnx_path):
        return False
    
    if compare:
        match, max_diff = compare_pytorch_onnx(pytorch_model, onnx_path)
        if not match and max_diff > 0.01:
            print("\n⚠️  Warning: Large difference")
    
    print("\n✅ ONNX export pipeline complete!")
    print("="*70 + "\n")
    
    return True
