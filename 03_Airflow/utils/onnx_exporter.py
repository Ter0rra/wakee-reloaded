"""
ONNX Exporter - FIX BATCHNORM
Force eval mode ET désactive tous les BatchNorm
"""

import os
import torch
import torch.onnx
import torch.nn as nn
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
# EXPORT TO ONNX (FIX BATCHNORM)
# ============================================================================

def export_to_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    opset_version: int = 11
) -> bool:
    """
    Exporte le modèle PyTorch vers ONNX
    FIX: Force eval mode et désactive BatchNorm
    """
    print(f"🔄 Exporting PyTorch model to ONNX...")
    print(f"   Output: {onnx_path}")
    print(f"   Opset version: {opset_version}")
    
    try:
        # ✅ CRITIQUE : Force eval mode
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(DEVICE)
        
        # ✅ CRITIQUE : Désactive EXPLICITEMENT tous les BatchNorm
        print("\n🔧 Setting all BatchNorm layers to eval mode...")
        
        bn_count = 0
        for module in pytorch_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False  # ✅ IMPORTANT
                bn_count += 1
        
        print(f"   Set {bn_count} BatchNorm layers to eval mode")
        
        # ✅ TEST PyTorch AVANT export
        print("\n🧪 Testing PyTorch model BEFORE export...")
        with torch.no_grad():
            test1 = torch.randn(INPUT_SHAPE, device=DEVICE)
            test2 = torch.randn(INPUT_SHAPE, device=DEVICE)
            
            out1 = pytorch_model(test1)
            out2 = pytorch_model(test2)
            
            diff_pytorch = torch.abs(out1 - out2).max().item()
            print(f"   PyTorch variation: {diff_pytorch:.6f}")
            
            if diff_pytorch < 1e-6:
                print(f"   ❌ PyTorch model is constant BEFORE export!")
                return False
            
            print(f"   ✅ PyTorch OK before export")
        
        # Créer un input dummy
        dummy_input = torch.randn(INPUT_SHAPE, device=DEVICE)
        
        print(f"\n📤 Exporting to ONNX (opset {opset_version})...")
        
        # ✅ EXPORT ONNX
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
            training=torch.onnx.TrainingMode.EVAL,  # ✅ EXPLICIT EVAL MODE
        )

        # Vérifier la taille du fichier
        file_size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"   ONNX file size: {file_size_mb:.2f} MB")
        
        if file_size_mb < 10:
            print(f"   ⚠️  File size too small ({file_size_mb:.2f} MB)")
            return False
        
        print("✅ ONNX export successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# VERIFY ONNX MODEL
# ============================================================================

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

# ============================================================================
# TEST ONNX INFERENCE (AVEC DEBUG)
# ============================================================================

def test_onnx_inference(onnx_path: str) -> bool:
    """Teste l'inférence avec ONNX Runtime"""
    print(f"🧪 Testing ONNX inference...")
    
    try:
        session = ort.InferenceSession(onnx_path)
        
        # ✅ TEST avec plusieurs inputs pour être sûr
        print(f"\n   Running 5 test inferences...")
        
        all_outputs = []
        
        for i in range(5):
            test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            output = session.run(['output'], {'input': test_input})[0]
            all_outputs.append(output[0])
            print(f"   Test {i+1}: {output[0]}")
        
        # Vérifier la variance
        all_outputs = np.array(all_outputs)
        variance = np.var(all_outputs, axis=0)
        
        print(f"\n   Variance par classe:")
        emotions = ['boredom', 'confusion', 'engagement', 'frustration']
        for i, emotion in enumerate(emotions):
            print(f"      {emotion}: {variance[i]:.6f}")
        
        max_variance = variance.max()
        print(f"\n   Max variance: {max_variance:.6f}")
        
        if max_variance < 1e-6:
            print(f"   ❌ All outputs are CONSTANT (variance < 1e-6)!")
            print(f"   → Model is broken!")
            return False
        
        print(f"   ✅ Outputs vary correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# COMPARE PYTORCH VS ONNX
# ============================================================================

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

# ============================================================================
# FULL EXPORT PIPELINE
# ============================================================================

def export_and_verify(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    verify: bool = True,
    test_inference: bool = True,
    compare: bool = True
) -> bool:
    """Pipeline complet d'export et vérification"""
    print("\n" + "="*70)
    print("🚀 ONNX EXPORT PIPELINE (BATCHNORM FIX)")
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
