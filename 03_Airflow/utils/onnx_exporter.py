"""
ONNX Exporter - Convert PyTorch to ONNX
Exporte le modèle PyTorch vers ONNX pour l'API
"""

import torch
import torch.onnx
import onnx
import onnxscript
import onnxruntime as ort
import numpy as np
from typing import Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = 'cpu'
INPUT_SHAPE = (1, 3, 224, 224)  # Batch, Channels, Height, Width

# ============================================================================
# EXPORT TO ONNX
# ============================================================================

def export_to_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    opset_version: int = 11
) -> bool:
    """
    Exporte le modèle PyTorch vers ONNX
    
    Args:
        pytorch_model (nn.Module): Modèle PyTorch
        onnx_path (str): Chemin de sauvegarde ONNX
        opset_version (int): Version ONNX opset
    
    Returns:
        bool: True si succès
    """
    print(f"🔄 Exporting PyTorch model to ONNX...")
    print(f"   Output: {onnx_path}")
    print(f"   Opset version: {opset_version}")
    
    try:
        # Mettre le modèle en mode eval
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(DEVICE)
        
        # Créer un input dummy
        dummy_input = torch.randn(INPUT_SHAPE, device=DEVICE)
        
        # Export ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        
        print("✅ ONNX export successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return False

# ============================================================================
# VERIFY ONNX MODEL
# ============================================================================

def verify_onnx_model(onnx_path: str) -> bool:
    """
    Vérifie que le modèle ONNX est valide
    
    Args:
        onnx_path (str): Chemin du modèle ONNX
    
    Returns:
        bool: True si valide
    """
    print(f"🔍 Verifying ONNX model...")
    
    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Check model
        onnx.checker.check_model(onnx_model)
        
        print("✅ ONNX model is valid")
        
        # Affiche les infos
        print(f"   Input: {onnx_model.graph.input[0].name}")
        print(f"   Output: {onnx_model.graph.output[0].name}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model verification failed: {e}")
        return False

# ============================================================================
# TEST ONNX INFERENCE
# ============================================================================

def test_onnx_inference(onnx_path: str) -> bool:
    """
    Teste l'inférence avec ONNX Runtime
    
    Args:
        onnx_path (str): Chemin du modèle ONNX
    
    Returns:
        bool: True si inférence réussie
    """
    print(f"🧪 Testing ONNX inference...")
    
    try:
        # Créer une session ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        
        # Créer un input dummy
        dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        
        # Inférence
        outputs = session.run(['output'], {'input': dummy_input})
        
        # Vérifie la sortie
        assert outputs[0].shape == (1, 4), f"Unexpected output shape: {outputs[0].shape}"
        
        print("✅ ONNX inference successful")
        print(f"   Output shape: {outputs[0].shape}")
        print(f"   Sample output: {outputs[0][0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        return False

# ============================================================================
# COMPARE PYTORCH VS ONNX
# ============================================================================

def compare_pytorch_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    num_tests: int = 10
) -> Tuple[bool, float]:
    """
    Compare les sorties PyTorch vs ONNX
    
    Args:
        pytorch_model (nn.Module): Modèle PyTorch
        onnx_path (str): Chemin ONNX
        num_tests (int): Nombre de tests
    
    Returns:
        tuple: (match, max_diff)
    """
    print(f"🔬 Comparing PyTorch vs ONNX outputs...")
    
    try:
        # Setup
        pytorch_model.eval()
        pytorch_model = pytorch_model.to(DEVICE)
        
        onnx_session = ort.InferenceSession(onnx_path)
        
        max_diff = 0.0
        
        for i in range(num_tests):
            # Créer input aléatoire
            test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_input = torch.from_numpy(test_input).to(DEVICE)
                pytorch_output = pytorch_model(pytorch_input).cpu().numpy()
            
            # ONNX inference
            onnx_output = onnx_session.run(['output'], {'input': test_input})[0]
            
            # Compare
            diff = np.abs(pytorch_output - onnx_output).max()
            max_diff = max(max_diff, diff)
        
        # Tolerance
        tolerance = 1e-5
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
    """
    Pipeline complet d'export et vérification
    
    Args:
        pytorch_model (nn.Module): Modèle PyTorch
        onnx_path (str): Chemin de sauvegarde ONNX
        verify (bool): Vérifier le modèle ONNX
        test_inference (bool): Tester l'inférence
        compare (bool): Comparer PyTorch vs ONNX
    
    Returns:
        bool: True si tout réussit
    """
    print("\n" + "="*70)
    print("🚀 ONNX EXPORT PIPELINE")
    print("="*70)
    
    # 1. Export
    if not export_to_onnx(pytorch_model, onnx_path):
        return False
    
    # 2. Verify
    if verify and not verify_onnx_model(onnx_path):
        return False
    
    # 3. Test inference
    if test_inference and not test_onnx_inference(onnx_path):
        return False
    
    # 4. Compare
    if compare:
        match, max_diff = compare_pytorch_onnx(pytorch_model, onnx_path)
        if not match:
            print("⚠️  Warning: PyTorch and ONNX outputs differ significantly")
    
    print("\n✅ ONNX export pipeline complete!")
    print("="*70 + "\n")
    
    return True
