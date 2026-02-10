"""
Model Validator - Comparaison et validation des modèles
"""

import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, List
import os

DEVICE = 'cpu'
NUM_CLASSES = 4

# ============================================================================
# VALIDATION MODÈLE
# ============================================================================

def validate_model_stability(
    model: nn.Module,
    num_tests: int = 20
) -> Dict[str, float]:
    """
    Vérifie la stabilité d'un modèle PyTorch
    
    Returns:
        Dict avec metrics de stabilité
    """
    print(f"\n🔬 Testing model stability ({num_tests} tests)...")
    
    model.eval()
    model = model.to(DEVICE)
    
    outputs = []
    
    with torch.no_grad():
        for i in range(num_tests):
            test_input = torch.randn(1, 3, 224, 224, device=DEVICE)
            output = model(test_input).cpu().numpy()[0]
            outputs.append(output)
    
    outputs = np.array(outputs)
    
    # Calculer métriques
    mean_outputs = outputs.mean(axis=0)
    std_outputs = outputs.std(axis=0)
    variance = outputs.var(axis=0)
    
    metrics = {
        'mean': mean_outputs.tolist(),
        'std': std_outputs.tolist(),
        'variance': variance.tolist(),
        'max_std': std_outputs.max(),
        'max_variance': variance.max(),
        'is_stable': variance.max() >= 1e-6  # Pas constant
    }
    
    print(f"   Max variance: {metrics['max_variance']:.6f}")
    print(f"   Max std: {metrics['max_std']:.6f}")
    print(f"   Status: {'✅ STABLE' if metrics['is_stable'] else '❌ CONSTANT'}")
    
    return metrics

def validate_onnx_stability(
    onnx_path: str,
    num_tests: int = 20
) -> Dict[str, float]:
    """
    Vérifie la stabilité d'un modèle ONNX
    """
    print(f"\n🔬 Testing ONNX stability ({num_tests} tests)...")
    
    session = ort.InferenceSession(onnx_path)
    
    outputs = []
    
    for i in range(num_tests):
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = session.run(['output'], {'input': test_input})[0][0]
        outputs.append(output)
    
    outputs = np.array(outputs)
    
    mean_outputs = outputs.mean(axis=0)
    std_outputs = outputs.std(axis=0)
    variance = outputs.var(axis=0)
    
    metrics = {
        'mean': mean_outputs.tolist(),
        'std': std_outputs.tolist(),
        'variance': variance.tolist(),
        'max_std': std_outputs.max(),
        'max_variance': variance.max(),
        'is_stable': variance.max() >= 1e-6
    }
    
    print(f"   Max variance: {metrics['max_variance']:.6f}")
    print(f"   Max std: {metrics['max_std']:.6f}")
    print(f"   Status: {'✅ STABLE' if metrics['is_stable'] else '❌ CONSTANT'}")
    
    return metrics

# ============================================================================
# COMPARAISON MODÈLES
# ============================================================================

def compare_models(
    baseline_onnx_path: str,
    new_model: nn.Module,
    test_images: List[str] = None,
    num_random_tests: int = 50
) -> Dict:
    """
    Compare nouveau modèle vs baseline ONNX
    
    Returns:
        Dict avec métriques de comparaison et recommandation
    """
    print(f"\n{'='*70}")
    print(f"📊 COMPARING MODELS")
    print(f"{'='*70}\n")
    
    print(f"Baseline: {baseline_onnx_path}")
    print(f"New model: PyTorch (fine-tuned)")
    
    # 1. Charger baseline ONNX
    print(f"\n📦 Loading baseline ONNX...")
    baseline_session = ort.InferenceSession(baseline_onnx_path)
    print(f"   ✅ Loaded")
    
    # 2. Préparer nouveau modèle
    print(f"\n📦 Preparing new model...")
    new_model.train()  # Mode TRAIN pour compatibilité
    new_model = new_model.to(DEVICE)
    print(f"   ✅ Ready")
    
    # 3. Tests sur inputs aléatoires
    print(f"\n🧪 Running {num_random_tests} random tests...")
    
    baseline_outputs = []
    new_outputs = []
    
    for i in range(num_random_tests):
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Baseline ONNX
        baseline_out = baseline_session.run(['output'], {'input': test_input})[0][0]
        baseline_outputs.append(baseline_out)
        
        # New model
        with torch.no_grad():
            test_tensor = torch.from_numpy(test_input).to(DEVICE)
            new_out = new_model(test_tensor).cpu().numpy()[0]
            new_outputs.append(new_out)
    
    baseline_outputs = np.array(baseline_outputs)
    new_outputs = np.array(new_outputs)
    
    # 4. Calculer métriques
    print(f"\n📈 Computing metrics...")
    
    # Stabilité - FORCE conversion en scalar
    baseline_variance = float(baseline_outputs.var(axis=0).max())
    new_variance = float(new_outputs.var(axis=0).max())
    
    baseline_stable = baseline_variance >= 1e-6
    new_stable = new_variance >= 1e-6
    
    # Différence moyenne - FORCE conversion en scalar
    mean_diff = float(np.abs(baseline_outputs - new_outputs).mean())
    max_diff = float(np.abs(baseline_outputs - new_outputs).max())
    
    # Distribution des prédictions
    baseline_mean = baseline_outputs.mean(axis=0)
    new_mean = new_outputs.mean(axis=0)
    
    baseline_std = baseline_outputs.std(axis=0)
    new_std = new_outputs.std(axis=0)
    
    # 5. Décision
    print(f"\n🔍 Analysis:")
    print(f"   Baseline variance: {baseline_variance:.6f} {'✅' if baseline_stable else '❌'}")
    print(f"   New model variance: {new_variance:.6f} {'✅' if new_stable else '❌'}")
    print(f"   Mean difference: {mean_diff:.4f}")
    print(f"   Max difference: {max_diff:.4f}")
    
    # Critères de validation
    criteria = {
        'new_is_stable': bool(new_stable),
        'baseline_is_stable': bool(baseline_stable),
        'not_too_different': bool(mean_diff < 0.5),  # Pas trop éloigné du baseline
        'new_variance_reasonable': bool(0.001 < new_variance < 0.1),  # Variance raisonnable
    }
    
    all_pass = all(criteria.values())
    
    recommendation = "APPROVE" if all_pass else "REJECT"
    
    print(f"\n✅ Validation criteria:")
    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {criterion}")
    
    print(f"\n{'='*70}")
    print(f"🎯 RECOMMENDATION: {recommendation}")
    print(f"{'='*70}\n")
    
    if recommendation == "APPROVE":
        print("✅ New model is STABLE and performs reasonably")
        print("   → Safe to deploy as new ONNX")
    else:
        print("❌ New model failed validation")
        print("   → Keep baseline ONNX")
        
        if not new_stable:
            print("   → Reason: Model outputs are constant")
        if not criteria['not_too_different']:
            print("   → Reason: Too different from baseline")
        if not criteria['new_variance_reasonable']:
            print("   → Reason: Variance out of acceptable range")
    
    # Résultat
    result = {
        'recommendation': recommendation,
        'criteria': criteria,
        'metrics': {
            'baseline_variance': baseline_variance,  # Déjà float
            'new_variance': new_variance,  # Déjà float
            'mean_diff': mean_diff,  # Déjà float
            'max_diff': max_diff,  # Déjà float
            'baseline_mean': baseline_mean.tolist(),
            'new_mean': new_mean.tolist(),
            'baseline_std': baseline_std.tolist(),
            'new_std': new_std.tolist(),
        }
    }
    
    return result
