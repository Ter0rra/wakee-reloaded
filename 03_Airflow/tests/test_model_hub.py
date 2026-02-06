"""
Tests Pytest pour HuggingFace Model Hub
"""

import pytest
from huggingface_hub import hf_hub_download, list_repo_files
import os
import numpy as np
import onnxruntime as ort

# ============================================================================
# TESTS Download
# ============================================================================

def test_can_download_model(hf_model_repo):
    """Vérifie qu'on peut télécharger le modèle"""
    model_path = hf_hub_download(
        repo_id=hf_model_repo,
        filename="model.onnx",
        cache_dir="/tmp/pytest_hf_cache"
    )
    assert os.path.exists(model_path)

def test_model_file_is_valid_onnx(hf_model_repo):
    """Vérifie que le modèle ONNX est valide"""
    model_path = hf_hub_download(
        repo_id=hf_model_repo,
        filename="model.onnx",
        cache_dir="/tmp/pytest_hf_cache"
    )
    
    # Essaye de charger avec ONNX Runtime
    session = ort.InferenceSession(model_path)
    assert session is not None

def test_model_has_correct_input_shape(hf_model_repo):
    """Vérifie que le modèle attend le bon input (1, 3, 224, 224)"""
    model_path = hf_hub_download(
        repo_id=hf_model_repo,
        filename="model.onnx",
        cache_dir="/tmp/pytest_hf_cache"
    )
    
    session = ort.InferenceSession(model_path)
    input_shape = session.get_inputs()[0].shape
    
    assert input_shape == [1, 3, 224, 224]

def test_model_has_correct_output_shape(hf_model_repo):
    """Vérifie que le modèle output 4 classes"""
    model_path = hf_hub_download(
        repo_id=hf_model_repo,
        filename="model.onnx",
        cache_dir="/tmp/pytest_hf_cache"
    )
    
    session = ort.InferenceSession(model_path)
    output_shape = session.get_outputs()[0].shape
    
    assert output_shape[1] == 4  # 4 émotions

# ============================================================================
# TESTS Repository
# ============================================================================

def test_repo_contains_required_files(hf_model_repo):
    """Vérifie que le repo contient les fichiers requis"""
    files = list_repo_files(repo_id=hf_model_repo)
    
    required_files = ['model.onnx', 'README.md']
    for required_file in required_files:
        assert required_file in files

def test_readme_exists(hf_model_repo):
    """Vérifie que README.md existe et est téléchargeable"""
    readme_path = hf_hub_download(
        repo_id=hf_model_repo,
        filename="README.md",
        cache_dir="/tmp/pytest_hf_cache"
    )
    assert os.path.exists(readme_path)

# ============================================================================
# TESTS Inference
# ============================================================================

def test_model_inference_works(hf_model_repo):
    """Vérifie qu'on peut faire une inférence"""
    
    model_path = hf_hub_download(
        repo_id=hf_model_repo,
        filename="model.onnx",
        cache_dir="/tmp/pytest_hf_cache"
    )
    
    session = ort.InferenceSession(model_path)
    
    # Crée un input dummy
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Inférence
    outputs = session.run(['output'], {'input': dummy_input})
    
    assert outputs[0].shape == (1, 4)
