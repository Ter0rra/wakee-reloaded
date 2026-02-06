"""
Tests Pytest pour l'API Wakee
Vérifie les endpoints /health et /predict
"""

import pytest
import requests
from PIL import Image
import io
import time

# ============================================================================
# TESTS /health
# ============================================================================

def test_health_endpoint_returns_200(api_url):
    """Vérifie que /health retourne 200 OK"""
    response = requests.get(f"{api_url}/health", timeout=10)
    assert response.status_code == 200

def test_health_returns_json(api_url):
    """Vérifie que /health retourne du JSON valide"""
    response = requests.get(f"{api_url}/health", timeout=10)
    data = response.json()
    assert isinstance(data, dict)

def test_health_status_healthy(api_url):
    """Vérifie que le status est 'healthy'"""
    response = requests.get(f"{api_url}/health", timeout=10)
    data = response.json()
    assert data.get('status') == 'healthy'

def test_health_model_loaded(api_url):
    """Vérifie que le modèle ONNX est chargé"""
    response = requests.get(f"{api_url}/health", timeout=10)
    data = response.json()
    assert data.get('model_loaded') == True

# ============================================================================
# TESTS /predict
# ============================================================================

def test_predict_endpoint_accepts_image(api_url, test_image_small):
    """Vérifie que /predict accepte une image"""
    files = {'file': ('test.jpg', test_image_small, 'image/jpeg')}
    response = requests.post(f"{api_url}/predict", files=files, timeout=30)
    assert response.status_code == 200

def test_predict_returns_all_emotions(api_url, test_image_small):
    """Vérifie que /predict retourne les 4 émotions"""
    files = {'file': ('test.jpg', test_image_small, 'image/jpeg')}
    response = requests.post(f"{api_url}/predict", files=files, timeout=30)
    data = response.json()
    
    required_keys = ['boredom', 'confusion', 'engagement', 'frustration']
    for key in required_keys:
        assert key in data

def test_predict_scores_in_valid_range(api_url, test_image_small):
    """Vérifie que les scores sont entre 0 et 3"""
    files = {'file': ('test.jpg', test_image_small, 'image/jpeg')}
    response = requests.post(f"{api_url}/predict", files=files, timeout=30)
    data = response.json()
    
    emotions = ['boredom', 'confusion', 'engagement', 'frustration']
    for emotion in emotions:
        score = data[emotion]
        assert 0 <= score <= 3

def test_predict_response_time(api_url, test_image_small):
    """Vérifie que /predict répond en moins de 5s"""
    files = {'file': ('test.jpg', test_image_small, 'image/jpeg')}
    
    start = time.time()
    response = requests.post(f"{api_url}/predict", files=files, timeout=30)
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 5.0
