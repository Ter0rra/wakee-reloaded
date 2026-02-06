"""
Tests Pytest pour Drift Detection
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajoute le chemin utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.drift_calculator import (
    calculate_mae,
    calculate_mse,
    calculate_drift_metrics,
    check_drift_threshold
)

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_annotations():
    """Crée un DataFrame d'annotations de test"""
    np.random.seed(42)
    n = 50
    
    df = pd.DataFrame({
        'predicted_boredom': np.random.uniform(0, 3, n),
        'predicted_confusion': np.random.uniform(0, 3, n),
        'predicted_engagement': np.random.uniform(0, 3, n),
        'predicted_frustration': np.random.uniform(0, 3, n),
        'user_boredom': np.random.uniform(0, 3, n),
        'user_confusion': np.random.uniform(0, 3, n),
        'user_engagement': np.random.uniform(0, 3, n),
        'user_frustration': np.random.uniform(0, 3, n),
    })
    
    return df

@pytest.fixture
def perfect_annotations():
    """Annotations parfaites (pas de drift)"""
    n = 50
    
    values = np.random.uniform(0, 3, n)
    
    df = pd.DataFrame({
        'predicted_boredom': values,
        'user_boredom': values,  # Identique
        'predicted_confusion': values,
        'user_confusion': values,
        'predicted_engagement': values,
        'user_engagement': values,
        'predicted_frustration': values,
        'user_frustration': values,
    })
    
    return df

# ============================================================================
# TESTS MAE/MSE
# ============================================================================

def test_calculate_mae_perfect():
    """MAE doit être 0 pour prédictions parfaites"""
    predicted = pd.Series([1.0, 2.0, 3.0])
    actual = pd.Series([1.0, 2.0, 3.0])
    
    mae = calculate_mae(predicted, actual)
    assert mae == 0.0

def test_calculate_mae_non_zero():
    """MAE doit être > 0 pour prédictions imparfaites"""
    predicted = pd.Series([1.0, 2.0, 3.0])
    actual = pd.Series([1.5, 2.5, 3.5])
    
    mae = calculate_mae(predicted, actual)
    assert mae == 0.5

def test_calculate_mse():
    """Test MSE"""
    predicted = pd.Series([1.0, 2.0, 3.0])
    actual = pd.Series([2.0, 3.0, 4.0])
    
    mse = calculate_mse(predicted, actual)
    assert mse == 1.0

# ============================================================================
# TESTS Drift Metrics
# ============================================================================

def test_calculate_drift_metrics_structure(sample_annotations):
    """Vérifie que calculate_drift_metrics retourne toutes les métriques"""
    metrics = calculate_drift_metrics(sample_annotations)
    
    required_keys = [
        'mae_boredom', 'mae_confusion', 'mae_engagement', 'mae_frustration',
        'mse_boredom', 'mse_confusion', 'mse_engagement', 'mse_frustration',
        'drift_score'
    ]
    
    for key in required_keys:
        assert key in metrics

def test_calculate_drift_metrics_perfect(perfect_annotations):
    """Drift doit être 0 pour annotations parfaites"""
    metrics = calculate_drift_metrics(perfect_annotations)
    
    assert metrics['drift_score'] == 0.0
    assert metrics['mae_boredom'] == 0.0
    assert metrics['mae_confusion'] == 0.0
    assert metrics['mae_engagement'] == 0.0
    assert metrics['mae_frustration'] == 0.0

def test_calculate_drift_metrics_empty():
    """Drift pour DataFrame vide"""
    df = pd.DataFrame()
    metrics = calculate_drift_metrics(df)
    
    assert metrics['drift_score'] == 0.0

# ============================================================================
# TESTS Drift Threshold
# ============================================================================

def test_check_drift_threshold_below():
    """Pas de drift si score < seuil"""
    drift_detected = check_drift_threshold(drift_score=0.10, threshold=0.15)
    assert drift_detected == False

def test_check_drift_threshold_above():
    """Drift détecté si score > seuil"""
    drift_detected = check_drift_threshold(drift_score=0.20, threshold=0.15)
    assert drift_detected == True

def test_check_drift_threshold_equal():
    """Pas de drift si score == seuil"""
    drift_detected = check_drift_threshold(drift_score=0.15, threshold=0.15)
    assert drift_detected == False

# ============================================================================
# TESTS Robustesse
# ============================================================================

def test_drift_metrics_with_nans():
    """Gestion des NaN"""
    df = pd.DataFrame({
        'predicted_boredom': [1.0, np.nan, 2.0],
        'user_boredom': [1.0, 2.0, 2.0],
        'predicted_confusion': [1.0, 2.0, 3.0],
        'user_confusion': [1.0, 2.0, 3.0],
        'predicted_engagement': [1.0, 2.0, 3.0],
        'user_engagement': [1.0, 2.0, 3.0],
        'predicted_frustration': [1.0, 2.0, 3.0],
        'user_frustration': [1.0, 2.0, 3.0],
    })
    
    # Ne devrait pas crasher
    metrics = calculate_drift_metrics(df)
    assert 'drift_score' in metrics

def test_drift_metrics_large_dataset():
    """Test avec grand dataset"""
    n = 10000
    df = pd.DataFrame({
        'predicted_boredom': np.random.uniform(0, 3, n),
        'user_boredom': np.random.uniform(0, 3, n),
        'predicted_confusion': np.random.uniform(0, 3, n),
        'user_confusion': np.random.uniform(0, 3, n),
        'predicted_engagement': np.random.uniform(0, 3, n),
        'user_engagement': np.random.uniform(0, 3, n),
        'predicted_frustration': np.random.uniform(0, 3, n),
        'user_frustration': np.random.uniform(0, 3, n),
    })
    
    metrics = calculate_drift_metrics(df)
    
    # Avec données aléatoires, drift devrait être > 0 mais raisonnable
    assert 0 < metrics['drift_score'] < 2.0
