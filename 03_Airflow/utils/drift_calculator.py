"""
Drift Calculator - Evidently AI
Calcule les métriques de drift entre prédictions et annotations utilisateurs
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently import Report

# ============================================================================
# CONFIGURATION
# ============================================================================

DRIFT_THRESHOLD = 0.15  # Seuil MAE global pour déclencher retrain
EMOTION_COLUMNS = ['boredom', 'confusion', 'engagement', 'frustration']

# ============================================================================
# CALCUL MÉTRIQUES SIMPLES
# ============================================================================

def calculate_mae(predicted: pd.Series, actual: pd.Series) -> float:
    """Calcule Mean Absolute Error"""
    return np.mean(np.abs(predicted - actual))

def calculate_mse(predicted: pd.Series, actual: pd.Series) -> float:
    """Calcule Mean Squared Error"""
    return np.mean((predicted - actual) ** 2)

def calculate_rmse(predicted: pd.Series, actual: pd.Series) -> float:
    """Calcule Root Mean Squared Error"""
    return np.sqrt(calculate_mse(predicted, actual))

# ============================================================================
# CALCUL DRIFT METRICS
# ============================================================================

def calculate_drift_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcule les métriques de drift pour toutes les émotions
    
    Args:
        df (pd.DataFrame): DataFrame avec colonnes predicted_* et user_*
    
    Returns:
        dict: Métriques de drift {
            'mae_boredom': float,
            'mae_confusion': float,
            'mae_engagement': float,
            'mae_frustration': float,
            'mse_boredom': float,
            'mse_confusion': float,
            'mse_engagement': float,
            'mse_frustration': float,
            'drift_score': float  # MAE global
        }
    """
    if len(df) == 0:
        print("⚠️  No data to calculate drift")
        return {
            'mae_boredom': 0.0,
            'mae_confusion': 0.0,
            'mae_engagement': 0.0,
            'mae_frustration': 0.0,
            'mse_boredom': 0.0,
            'mse_confusion': 0.0,
            'mse_engagement': 0.0,
            'mse_frustration': 0.0,
            'drift_score': 0.0
        }
    
    metrics = {}
    mae_scores = []
    
    for emotion in EMOTION_COLUMNS:
        predicted_col = f'predicted_{emotion}'
        user_col = f'user_{emotion}'
        
        # Vérifie que les colonnes existent
        if predicted_col not in df.columns or user_col not in df.columns:
            print(f"⚠️  Missing columns for {emotion}")
            metrics[f'mae_{emotion}'] = 0.0
            metrics[f'mse_{emotion}'] = 0.0
            continue
        
        # Calcule MAE et MSE
        mae = calculate_mae(df[predicted_col], df[user_col])
        mse = calculate_mse(df[predicted_col], df[user_col])
        
        metrics[f'mae_{emotion}'] = float(mae)
        metrics[f'mse_{emotion}'] = float(mse)
        mae_scores.append(mae)
        
        print(f"  {emotion.capitalize():12} - MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    # Drift score global = moyenne des MAE
    if mae_scores:
        drift_score = np.mean(mae_scores)
    else:
        drift_score = 0.0
    
    metrics['drift_score'] = float(drift_score)
    
    print(f"\n📊 Global Drift Score: {drift_score:.4f}")
    
    return metrics

# ============================================================================
# CHECK DRIFT THRESHOLD
# ============================================================================

def check_drift_threshold(drift_score: float, threshold: float = DRIFT_THRESHOLD) -> bool:
    """
    Vérifie si le drift dépasse le seuil
    
    Args:
        drift_score (float): Score de drift global
        threshold (float): Seuil de détection
    
    Returns:
        bool: True si drift détecté
    """
    drift_detected = drift_score > threshold
    
    if drift_detected:
        print(f"🚨 DRIFT DETECTED! Score {drift_score:.4f} > threshold {threshold}")
    else:
        print(f"✅ No drift. Score {drift_score:.4f} <= threshold {threshold}")
    
    return drift_detected

# ============================================================================
# EVIDENTLY AI REPORT (AVANCÉ)
# ============================================================================

def generate_evidently_report(df: pd.DataFrame) -> Dict:
    """
    Génère un rapport Evidently AI détaillé
    
    Args:
        df (pd.DataFrame): DataFrame avec predicted_* et user_*
    
    Returns:
        dict: Rapport Evidently au format JSON
    """
    if len(df) < 10:
        print("⚠️  Not enough data for Evidently report (need >= 10 samples)")
        return {}
    
    try:
        # Prépare les données pour Evidently
        # Reference = prédictions du modèle
        # Current = annotations utilisateurs
        
        reference_data = pd.DataFrame({
            'boredom': df['predicted_boredom'],
            'confusion': df['predicted_confusion'],
            'engagement': df['predicted_engagement'],
            'frustration': df['predicted_frustration']
        })
        
        current_data = pd.DataFrame({
            'boredom': df['user_boredom'],
            'confusion': df['user_confusion'],
            'engagement': df['user_engagement'],
            'frustration': df['user_frustration']
        })
        
        # Crée le rapport Evidently
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Convertit en JSON
        report_dict = report.as_dict()
        
        print("✅ Evidently report generated")
        
        return report_dict
        
    except Exception as e:
        print(f"❌ Error generating Evidently report: {e}")
        return {}

# ============================================================================
# ANALYSE DÉTAILLÉE
# ============================================================================

def analyze_drift_by_emotion(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyse détaillée du drift par émotion
    
    Returns:
        dict: {
            'boredom': {
                'mae': float,
                'mse': float,
                'rmse': float,
                'mean_predicted': float,
                'mean_user': float,
                'diff': float
            },
            ...
        }
    """
    analysis = {}
    
    for emotion in EMOTION_COLUMNS:
        predicted_col = f'predicted_{emotion}'
        user_col = f'user_{emotion}'
        
        if predicted_col not in df.columns or user_col not in df.columns:
            continue
        
        mae = calculate_mae(df[predicted_col], df[user_col])
        mse = calculate_mse(df[predicted_col], df[user_col])
        rmse = calculate_rmse(df[predicted_col], df[user_col])
        
        mean_predicted = df[predicted_col].mean()
        mean_user = df[user_col].mean()
        diff = mean_user - mean_predicted
        
        analysis[emotion] = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mean_predicted': float(mean_predicted),
            'mean_user': float(mean_user),
            'diff': float(diff)
        }
    
    return analysis

# ============================================================================
# HELPER : Drift Trend
# ============================================================================

def calculate_drift_trend(recent_reports: list) -> str:
    """
    Calcule la tendance du drift (augmente/diminue/stable)
    
    Args:
        recent_reports (list): Liste des derniers rapports (dict)
    
    Returns:
        str: 'increasing', 'decreasing', 'stable'
    """
    if len(recent_reports) < 2:
        return 'stable'
    
    scores = [r['drift_score'] for r in recent_reports]
    
    # Régression linéaire simple
    x = np.arange(len(scores))
    slope = np.polyfit(x, scores, 1)[0]
    
    if slope > 0.01:
        return 'increasing'
    elif slope < -0.01:
        return 'decreasing'
    else:
        return 'stable'
