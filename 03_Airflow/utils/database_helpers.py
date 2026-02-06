"""
Database Helpers - NeonDB
Fonctions utilitaires pour interagir avec la base de données
"""

from sqlalchemy import create_engine, text
import pandas as pd
import os
from datetime import datetime, timedelta
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

NEON_DATABASE_URL = os.getenv("NEONDB_WR")

# ============================================================================
# ENGINE
# ============================================================================

def get_db_engine():
    """Crée un moteur SQLAlchemy pour NeonDB"""
    if not NEON_DATABASE_URL:
        raise ValueError("NEONDB_WR not configured")
    
    return create_engine(NEON_DATABASE_URL)

# ============================================================================
# FETCH ANNOTATIONS
# ============================================================================

def fetch_recent_annotations(days=7, validated_only=True):
    """
    Récupère les annotations récentes depuis NeonDB
    
    Args:
        days (int): Nombre de jours à récupérer
        validated_only (bool): Uniquement les annotations validées
    
    Returns:
        pd.DataFrame: Annotations avec colonnes predicted_* et user_*
    """
    engine = get_db_engine()
    
    query = text("""
        SELECT 
            id,
            img_name,
            s3_path,
            predicted_boredom,
            predicted_confusion,
            predicted_engagement,
            predicted_frustration,
            user_boredom,
            user_confusion,
            user_engagement,
            user_frustration,
            timestamp 
        FROM emotion_labels
        WHERE timestamp >= CURRENT_DATE - INTERVAL ':days day' 
    """)
    
    if validated_only:
        query = text(str(query) + " AND is_validated = TRUE")
    
    query = text(str(query) + " ORDER BY timestamp DESC")
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'days': days})
    
    print(f"📊 Fetched {len(df)} annotations from last {days} days")
    
    return df

# ============================================================================
# SAVE DRIFT REPORT
# ============================================================================

def save_drift_report(
    report_date,
    drift_detected,
    drift_score,
    metrics,
    num_samples,
    retrain_triggered=False,
    report_url=None
):
    """
    Sauvegarde un rapport de drift dans NeonDB
    Compatible avec le schéma existant
    
    Args:
        report_date (datetime): Date du rapport
        drift_detected (bool): Drift détecté ou non
        drift_score (float): Score de drift global
        metrics (dict): Métriques MAE par émotion
        num_samples (int): Nombre d'échantillons analysés
        retrain_triggered (bool): Si retrain a été déclenché
        report_url (str): URL du rapport Evidently (optionnel)
    
    Returns:
        int: ID du rapport créé
    """
    engine = get_db_engine()
    
    query = text("""
        INSERT INTO drift_reports (
            report_date,
            drift_score_boredom,
            drift_score_confusion,
            drift_score_engagement,
            drift_score_frustration,
            drift_score_global,
            drift_detected,
            n_samples_analyzed,
            report_url,
            retrain_triggered
        ) VALUES (
            :report_date,
            :drift_score_boredom,
            :drift_score_confusion,
            :drift_score_engagement,
            :drift_score_frustration,
            :drift_score_global,
            :drift_detected,
            :n_samples_analyzed,
            :report_url,
            :retrain_triggered
        )
        RETURNING id
    """)
    
    params = {
        'report_date': report_date,
        'drift_score_boredom': metrics.get('mae_boredom', 0.0),
        'drift_score_confusion': metrics.get('mae_confusion', 0.0),
        'drift_score_engagement': metrics.get('mae_engagement', 0.0),
        'drift_score_frustration': metrics.get('mae_frustration', 0.0),
        'drift_score_global': drift_score,
        'drift_detected': drift_detected,
        'n_samples_analyzed': num_samples,
        'report_url': report_url,
        'retrain_triggered': retrain_triggered
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        conn.commit()
        report_id = result.fetchone()[0]
    
    print(f"✅ Drift report saved (ID: {report_id})")
    
    return report_id

# ============================================================================
# GET LATEST DRIFT REPORT
# ============================================================================

def get_latest_drift_report():
    """
    Récupère le dernier rapport de drift
    
    Returns:
        dict: Dernier rapport ou None
    """
    engine = get_db_engine()
    
    query = text("""
        SELECT * FROM drift_reports
        ORDER BY report_date DESC
        LIMIT 1
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
    
    if result:
        return dict(result._mapping)
    else:
        return None

# ============================================================================
# UPDATE RETRAIN TRIGGERED
# ============================================================================

def update_retrain_triggered(report_id):
    """
    Marque qu'un retrain a été déclenché pour ce rapport
    
    Args:
        report_id (int): ID du rapport
    """
    engine = get_db_engine()
    
    query = text("""
        UPDATE drift_reports
        SET retrain_triggered = TRUE
        WHERE id = :report_id
    """)
    
    with engine.connect() as conn:
        conn.execute(query, {'report_id': report_id})
        conn.commit()
    
    print(f"✅ Updated report {report_id}: retrain_triggered = TRUE")

# ============================================================================
# COUNT DRIFT DETECTIONS
# ============================================================================

def count_drift_detections(days=30):
    """
    Compte le nombre de drifts détectés sur N jours
    
    Args:
        days (int): Nombre de jours
    
    Returns:
        int: Nombre de drifts détectés
    """
    engine = get_db_engine()
    
    query = text("""
        SELECT COUNT(*) 
        FROM drift_reports
        WHERE drift_detected = TRUE
          AND report_date >= NOW() - INTERVAL ':days days'
    """)
    
    with engine.connect() as conn:
        count = conn.execute(query, {'days': days}).fetchone()[0]
    
    return count
