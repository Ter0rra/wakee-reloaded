"""
DAG Drift Detection - Wakee Reloaded
Détecte le drift du modèle en comparant prédictions vs annotations utilisateurs
Schedule : Quotidien (2h du matin)
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys
import os

# ✅ Ajoute le chemin parent (03_Airflow) au PYTHONPATH
# Permet d'importer depuis utils/
AIRFLOW_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AIRFLOW_HOME not in sys.path:
    sys.path.insert(0, AIRFLOW_HOME)

from utils.database_helpers import (
    fetch_recent_annotations,
    save_drift_report,
    get_latest_drift_report,
    count_drift_detections
)
from utils.drift_calculator import (
    calculate_drift_metrics,
    check_drift_threshold,
    generate_evidently_report,
    analyze_drift_by_emotion
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DRIFT_THRESHOLD = 0.15  # Seuil MAE pour déclencher retrain
LOOKBACK_DAYS = 7  # Nombre de jours d'annotations à analyser
MIN_SAMPLES = 5  # Minimum d'annotations requises

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'drift_detection_daily',
    default_args=default_args,
    description='Détection quotidienne du drift du modèle Wakee',
    schedule='0 2 * * *',  # Tous les jours à 2h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['drift', 'monitoring', 'mlops'],
)

# ============================================================================
# TASK 1 : Fetch Recent Annotations
# ============================================================================

def task_fetch_annotations(**context):
    """Récupère les annotations récentes depuis NeonDB"""
    print(f"🔍 Fetching annotations from last {LOOKBACK_DAYS} days...")
    
    df = fetch_recent_annotations(days=LOOKBACK_DAYS, validated_only=True)
    
    if len(df) < MIN_SAMPLES:
        print(f"⚠️  Not enough data: {len(df)} samples (minimum: {MIN_SAMPLES})")
        print("   Skipping drift detection for today")
        context['task_instance'].xcom_push(key='skip_drift', value=True)
        context['task_instance'].xcom_push(key='num_samples', value=len(df))
        return
    
    print(f"✅ Fetched {len(df)} validated annotations")
    
    # Statistiques rapides
    print(f"\n📊 Data Summary:")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Mean predicted engagement: {df['predicted_engagement'].mean():.2f}")
    print(f"   Mean user engagement: {df['user_engagement'].mean():.2f}")
    
    # Push data to XCom (en JSON pour sérialisation)
    context['task_instance'].xcom_push(key='skip_drift', value=False)
    context['task_instance'].xcom_push(key='num_samples', value=len(df))
    context['task_instance'].xcom_push(key='annotations_json', value=df.to_json())

# ============================================================================
# TASK 2 : Calculate Drift Metrics
# ============================================================================

def task_calculate_drift(**context):
    """Calcule les métriques de drift"""
    import pandas as pd
    
    ti = context['task_instance']
    
    # Vérifie si on doit skip
    skip_drift = ti.xcom_pull(task_ids='fetch_annotations', key='skip_drift')
    if skip_drift:
        print("⏭️  Skipping drift calculation (not enough data)")
        ti.xcom_push(key='drift_detected', value=False)
        ti.xcom_push(key='drift_score', value=0.0)
        return
    
    # Récupère les annotations
    annotations_json = ti.xcom_pull(task_ids='fetch_annotations', key='annotations_json')
    df = pd.read_json(annotations_json)
    
    print("📊 Calculating drift metrics...")
    
    # Calcule les métriques
    metrics = calculate_drift_metrics(df)
    drift_score = metrics['drift_score']
    
    # Vérifie le seuil
    drift_detected = check_drift_threshold(drift_score, DRIFT_THRESHOLD)
    
    # Analyse détaillée
    detailed_analysis = analyze_drift_by_emotion(df)
    
    print(f"\n📈 Detailed Analysis:")
    for emotion, stats in detailed_analysis.items():
        print(f"  {emotion.capitalize()}:")
        print(f"    MAE: {stats['mae']:.4f}")
        print(f"    Mean Predicted: {stats['mean_predicted']:.2f}")
        print(f"    Mean User: {stats['mean_user']:.2f}")
        print(f"    Diff: {stats['diff']:+.2f}")
    
    # Push to XCom
    ti.xcom_push(key='drift_detected', value=drift_detected)
    ti.xcom_push(key='drift_score', value=drift_score)
    ti.xcom_push(key='metrics', value=metrics)
    ti.xcom_push(key='detailed_analysis', value=detailed_analysis)

# ============================================================================
# TASK 3 : Generate Evidently Report
# ============================================================================

def task_generate_report(**context):
    """Génère un rapport Evidently AI détaillé"""
    import pandas as pd
    
    ti = context['task_instance']
    
    # Vérifie si on doit skip
    skip_drift = ti.xcom_pull(task_ids='fetch_annotations', key='skip_drift')
    if skip_drift:
        print("⏭️  Skipping report generation")
        ti.xcom_push(key='evidently_report', value={})
        return
    
    # Récupère les annotations
    annotations_json = ti.xcom_pull(task_ids='fetch_annotations', key='annotations_json')
    df = pd.read_json(annotations_json)
    
    print("📄 Generating Evidently AI report...")
    
    # Génère le rapport
    report = generate_evidently_report(df)
    
    # Push to XCom
    ti.xcom_push(key='evidently_report', value=report)

# ============================================================================
# TASK 4 : Save Report to Database
# ============================================================================

def task_save_report(**context):
    """Sauvegarde le rapport dans NeonDB"""
    ti = context['task_instance']
    
    # Récupère les données
    skip_drift = ti.xcom_pull(task_ids='fetch_annotations', key='skip_drift')
    num_samples = ti.xcom_pull(task_ids='fetch_annotations', key='num_samples')
    
    if skip_drift:
        print("⏭️  Skipping report save (not enough data)")
        return
    
    drift_detected = ti.xcom_pull(task_ids='calculate_drift', key='drift_detected')
    drift_score = ti.xcom_pull(task_ids='calculate_drift', key='drift_score')
    metrics = ti.xcom_pull(task_ids='calculate_drift', key='metrics')
    
    print("💾 Saving drift report to database...")
    
    # Sauvegarde dans NeonDB (compatible avec schéma existant)
    report_id = save_drift_report(
        report_date=datetime.now(),
        drift_detected=drift_detected,
        drift_score=drift_score,
        metrics=metrics,
        num_samples=num_samples,
        retrain_triggered=False,  # Sera mis à jour si retrain déclenché
        report_url=None
    )
    
    print(f"✅ Report saved with ID: {report_id}")
    
    # Stats historiques
    drift_count = count_drift_detections(days=30)
    print(f"\n📊 Historical Stats:")
    print(f"   Drifts detected in last 30 days: {drift_count}")
    
    # Push to XCom
    ti.xcom_push(key='report_id', value=report_id)

# ============================================================================
# TASK 5 : Generate Summary
# ============================================================================

def task_generate_summary(**context):
    """Génère un résumé du rapport de drift"""
    ti = context['task_instance']
    
    # Récupère les données
    skip_drift = ti.xcom_pull(task_ids='fetch_annotations', key='skip_drift')
    num_samples = ti.xcom_pull(task_ids='fetch_annotations', key='num_samples')
    
    print("\n" + "="*70)
    print("📊 DRIFT DETECTION SUMMARY")
    print("="*70)
    
    if skip_drift:
        print(f"⚠️  Insufficient data: {num_samples} samples (minimum: {MIN_SAMPLES})")
        print("   No drift analysis performed today")
    else:
        drift_detected = ti.xcom_pull(task_ids='calculate_drift', key='drift_detected')
        drift_score = ti.xcom_pull(task_ids='calculate_drift', key='drift_score')
        metrics = ti.xcom_pull(task_ids='calculate_drift', key='metrics')
        
        print(f"📈 Samples analyzed: {num_samples}")
        print(f"📊 Global drift score: {drift_score:.4f} (threshold: {DRIFT_THRESHOLD})")
        print(f"")
        print(f"MAE by emotion:")
        print(f"  - Boredom:    {metrics['mae_boredom']:.4f}")
        print(f"  - Confusion:  {metrics['mae_confusion']:.4f}")
        print(f"  - Engagement: {metrics['mae_engagement']:.4f}")
        print(f"  - Frustration: {metrics['mae_frustration']:.4f}")
        print(f"")
        
        if drift_detected:
            print(f"🚨 DRIFT DETECTED! Model retraining will be triggered.")
        else:
            print(f"✅ No drift detected. Model performing well.")
    
    print("="*70 + "\n")

# ============================================================================
# TASK 6 : Trigger Retrain (Conditional)
# ============================================================================

def task_check_trigger_retrain(**context):
    """Vérifie si on doit déclencher le retrain"""
    ti = context['task_instance']
    
    skip_drift = ti.xcom_pull(task_ids='fetch_annotations', key='skip_drift')
    
    if skip_drift:
        print("⏭️  No retrain trigger (insufficient data)")
        return False
    
    drift_detected = ti.xcom_pull(task_ids='calculate_drift', key='drift_detected')
    
    if drift_detected:
        print("🚨 Drift detected - WILL TRIGGER RETRAIN DAG")
        return True
    else:
        print("✅ No drift - No retrain needed")
        return False

# ============================================================================
# DAG STRUCTURE
# ============================================================================

start = EmptyOperator(
    task_id='start',
    dag=dag
)

fetch_annotations = PythonOperator(
    task_id='fetch_annotations',
    python_callable=task_fetch_annotations,
    dag=dag
)

calculate_drift = PythonOperator(
    task_id='calculate_drift',
    python_callable=task_calculate_drift,
    dag=dag
)

generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=task_generate_report,
    dag=dag
)

save_report = PythonOperator(
    task_id='save_report',
    python_callable=task_save_report,
    dag=dag
)

generate_summary = PythonOperator(
    task_id='generate_summary',
    python_callable=task_generate_summary,
    dag=dag
)

check_trigger = PythonOperator(
    task_id='check_trigger_retrain',
    python_callable=task_check_trigger_retrain,
    dag=dag
)

# Trigger retrain DAG (si drift détecté)
# Note: Le DAG model_retrain doit exister
trigger_retrain = TriggerDagRunOperator(
    task_id='trigger_model_retrain',
    trigger_dag_id='model_retrain_safe',  # DAG 3
    wait_for_completion=False,
    dag=dag
)

end = EmptyOperator(
    task_id='end',
    trigger_rule='all_done',
    dag=dag
)

# Flow
start >> fetch_annotations >> calculate_drift >> generate_report >> save_report >> generate_summary >> check_trigger

# Si drift détecté, trigger retrain
check_trigger >> trigger_retrain >> end

# Sinon, juste end
check_trigger >> end
