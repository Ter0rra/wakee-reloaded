"""
DAG Model Retrain - VERSION SAFE avec Validation
Utilise les scripts utils existants: data_loader, database_helpers, hf_uploader
Architecture: Style Terorra
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup

import mlflow
import shutil
import numpy as np
import requests

from datetime import datetime, timedelta
import os
import sys

# Ajoute le chemin utils
AIRFLOW_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AIRFLOW_HOME not in sys.path:
    sys.path.insert(0, AIRFLOW_HOME)

from utils.onnx_converter import pytorch_to_onnx, onnx_to_pytorch
from utils.data_loader import split_dataset, prepare_training_data
from utils.model_validator import compare_models
from utils.model_trainer import finetune_model, save_model, load_pretrained_model
from utils.database_helpers import update_retrain_triggered, save_drift_report
from utils.hf_uploader import download_model_from_hf, upload_model_to_hf


# ============================================================================
# CONFIGURATION
# ============================================================================

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "wakee-model-retrain"

R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_URI")

# Training
MIN_SAMPLES = 5  # Minimum d'annotations pour retrain => preference 100 
NUM_EPOCHS = 5   # preference 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 4 # idealement 16

# Versioning
def generate_version_name():
    """Génère un nom de version basé sur la date"""
    from datetime import datetime
    return f"v{datetime.now().strftime('%Y.%m.%d.%H%M')}"

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Terorra/wakee-reloaded")
NEON_DATABASE_URL = os.getenv("NEONDB_WR")

default_args = {
    'owner': 'terorra',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

# ============================================================================
# TASK FUNCTIONS
# ============================================================================

# ============================================================================
# TASK 1 : Setup MLflow + R2
# ============================================================================

def task_setup_mlflow(**context):
    """Initialize MLflow tracking"""
    
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        print(f"✅ MLflow URI: {mlflow_tracking_uri}")
    
    mlflow.set_experiment("wakee_safe_retraining")
    
    run = mlflow.start_run(run_name=f"retrain_safe_{datetime.now().strftime('%Y.%m.%d.%H%M')}")
    run_id = run.info.run_id
    
    mlflow.end_run()
    
    context['task_instance'].xcom_push(key='mlflow_run_id', value=run_id)
    print(f"✅ MLflow run: {run_id}")

# ============================================================================
# TASK 2 : Download Base Model
# ============================================================================

def task_download_baseline_onnx(**context):
    """Télécharge le modèle ONNX de production (baseline)"""
    
    print("📥 Downloading baseline ONNX model...")
    
    # ✅ Utilise hf_uploader.py existant
    baseline_onnx_path = download_model_from_hf(
        filename="model.onnx",
        cache_dir="/tmp/models"
    )
    
    # Copie vers /tmp pour usage
    baseline_copy = "/tmp/baseline_model.onnx"
    shutil.copy(baseline_onnx_path, baseline_copy)
    
    file_size = os.path.getsize(baseline_copy) / 1e6
    print(f"✅ Baseline ONNX: {baseline_copy} ({file_size:.2f} MB)")
    
    context['task_instance'].xcom_push(key='baseline_onnx_path', value=baseline_copy)

# ============================================================================
# TASK 3 : Export to ONNX
# ============================================================================

def task_convert_onnx_to_pytorch(**context):
    """Convertit ONNX baseline → PyTorch .bin pour fine-tuning"""
    
    baseline_onnx_path = context['task_instance'].xcom_pull(
        task_ids='download_baseline_onnx',
        key='baseline_onnx_path'
    )
    
    output_bin_path = "/tmp/baseline_model.bin"
    
    success = onnx_to_pytorch(
        onnx_path=baseline_onnx_path,
        output_bin_path=output_bin_path,
        verify=True
    )
    
    if not success:
        raise RuntimeError("ONNX → PyTorch conversion failed")
    
    context['task_instance'].xcom_push(key='base_model_path', value=output_bin_path)
    print(f"✅ PyTorch baseline: {output_bin_path}")

# ============================================================================
# TASK 4 : Fetch Training Data
# ============================================================================

def task_fetch_training_data(**context):
    """Récupère données validées depuis NeonDB et R2"""
    
    print("📊 Fetching training data...")
    
    # ✅ Utilise data_loader.py existant
    image_paths, labels, metadata_df = prepare_training_data(
        min_samples=5,  # Minimum requis
        download_dir="/tmp/wakee_training_data"
    )
    
    print(f"✅ Prepared {len(image_paths)} samples")
    
    context['task_instance'].xcom_push(key='image_paths', value=image_paths)
    context['task_instance'].xcom_push(key='labels', value=labels.tolist())
    context['task_instance'].xcom_push(key='num_samples', value=len(image_paths))

# ============================================================================
# TASK 5 : Split Dataset & Fine-tune Model
# ============================================================================

def task_finetune_model(**context):
    """Fine-tune avec freeze backbone"""

    print("🔥 Fine-tuning model...")
    
    base_model_path = context['task_instance'].xcom_pull(
        task_ids='convert_onnx_to_pytorch',
        key='base_model_path'
    )
    
    image_paths = context['task_instance'].xcom_pull(
        task_ids='fetch_training_data',
        key='image_paths'
    )
    
    labels = np.array(context['task_instance'].xcom_pull(
        task_ids='fetch_training_data',
        key='labels'
    ))
    
    # ✅ Utilise split_dataset de data_loader.py
    (train_images, train_labels), (val_images, val_labels), _ = split_dataset(
        image_paths=image_paths,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    # Fine-tune (freeze backbone)
    model, history = finetune_model(
        model_path=base_model_path,
        train_data=(train_images, train_labels),
        val_data=(val_images, val_labels),
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=16,
        freeze_backbone=True
    )
    
    # Save
    finetuned_path = "/tmp/wakee_finetuned_model.bin"
    save_model(model, finetuned_path)
    
    # Log to MLflow
    mlflow_run_id = context['task_instance'].xcom_pull(
        task_ids='setup_mlflow',
        key='mlflow_run_id'
    )
    
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metrics({
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'best_mae': min([m['mae_global'] for m in history['val_metrics']])
            })
            mlflow.log_artifact(finetuned_path, artifact_path="models")
    
    context['task_instance'].xcom_push(key='finetuned_model_path', value=finetuned_path)
    context['task_instance'].xcom_push(key='training_history', value=history)
    
    print(f"✅ Model fine-tuned and saved")

# ============================================================================
# TASK 6 : Evaluate & Validate Model
# ============================================================================

def task_validate_model(**context):
    """Compare nouveau modèle vs baseline ONNX"""

    print("\n" + "="*70)
    print("🔬 MODEL VALIDATION")
    print("="*70)
    
    baseline_onnx_path = context['task_instance'].xcom_pull(
        task_ids='download_baseline_onnx',
        key='baseline_onnx_path'
    )
    
    finetuned_model_path = context['task_instance'].xcom_pull(
        task_ids='finetune_model',
        key='finetuned_model_path'
    )
    
    # Load new model
    new_model = load_pretrained_model(finetuned_model_path)
    
    # Compare
    result = compare_models(
        baseline_onnx_path=baseline_onnx_path,
        new_model=new_model,
        num_random_tests=50
    )
    
    recommendation = result['recommendation']

    # ✅ Sauvegarde le rapport de validation dans NeonDB
    num_samples = context['task_instance'].xcom_pull(
        task_ids='fetch_training_data',
        key='num_samples'
    )
    
    drift_detected = recommendation == "REJECT"  # Si reject = drift trop important
    
    report_id = save_drift_report(
        report_date=datetime.now(),
        drift_detected=drift_detected,
        drift_score=result['metrics']['new_variance'],
        metrics={
            'mae_boredom': result['metrics']['new_mean'][0],
            'mae_confusion': result['metrics']['new_mean'][1],
            'mae_engagement': result['metrics']['new_mean'][2],
            'mae_frustration': result['metrics']['new_mean'][3],
        },
        num_samples=num_samples,
        retrain_triggered=False,  # Sera mis à jour si déploiement
        report_url=None
    )
    
    print(f"✅ Validation report saved (ID: {report_id})")
    
    # Log to MLflow
    mlflow_run_id = context['task_instance'].xcom_pull(
        task_ids='setup_mlflow',
        key='mlflow_run_id'
    )
    
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_params({
                'validation_recommendation': recommendation,
                **{f'criteria_{k}': v for k, v in result['criteria'].items()}
            })
            
            # ✅ FIX: Log seulement les SCALARS (pas les listes)
            scalar_metrics = {
                k: v for k, v in result['metrics'].items()
                if isinstance(v, (int, float, bool)) and not isinstance(v, list)
            }
            
            mlflow.log_metrics(scalar_metrics)
            
            # Log les arrays comme params (stringifiés)
            array_metrics = {
                k: str(v) for k, v in result['metrics'].items()
                if isinstance(v, list)
            }
            
            if array_metrics:
                mlflow.log_params(array_metrics)
    
    context['task_instance'].xcom_push(key='validation_result', value=result)
    context['task_instance'].xcom_push(key='recommendation', value=recommendation)
    context['task_instance'].xcom_push(key='report_id', value=report_id)
    
    print(f"\n✅ Validation complete: {recommendation}")

def task_decide_deployment(**context):
    """Décide si on déploie le nouveau modèle ou garde le baseline"""
    recommendation = context['task_instance'].xcom_pull(
        task_ids='validate_model',
        key='recommendation'
    )
    
    if recommendation == "APPROVE":
        print("✅ Deploying new model")
        return 'export_new_onnx'
    else:
        print("❌ Keeping baseline model")
        return 'keep_baseline'

# ============================================================================
# TASK 7 : Export to ONNX
# ============================================================================

def task_export_new_onnx(**context):
    """Exporte nouveau modèle en ONNX"""
    
    print("🔄 Exporting new model to ONNX...")
    
    finetuned_model_path = context['task_instance'].xcom_pull(
        task_ids='finetune_model',
        key='finetuned_model_path'
    )
    
    model = load_pretrained_model(finetuned_model_path)
    
    new_onnx_path = "/tmp/wakee_model_new.onnx"
    
    success = pytorch_to_onnx(
        pytorch_model=model,
        output_onnx_path=new_onnx_path,
        opset_version=11
    )
    
    if not success:
        raise RuntimeError("ONNX export failed")
    
    # ✅ Marque le rapport comme "retrain triggered"
    report_id = context['task_instance'].xcom_pull(
        task_ids='validate_model',
        key='report_id'
    )
    
    if report_id:
        update_retrain_triggered(report_id)
    
    context['task_instance'].xcom_push(key='new_onnx_path', value=new_onnx_path)
    
    print(f"✅ New ONNX exported: {new_onnx_path}")

# ============================================================================
# TASK 8 : keep baseline
# ============================================================================

def task_keep_baseline(**context):
    """Garde le modèle baseline"""
    print("📌 Keeping baseline ONNX model")
    
    baseline_onnx_path = context['task_instance'].xcom_pull(
        task_ids='download_baseline_onnx',
        key='baseline_onnx_path'
    )
    
    context['task_instance'].xcom_push(key='final_onnx_path', value=baseline_onnx_path)
    
    print(f"✅ Final model: {baseline_onnx_path} (baseline)")

# ============================================================================
# TASK 9 : Upload to HF Hub
# ============================================================================

# def task_upload_to_hf(**context):
#     """Upload modèle final sur HF Hub"""
    
#     recommendation = context['task_instance'].xcom_pull(
#         task_ids='validate_model',
#         key='recommendation'
#     )
    
#     if recommendation == "APPROVE":
#         # Nouveau modèle
#         onnx_path = context['task_instance'].xcom_pull(
#             task_ids='export_new_onnx',
#             key='new_onnx_path'
#         )
        
#         bin_path = context['task_instance'].xcom_pull(
#             task_ids='finetune_model',
#             key='finetuned_model_path'
#         )
        
#         version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M')}"
#         commit_message = f"✅ Deploy new model (validated {datetime.now().strftime('%Y-%m-%d %H:%M')})"
        
#     else:
#         # Baseline (on re-upload pour confirmation)
#         onnx_path = context['task_instance'].xcom_pull(
#             task_ids='keep_baseline',
#             key='final_onnx_path'
#         )
        
#         # Download le .bin baseline aussi
#         bin_path = download_model_from_hf(
#             filename="pytorch_model.bin",
#             cache_dir="/tmp/models"
#         )
        
#         version_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M')}"
#         commit_message = f"📌 Keep baseline (new model rejected {datetime.now().strftime('%Y-%m-%d %H:%M')})"
    
#     print(f"🚀 Uploading to HF Hub...")
#     print(f"   Status: {recommendation}")
    
#     # ✅ Utilise hf_uploader.py existant
#     uploaded_files = upload_model_to_hf(
#         model_bin_path=bin_path,
#         model_onnx_path=onnx_path,
#         version_name=version_name,
#         commit_message=commit_message
#     )
    
#     print(f"✅ Uploaded to {HF_MODEL_REPO}")
    
#     context['task_instance'].xcom_push(key='uploaded_files', value=uploaded_files)

def task_upload_to_hf(**context):
    """Upload modèle final sur HF Hub"""
    from utils.hf_uploader import upload_model_to_hf
    
    recommendation = context['task_instance'].xcom_pull(
        task_ids='validate_model',
        key='recommendation'
    )
    
    if recommendation == "APPROVE":
        # Nouveau modèle
        onnx_path = context['task_instance'].xcom_pull(
            task_ids='export_new_onnx',
            key='new_onnx_path'
        )
        
        bin_path = context['task_instance'].xcom_pull(
            task_ids='finetune_model',
            key='finetuned_model_path'
        )
        
        # ✅ Vérifie que les paths existent
        if not onnx_path or not bin_path:
            raise ValueError("Missing model paths for APPROVE deployment")
        
        version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M')}"
        commit_message = f"✅ Deploy new model (validated {datetime.now().strftime('%Y-%m-%d %H:%M')})"
        
    else:
        # Baseline (on re-upload pour confirmation)
        onnx_path = context['task_instance'].xcom_pull(
            task_ids='keep_baseline',
            key='final_onnx_path'
        )
        
        # ✅ Vérifie que le path existe
        if not onnx_path:
            # Fallback: re-download depuis HF
            print("⚠️  No baseline path in XCom, re-downloading from HF...")
            from utils.hf_uploader import download_model_from_hf
            onnx_path = download_model_from_hf(
                filename="model.onnx",
                cache_dir="/tmp/models"
            )
        
        # Download le .bin baseline aussi
        from utils.hf_uploader import download_model_from_hf
        bin_path = download_model_from_hf(
            filename="pytorch_model.bin",
            cache_dir="/tmp/models"
        )
        
        version_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M')}"
        commit_message = f"📌 Keep baseline (new model rejected {datetime.now().strftime('%Y-%m-%d %H:%M')})"
    
    print(f"🚀 Uploading to HF Hub...")
    print(f"   Status: {recommendation}")
    print(f"   ONNX: {onnx_path}")
    print(f"   BIN: {bin_path}")
    
    # ✅ Utilise hf_uploader.py existant
    uploaded_files = upload_model_to_hf(
        model_bin_path=bin_path,
        model_onnx_path=onnx_path,
        version_name=version_name,
        commit_message=commit_message
    )
    
    print(f"✅ Uploaded to {HF_MODEL_REPO}")
    
    context['task_instance'].xcom_push(key='uploaded_files', value=uploaded_files)

# ============================================================================
# TASK 9.2 : trigger github action
# ============================================================================

def trigger_deploy_api(**context):
    """
    Trigger a GitHub Action via workflow_dispatch
    """

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not set")

    url = (
        f"https://api.github.com/repos/"
        "Ter0rra/wakee-reloaded/"
        "actions/workflows/deploy-api.yml/dispatches"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    payload = {
        "ref": "main"
    }

    r = requests.post(url, json=payload, headers=headers)

    if r.status_code != 204:
        raise RuntimeError(
            f"GitHub Action trigger failed: {r.status_code} - {r.text}"
        )

    print("✅ deploy-api.yml déclenché")

# ============================================================================
# TASK 10 : clean up
# ============================================================================

def task_cleanup(**context):
    """Nettoie fichiers temporaires"""
    
    print("🧹 Cleaning up...")
    
    dirs_to_clean = [
        "/tmp/wakee_training_data",
        "/tmp/models"
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"   Removed: {dir_path}")
    
    print("✅ Cleanup complete")

# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    'model_retrain_safe',
    default_args=default_args,
    description='MLOps pipeline avec validation et rollback',
    schedule=None,
    catchup=False,
    tags=['mlops', 'retraining', 'safe'],
) as dag:
    
    start = EmptyOperator(task_id='start')

    setup_mlflow_task = PythonOperator(
        task_id='setup_mlflow',
        python_callable=task_setup_mlflow
    )

    download_baseline_task = PythonOperator(
        task_id='download_baseline_onnx',
        python_callable=task_download_baseline_onnx
    )
    
    convert_onnx_task = PythonOperator(
        task_id='convert_onnx_to_pytorch',
        python_callable=task_convert_onnx_to_pytorch
    )

    fetch_data_task = PythonOperator(
        task_id='fetch_training_data',
        python_callable=task_fetch_training_data
    )

    finetune_task = PythonOperator(
        task_id='finetune_model',
        python_callable=task_finetune_model
    )

    validate_task = PythonOperator(
        task_id='validate_model',
        python_callable=task_validate_model
    )

    decide_task = BranchPythonOperator(
        task_id='decide_deployment',
        python_callable=task_decide_deployment
    )

    with TaskGroup(group_id="new_model") as new_model_deploy:
        export_new_task = PythonOperator(
                task_id ='export_new_onnx',
                python_callable=task_export_new_onnx
                    )

        trigger_ci = PythonOperator(
                task_id="trigger_deploy_api",
                python_callable=trigger_deploy_api
                    )

        export_new_task >> trigger_ci

    keep_baseline_task = PythonOperator(
        task_id='keep_baseline',
        python_callable=task_keep_baseline
            )

    join_task = EmptyOperator(
        task_id='join_branches',
        trigger_rule='none_failed_min_one_success'
            )

    upload_task = PythonOperator(
        task_id='upload_to_hf',
        python_callable=task_upload_to_hf,
        trigger_rule='none_failed_min_one_success'
    )


    trigger_ci2 = PythonOperator(
            task_id="trigger_deploy_api_2",
            python_callable=trigger_deploy_api
                )

    cleanup_task = PythonOperator(
        task_id='cleanup',
        python_callable=task_cleanup,
        trigger_rule='all_done'
    )

    end = EmptyOperator(task_id='end', trigger_rule='all_done')

    # ========================================================================
    # DAG FLOW
    # ========================================================================
    
    start >> setup_mlflow_task >> download_baseline_task >> convert_onnx_task >> fetch_data_task >> finetune_task >> validate_task >> decide_task
    
    decide_task >> [new_model_deploy, keep_baseline_task] 

    [new_model_deploy, keep_baseline_task] >> join_task >> upload_task >> trigger_ci2 >> cleanup_task >> end