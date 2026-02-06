"""
DAG Model Retrain - Wakee Reloaded
Réentraîne le modèle quand drift détecté
Schedule : Manuel / Triggered by DAG 2
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys
import os
import mlflow
import mlflow.pytorch

# Ajoute le chemin utils
AIRFLOW_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if AIRFLOW_HOME not in sys.path:
    sys.path.insert(0, AIRFLOW_HOME)

from utils.data_loader import prepare_training_data, split_dataset
from utils.model_trainer import finetune_model, save_model
from utils.onnx_exporter import export_and_verify
from utils.hf_uploader import upload_model_to_hf, download_model_from_hf
from utils.database_helpers import get_db_engine
from sqlalchemy import text

# ============================================================================
# CONFIGURATION
# ============================================================================

# MLflow (commence avec local, migration HF Spaces optionnelle)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = "wakee-model-retrain"

# Training
MIN_SAMPLES = 5  # Minimum d'annotations pour retrain => preference 100 
NUM_EPOCHS = 2   # preference 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # idealement 16

# Versioning
def generate_version_name():
    """Génère un nom de version basé sur la date"""
    from datetime import datetime
    return f"v{datetime.now().strftime('%Y.%m.%d.%H%M')}"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'model_retrain',
    default_args=default_args,
    description='Réentraînement du modèle Wakee suite à drift',
    schedule=None,  # Triggered manuellement ou par DAG 2
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['retrain', 'mlops', 'pytorch'],
)

# ============================================================================
# TASK 1 : Setup MLflow
# ============================================================================

def task_setup_mlflow(**context):
    """Configure MLflow tracking"""
    print("🔧 Setting up MLflow...")
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        print(f"✅ MLflow configured:")
        print(f"   Tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"   Experiment: {MLFLOW_EXPERIMENT_NAME}")
        
        # Démarre un run MLflow
        run = mlflow.start_run(run_name=f"retrain_{generate_version_name()}")
        
        # Push run_id to XCom
        context['task_instance'].xcom_push(key='mlflow_run_id', value=run.info.run_id)
        
        print(f"✅ MLflow run started: {run.info.run_id}")
        
    except Exception as e:
        print(f"⚠️  MLflow setup failed: {e}")
        print("   Continuing without MLflow tracking...")
        context['task_instance'].xcom_push(key='mlflow_run_id', value=None)

# ============================================================================
# TASK 2 : Fetch Training Data
# ============================================================================

def task_fetch_training_data(**context):
    """Télécharge les données d'entraînement depuis R2 + NeonDB"""
    print("📥 Fetching training data...")
    
    try:
        # Prépare les données
        image_paths, labels, metadata_df = prepare_training_data(
            min_samples=MIN_SAMPLES,
            download_dir="/tmp/wakee_training_data"
        )
        
        # Log dans MLflow
        mlflow_run_id = context['task_instance'].xcom_pull(task_ids='setup_mlflow', key='mlflow_run_id')
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_param("total_samples", len(image_paths))
        
        # Push to XCom (stocke metadata_df en JSON)
        context['task_instance'].xcom_push(key='n_samples', value=len(image_paths))
        context['task_instance'].xcom_push(key='image_paths_json', value=str(image_paths))  # Simplifié
        context['task_instance'].xcom_push(key='labels_shape', value=labels.shape)
        
        print(f"✅ Training data fetched: {len(image_paths)} samples")
        
    except ValueError as e:
        print(f"❌ Not enough training data: {e}")
        raise

# ============================================================================
# TASK 3 : Split Dataset
# ============================================================================

def task_split_dataset(**context):
    """Split train/val/test"""
    print("✂️  Splitting dataset...")
    
    # Note: En production, tu récupérerais les vrais paths/labels depuis XCom
    # Ici simplifié pour la démo
    
    n_samples = context['task_instance'].xcom_pull(task_ids='fetch_training_data', key='n_samples')
    
    # Log dans MLflow
    mlflow_run_id = context['task_instance'].xcom_pull(task_ids='setup_mlflow', key='mlflow_run_id')
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_param("train_ratio", 0.7)
            mlflow.log_param("val_ratio", 0.15)
            mlflow.log_param("test_ratio", 0.15)
    
    print(f"✅ Dataset split complete")

# ============================================================================
# TASK 4 : Download Base Model
# ============================================================================

def task_download_base_model(**context):
    """Télécharge model.bin depuis HF Hub"""
    print("📥 Downloading base model from HuggingFace...")
    
    try:
        model_bin_path = download_model_from_hf(
            filename="pytorch_model.bin",
            cache_dir="/tmp/wakee_models"
        )
        
        context['task_instance'].xcom_push(key='base_model_path', value=model_bin_path)
        
        print(f"✅ Base model downloaded: {model_bin_path}")
        
    except Exception as e:
        print(f"❌ Failed to download base model: {e}")
        raise

# ============================================================================
# TASK 5 : Fine-tune Model
# ============================================================================

def task_finetune_model(**context):
    """Fine-tune le modèle PyTorch"""
    print("🔥 Fine-tuning model...")
    
    # Récupère les données depuis XCom
    base_model_path = context['task_instance'].xcom_pull(
        task_ids='download_base_model', 
        key='base_model_path'
    )
    
    # Reconstruit train/val data (simplifié ici)
    # En vrai, tu devrais passer les chemins via XCom
    from utils.data_loader import prepare_training_data, split_dataset
    
    image_paths, labels, _ = prepare_training_data(
        min_samples=MIN_SAMPLES,
        download_dir="/tmp/wakee_training_data"
    )
    
    train_data, val_data, test_data = split_dataset(image_paths, labels)
    
    # VRAI fine-tuning
    from utils.model_trainer import finetune_model, save_model
    
    model, history = finetune_model(
        model_path=base_model_path,
        train_data=train_data,
        val_data=val_data,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )
    
    # Sauvegarde le modèle fine-tuné
    finetuned_model_path = "/tmp/wakee_finetuned_model.bin"
    save_model(model, finetuned_model_path)
    
    # Log dans MLflow
    mlflow_run_id = context['task_instance'].xcom_pull(
        task_ids='setup_mlflow', 
        key='mlflow_run_id'
    )
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("num_epochs", NUM_EPOCHS)
            
            # Log metrics depuis history
            for epoch, (train_loss, val_loss) in enumerate(
                zip(history['train_loss'], history['val_loss'])
            ):
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
    
    context['task_instance'].xcom_push(
        key='finetuned_model_path', 
        value=finetuned_model_path
    )
    # context['task_instance'].xcom_push(
    #     key='pytorch_model_object',
    #     value=model  # Pour l'export ONNX
    # )
    
    print(f"✅ Model fine-tuned: {finetuned_model_path}")

# ============================================================================
# TASK 6 : Evaluate Model
# ============================================================================

def task_evaluate_model(**context):
    """Évalue le modèle sur test set"""
    print("📊 Evaluating model...")
    
    # les object pytorch ne passe pas dans les xcom
    # Récupère le modèle et les données
    # pytorch_model = context['task_instance'].xcom_pull(
    #     task_ids='finetune_model',
    #     key='pytorch_model_object'
    # )

    # Récupère le chemin
    finetuned_model_path = context['task_instance'].xcom_pull(
    task_ids='finetune_model',
    key='finetuned_model_path'
)
    
    # Recharge le modèle
    from utils.model_trainer import load_pretrained_model
    pytorch_model = load_pretrained_model(finetuned_model_path)
    
    # Reconstruit test data (ou récupère depuis XCom si tu l'as passé)
    from utils.data_loader import prepare_training_data, split_dataset
    
    image_paths, labels, _ = prepare_training_data(
        min_samples=MIN_SAMPLES,
        download_dir="/tmp/wakee_training_data"
    )
    
    _, _, test_data = split_dataset(image_paths, labels)
    
    # VRAIE évaluation
    from utils.model_trainer import evaluate_model, EmotionDataset, get_val_transform
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    
    test_images, test_labels = test_data
    test_dataset = EmotionDataset(test_images, test_labels, get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.MSELoss()
    
    test_loss, test_metrics = evaluate_model(
        model=pytorch_model,
        val_loader=test_loader,
        criterion=criterion
    )
    
    # Calcule accuracy et F1 (si tu veux)
    # Pour de la régression, ça n'a pas trop de sens mais si tu arrondis :
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    # Prédictions
    pytorch_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = pytorch_model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Arrondir pour avoir des classes (0, 1, 2, 3)
    preds_rounded = np.round(all_preds).astype(int)
    labels_rounded = all_labels.astype(int)
    
    # Calcule accuracy moyenne sur les 4 émotions
    accuracies = []
    for i in range(4):
        acc = accuracy_score(labels_rounded[:, i], preds_rounded[:, i])
        accuracies.append(acc)
    
    avg_accuracy = np.mean(accuracies)
    
    # F1 score (macro average)
    f1_scores = []
    for i in range(4):
        f1 = f1_score(
            labels_rounded[:, i], 
            preds_rounded[:, i], 
            average='macro',
            zero_division=0
        )
        f1_scores.append(f1)
    
    avg_f1 = np.mean(f1_scores)
    
    # Compile les métriques
    metrics = {
        'accuracy': avg_accuracy,
        'f1_score': avg_f1,
        'test_loss': test_loss,
        'mae_global': test_metrics['mae_global'],
        'mae_boredom': test_metrics['mae_boredom'],
        'mae_confusion': test_metrics['mae_confusion'],
        'mae_engagement': test_metrics['mae_engagement'],
        'mae_frustration': test_metrics['mae_frustration']
    }
    
    # Log dans MLflow
    mlflow_run_id = context['task_instance'].xcom_pull(
        task_ids='setup_mlflow',
        key='mlflow_run_id'
    )
    
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='metrics', value=metrics)
    
    print(f"✅ Model evaluated:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   MAE Global: {metrics['mae_global']:.4f}")
    
    return metrics

# ============================================================================
# TASK 7 : Export to ONNX
# ============================================================================

def task_export_onnx(**context):
    """Exporte le modèle vers ONNX"""
    print("📦 Exporting to ONNX...")
    
    # # Récupère le modèle PyTorch
    # pytorch_model = context['task_instance'].xcom_pull(
    #     task_ids='finetune_model',
    #     key='pytorch_model_object'
    # )

    # Récupère le chemin
    finetuned_model_path = context['task_instance'].xcom_pull(
        task_ids='finetune_model',
        key='finetuned_model_path'
    )
    
    # Recharge le modèle
    from utils.model_trainer import load_pretrained_model
    pytorch_model = load_pretrained_model(finetuned_model_path)    
    
    onnx_model_path = "/tmp/wakee_model.onnx"
    
    # VRAI export ONNX
    from utils.onnx_exporter import export_and_verify
    
    success = export_and_verify(
        pytorch_model=pytorch_model,
        onnx_path=onnx_model_path,
        verify=True,
        test_inference=True,
        compare=True
    )
    
    if not success:
        raise RuntimeError("ONNX export failed")
    
    context['task_instance'].xcom_push(
        key='onnx_model_path',
        value=onnx_model_path
    )
    
    print(f"✅ ONNX model exported: {onnx_model_path}")

# ============================================================================
# TASK 8 : Upload to HF Hub
# ============================================================================

def task_upload_to_hf(**context):
    """Upload model.bin et model.onnx vers HF Hub"""
    print("🚀 Uploading to HuggingFace Hub...")
    
    finetuned_model_path = context['task_instance'].xcom_pull(
        task_ids='finetune_model',
        key='finetuned_model_path'
    )
    onnx_model_path = context['task_instance'].xcom_pull(
        task_ids='export_onnx',
        key='onnx_model_path'
    )
    
    version_name = generate_version_name()
    
    # VRAI upload HF
    from utils.hf_uploader import upload_model_to_hf
    
    uploaded_urls = upload_model_to_hf(
        model_bin_path=finetuned_model_path,
        model_onnx_path=onnx_model_path,
        version_name=version_name,
        commit_message=f"Automated retrain {version_name}"
    )
    
    context['task_instance'].xcom_push(key='version_name', value=version_name)
    context['task_instance'].xcom_push(key='uploaded_urls', value=uploaded_urls)
    
    print(f"✅ Models uploaded for version: {version_name}")

# ============================================================================
# TASK 9 : Update Model Versions Table
# ============================================================================

def task_update_model_versions(**context):
    """Insert nouvelle version dans model_versions table"""
    print("💾 Updating model_versions table...")
    
    version_name = context['task_instance'].xcom_pull(task_ids='upload_to_hf', key='version_name')
    uploaded_urls = context['task_instance'].xcom_pull(task_ids='upload_to_hf', key='uploaded_urls')
    metrics = context['task_instance'].xcom_pull(task_ids='evaluate_model', key='metrics')
    mlflow_run_id = context['task_instance'].xcom_pull(task_ids='setup_mlflow', key='mlflow_run_id')
    n_samples = context['task_instance'].xcom_pull(task_ids='fetch_training_data', key='n_samples')
    
    engine = get_db_engine()
    
    query = text("""
        INSERT INTO model_versions (
            version_name,
            hf_model_bin_url,
            hf_model_onnx_url,
            accuracy,
            f1_score,
            mae_boredom,
            mae_confusion,
            mae_engagement,
            mae_frustration,
            n_samples_train,
            n_epochs,
            learning_rate,
            batch_size,
            mlflow_run_id,
            is_production
        ) VALUES (
            :version_name,
            :hf_model_bin_url,
            :hf_model_onnx_url,
            :accuracy,
            :f1_score,
            :mae_boredom,
            :mae_confusion,
            :mae_engagement,
            :mae_frustration,
            :n_samples_train,
            :n_epochs,
            :learning_rate,
            :batch_size,
            :mlflow_run_id,
            FALSE
        )
        RETURNING id
    """)
    
    params = {
        'version_name': version_name,
        'hf_model_bin_url': uploaded_urls.get('model_bin_url'),
        'hf_model_onnx_url': uploaded_urls.get('model_onnx_url'),
        'accuracy': float(metrics.get('accuracy')),
        'f1_score': float(metrics.get('f1_score')),
        'mae_boredom': float(metrics.get('mae_boredom')),
        'mae_confusion': float(metrics.get('mae_confusion')),
        'mae_engagement': float(metrics.get('mae_engagement')),
        'mae_frustration': float(metrics.get('mae_frustration')),
        'n_samples_train': int(n_samples * 0.7) if n_samples else 0,
        'n_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'mlflow_run_id': mlflow_run_id
    }
    
    with engine.connect() as conn:
        result = conn.execute(query, params)
        conn.commit()
        model_version_id = result.fetchone()[0]
    
    print(f"✅ Model version saved (ID: {model_version_id})")
    
    context['task_instance'].xcom_push(key='model_version_id', value=model_version_id)

# ============================================================================
# TASK 10 : Finalize MLflow
# ============================================================================

def task_finalize_mlflow(**context):
    """Finalise le run MLflow"""
    mlflow_run_id = context['task_instance'].xcom_pull(task_ids='setup_mlflow', key='mlflow_run_id')
    
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.set_tag("status", "completed")
            mlflow.end_run()
        print(f"✅ MLflow run finalized: {mlflow_run_id}")
    else:
        print("⏭️  No MLflow run to finalize")

# ============================================================================
# TASK 11 : Summary
# ============================================================================

def task_generate_summary(**context):
    """Génère un résumé du réentraînement"""
    version_name = context['task_instance'].xcom_pull(task_ids='upload_to_hf', key='version_name')
    metrics = context['task_instance'].xcom_pull(task_ids='evaluate_model', key='metrics')
    n_samples = context['task_instance'].xcom_pull(task_ids='fetch_training_data', key='n_samples')
    
    print("\n" + "="*70)
    print("🎉 MODEL RETRAIN SUMMARY")
    print("="*70)
    print(f"📦 New version: {version_name}")
    print(f"📊 Training samples: {n_samples}")
    print(f"")
    print(f"Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("="*70 + "\n")

# ============================================================================
# DAG STRUCTURE
# ============================================================================

start = EmptyOperator(task_id='start', dag=dag)

setup_mlflow = PythonOperator(
    task_id='setup_mlflow',
    python_callable=task_setup_mlflow,
    dag=dag
)

fetch_data = PythonOperator(
    task_id='fetch_training_data',
    python_callable=task_fetch_training_data,
    dag=dag
)

split_data = PythonOperator(
    task_id='split_dataset',
    python_callable=task_split_dataset,
    dag=dag
)

download_model = PythonOperator(
    task_id='download_base_model',
    python_callable=task_download_base_model,
    dag=dag
)

finetune = PythonOperator(
    task_id='finetune_model',
    python_callable=task_finetune_model,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=task_evaluate_model,
    dag=dag
)

export_onnx = PythonOperator(
    task_id='export_onnx',
    python_callable=task_export_onnx,
    dag=dag
)

upload_hf = PythonOperator(
    task_id='upload_to_hf',
    python_callable=task_upload_to_hf,
    dag=dag
)

update_db = PythonOperator(
    task_id='update_model_versions',
    python_callable=task_update_model_versions,
    dag=dag
)

finalize_mlflow = PythonOperator(
    task_id='finalize_mlflow',
    python_callable=task_finalize_mlflow,
    dag=dag
)

summary = PythonOperator(
    task_id='generate_summary',
    python_callable=task_generate_summary,
    dag=dag
)

end = EmptyOperator(task_id='end', trigger_rule='all_done', dag=dag)

# Flow
start >> setup_mlflow >> fetch_data >> split_data >> download_model >> finetune >> evaluate >> export_onnx >> upload_hf >> update_db >> finalize_mlflow >> summary >> end
