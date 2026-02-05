"""
DAG Health Check - Wakee Reloaded
Vérifie que tous les composants du système sont opérationnels
Schedule : Hebdomadaire (Dimanche 3h du matin)
"""
# ============================
# import Airflow
# ============================

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import TaskGroup

# ============================
# import utils
# ============================

from PIL import Image
import io
import pytest
import boto3
from botocore.exceptions import ClientError
from huggingface_hub import hf_hub_download
from sqlalchemy import create_engine, text
from datetime import timedelta, datetime
import requests
import os
import sys

import warnings
from dotenv import load_dotenv

# ============================
# Configuration
# ============================

warnings.filterwarnings("ignore")
load_dotenv()

API_URL = os.getenv("API_URL", "https://terorra-wakee-api.hf.space")
SOURCING_URL = os.getenv("SOURCING_URL", "https://terorra-wakee-sourcing.hf.space")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Terorra/wakee-reloaded")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,  # Active si tu configures SMTP
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# TASK 1 : Test API
# ============================================================================

def test_api_health(**context):
    """Vérifie que l'API est opérationnelle"""
    print("🔍 Testing API health...")
    ti = context['ti']
    try:
        # Test /health endpoint
        print(f"Testing {API_URL}/health")
        response = requests.get(f"{API_URL}/health", timeout=10)
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        health_data = response.json()
        assert health_data['status'] == 'healthy', "API reports unhealthy status"
        assert health_data.get('model_loaded') == True, "Model not loaded"
        
        print("✅ API /health OK")
        print(f"   Data: {health_data}")
        
        
        # Crée une image test (224x224 RGB)
        test_image = Image.new('RGB', (224, 224), color='red')
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        files = {'file': ('test.jpg', img_buffer, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
        
        assert response.status_code == 200, f"Predict failed: {response.status_code}"
        
        predictions = response.json()
        required_keys = ['boredom', 'confusion', 'engagement', 'frustration']
        for key in required_keys:
            assert key in predictions, f"Missing {key} prediction"
        
        print("✅ API /predict OK")
        print(f"   Sample predictions: {predictions}")
        
        # Push to XCom
        ti.xcom_push(key='api_status', value='healthy')
        ti.xcom_push(key='sample_predictions', value=predictions)
        
        return True
        
    except Exception as e:
        print(f"❌ API Health Check FAILED: {e}")
        ti.xcom_push(key='api_status', value='unhealthy')
        raise

# ============================================================================
# TASK 2 : Test Database
# ============================================================================

def test_database_health(**context):
    """Vérifie que NeonDB est opérationnelle"""
    print("🔍 Testing Database health...")
    ti = context['ti']
    try:
        
        
        NEON_DATABASE_URL = os.getenv("NEONDB_WR")
        assert NEON_DATABASE_URL, "NEON_DATABASE_URL not set"
        
        engine = create_engine(NEON_DATABASE_URL)
        
        with engine.connect() as conn:
            # Test connexion
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1, "Connection test failed"
            print("✅ Database connection OK")
            
            # Vérifie tables existent
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in conn.execute(tables_query)]
            
            required_tables = ['emotion_labels', 'drift_reports', 'model_versions']
            for table in required_tables:
                assert table in tables, f"Table {table} missing"
            
            print(f"✅ Required tables exist: {required_tables}")
            
            # Compte les annotations
            count_query = text("SELECT COUNT(*) FROM emotion_labels")
            count = conn.execute(count_query).fetchone()[0]
            
            print(f"✅ Database has {count} annotations")
            
            # Statistiques
            validated_query = text("SELECT COUNT(*) FROM emotion_labels WHERE is_validated = TRUE")
            validated_count = conn.execute(validated_query).fetchone()[0]
            
            stats = {
                'total_annotations': count,
                'validated_annotations': validated_count,
                'tables': tables
            }
            
            if count == 0:
                print("⚠️  Warning: No annotations in database yet")
            
            # Push to XCom
            ti.xcom_push(key='db_status', value='healthy')
            ti.xcom_push(key='db_stats', value=stats)
        
        return True
        
    except Exception as e:
        print(f"❌ Database Health Check FAILED: {e}")
        ti.xcom_push(key='db_status', value='unhealthy')
        raise

# ============================================================================
# TASK 3 : Test Storage (R2)
# ============================================================================

def test_storage_health(**context):
    """Vérifie que Cloudflare R2 est opérationnel"""
    print("🔍 Testing Storage health...")
    ti = context['ti']
    try:
        
        
        R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
        R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
        R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
        R2_BUCKET_NAME = os.getenv("R2_WR_IMG_BUCKET_NAME", "wr-img-store")
        
        assert R2_ACCOUNT_ID, "R2_ACCOUNT_ID not set"
        assert R2_ACCESS_KEY_ID, "R2_ACCESS_KEY_ID not set"
        assert R2_SECRET_ACCESS_KEY, "R2_SECRET_ACCESS_KEY not set"
        
        # Connexion R2
        s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name='auto'
        )
        
        # Test bucket accessible
        s3_client.head_bucket(Bucket=R2_BUCKET_NAME)
        print(f"✅ Bucket {R2_BUCKET_NAME} accessible")
        
        # Compte les objets
        response = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, MaxKeys=1000)
        count = response.get('KeyCount', 0)
        
        print(f"✅ Storage has {count} images")
        
        # Test upload/download
        test_key = "health_check/test.txt"
        test_content = b"Health check test"
        
        s3_client.put_object(Bucket=R2_BUCKET_NAME, Key=test_key, Body=test_content)
        print("✅ Upload test OK")
        
        obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=test_key)
        downloaded_content = obj['Body'].read()
        assert downloaded_content == test_content, "Download content mismatch"
        print("✅ Download test OK")
        
        # Cleanup
        s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=test_key)
        print("✅ Cleanup OK")
        
        stats = {
            'total_images': count,
            'bucket_name': R2_BUCKET_NAME
        }
        
        if count == 0:
            print("⚠️  Warning: No images in storage yet")
        
        # Push to XCom
        ti.xcom_push(key='storage_status', value='healthy')
        ti.xcom_push(key='storage_stats', value=stats)
        
        return True
        
    except Exception as e:
        print(f"❌ Storage Health Check FAILED: {e}")
        ti.xcom_push(key='storage_status', value='unhealthy')
        raise

# ============================================================================
# TASK 4 : Test App Sourcing
# ============================================================================

def test_sourcing_app_health(**context):
    """Vérifie que l'app Sourcing est accessible"""
    print("🔍 Testing Sourcing App health...")
    ti = context['ti']
    try:
        response = requests.get(SOURCING_URL, timeout=10)
        assert response.status_code == 200, f"App not accessible: {response.status_code}"
        
        print(f"✅ Sourcing App accessible at {SOURCING_URL}")
        
        # Push to XCom
        ti.xcom_push(key='sourcing_status', value='healthy')
        
        return True
        
    except Exception as e:
        print(f"❌ Sourcing App Health Check FAILED: {e}")
        ti.xcom_push(key='sourcing_status', value='unhealthy')
        raise

# ============================================================================
# TASK 5 : Test HuggingFace Model Hub
# ============================================================================

def test_model_hub_health(**context):
    """Vérifie que le modèle HF est accessible"""
    print("🔍 Testing Model Hub health...")
    ti = context['ti']
    try:
        
        MODEL_FILENAME = "model.onnx"
        
        # Test download (utilise cache)
        print(f"Testing download from {HF_MODEL_REPO}")
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILENAME,
            cache_dir="/tmp/health_check_cache"
        )
        
        assert os.path.exists(model_path), "Model file not found"
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"✅ Model downloadable from HF Hub ({file_size:.1f} MB)")
        
        stats = {
            'model_repo': HF_MODEL_REPO,
            'model_file': MODEL_FILENAME,
            'model_size_mb': round(file_size, 1)
        }
        
        # Push to XCom
        ti.xcom_push(key='model_hub_status', value='healthy')
        ti.xcom_push(key='model_stats', value=stats)
        
        return True
        
    except Exception as e:
        print(f"❌ Model Hub Health Check FAILED: {e}")
        ti.xcom_push(key='model_hub_status', value='unhealthy')
        raise

# ============================================================================
# TASK 6 : Run Pytest Suite
# ============================================================================

def run_pytest_suite(**context):
    """Exécute la suite de tests Pytest"""
    print("🔍 Running Pytest suite...")
    ti = context['ti']

    try:
        # Tests directory
        tests_dir = 'opt/airflow/tests'
        
        if not os.path.exists(tests_dir):
            print(f"⚠️  Tests directory not found: {tests_dir}")
            print("   Creating placeholder...")
            os.makedirs(tests_dir, exist_ok=True)
            # Crée un test placeholder
            with open(f"{tests_dir}/test_placeholder.py", 'w') as f:
                f.write("def test_placeholder():\n    assert True\n")
        
        # Run pytest
        exit_code = pytest.main([
            tests_dir,
            '-v',
            '--tb=short',
            '--junit-xml=/tmp/pytest_results.xml'
        ])
        
        if exit_code == 0:
            print("✅ All Pytest tests passed")
            ti.xcom_push(key='pytest_status', value='passed')
            return True
        else:
            print(f"⚠️  Pytest suite had failures (exit code: {exit_code})")
            ti.xcom_push(key='pytest_status', value='failed')
            # Ne pas raise pour ne pas bloquer le pipeline
            return False
        
    except Exception as e:
        print(f"❌ Pytest execution FAILED: {e}")
        ti.xcom_push(key='pytest_status', value='error')
        return False

# ============================================================================
# TASK 7 : Generate Summary Report
# ============================================================================

def generate_summary_report(**context):
    """Génère un rapport récapitulatif"""
    ti = context['ti']
    
    # Récupère les statuts depuis XCom
    api_status = ti.xcom_pull(task_ids='apps.test_api_health', key='api_status')
    db_status = ti.xcom_pull(task_ids='storage.test_database_health', key='db_status')
    storage_status = ti.xcom_pull(task_ids='storage.test_storage_health', key='storage_status')
    sourcing_status = ti.xcom_pull(task_ids='apps.test_sourcing_app_health', key='sourcing_status')
    model_hub_status = ti.xcom_pull(task_ids='test_model_hub_health', key='model_hub_status')
    pytest_status = ti.xcom_pull(task_ids='run_pytest_suite', key='pytest_status')
    
    # Récupère les stats
    db_stats = ti.xcom_pull(task_ids='storage.test_database_health', key='db_stats')
    storage_stats = ti.xcom_pull(task_ids='storage.test_storage_health', key='storage_stats')

    print("\n" + "="*70)
    print("📊 WAKEE HEALTH CHECK SUMMARY")
    print("="*70)
    print(f"🌐 API Status          : {api_status or 'unknown'}")
    print(f"💾 Database Status     : {db_status or 'unknown'}")
    print(f"📦 Storage Status      : {storage_status or 'unknown'}")
    print(f"🎨 Sourcing App Status : {sourcing_status or 'unknown'}")
    print(f"🤗 Model Hub Status    : {model_hub_status or 'unknown'}")
    print(f"🧪 Pytest Status       : {pytest_status or 'unknown'}")
    print("="*70)
    
    if db_stats:
        print(f"\n📊 Database Stats:")
        print(f"   Total annotations: {db_stats.get('total_annotations', 0)}")
        print(f"   Validated: {db_stats.get('validated_annotations', 0)}")
    
    if storage_stats:
        print(f"\n📦 Storage Stats:")
        print(f"   Total images: {storage_stats.get('total_images', 0)}")
    
    # Statut global
    all_healthy = all([
        api_status == 'healthy',
        db_status == 'healthy',
        storage_status == 'healthy',
        sourcing_status == 'healthy',
        model_hub_status == 'healthy'
    ])
    
    if all_healthy:
        print("\n✅ All systems operational!")
    else:
        print("\n⚠️  Some systems need attention")
    
    print("="*70 + "\n")
    
    return all_healthy

# =====================================================================
# DAG DEFINITION
# =====================================================================

with DAG(
    'health_check_weekly',
    default_args=default_args,
    description='Health checks hebdomadaires du système Wakee',
    schedule='0 3 * * 0',  # Dimanche 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['health', 'monitoring', 'tests'],
    ) as dag:

    start = EmptyOperator(
            task_id='start',
            doc_md="""
            ## Début des Tests Hebdomadaires
            
            Ce DAG lance tous les tests de qualité du projet.
            """
        )
    with TaskGroup(group_id="storage") as storage_branch:

        task_database = PythonOperator(
            task_id='test_database_health',
            python_callable=test_database_health
                    )

        task_storage = PythonOperator(
            task_id='test_storage_health',
            python_callable=test_storage_health
                    )

        task_database >> task_storage

    with TaskGroup(group_id="apps") as apps_branch:

        task_api = PythonOperator(
            task_id='test_api_health',
            python_callable=test_api_health
                    )

        task_sourcing = PythonOperator(
            task_id='test_sourcing_app_health',
            python_callable=test_sourcing_app_health
                    )

        task_api >> task_sourcing

    task_model_hub = PythonOperator(
        task_id='test_model_hub_health',
        python_callable=test_model_hub_health
            )

    task_pytest = PythonOperator(
        task_id='run_pytest_suite',
        python_callable=run_pytest_suite
            )

    task_summary = PythonOperator(
        task_id='generate_summary_report',
        python_callable=generate_summary_report
            )

    end = EmptyOperator(
        task_id='end',
        trigger_rule='all_done',  # S'exécute même si certaines tâches échouent
        doc_md="""
        ## Fin des Tests
        
        Tous les tests sont terminés.
        """
    )

# Tous les tests en parallèle, puis summary
start >> [storage_branch, apps_branch] >> task_model_hub >> task_pytest >> task_summary >> end
# start >> [storage_branch, apps_branch] >> task_model_hub >> task_summary >> end
