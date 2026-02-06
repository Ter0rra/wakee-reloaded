"""
MLflow Tracking Server - Wakee Reloaded
Deployed on HuggingFace Spaces
"""

import os
import subprocess
import sys
import dotenv


# Configuration

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_env_vars():
    """Charge .env en local, utilise env vars en prod"""
    is_production = os.getenv("SPACE_ID") is not None
    
    if not is_production:
        from pathlib import Path
        try:
            from dotenv import load_dotenv
            root_dir = Path(__file__).resolve().parent.parent
            dotenv_path = root_dir / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
                print(f"✅ .env chargé depuis : {dotenv_path}")
        except ImportError:
            print("⚠️  python-dotenv non installé (OK en production)")

load_env_vars()

MLFLOW_BACKEND_STORE_URI = os.getenv("NEONDB_MLFLOW")
MLFLOW_ARTIFACT_ROOT = os.getenv("R2_WR_MLFLOW_URI")

# ✅ Configuration S3 pour R2 (CRITIQUE)
AWS_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_URI")

# HF Spaces port
PORT = int(os.getenv("PORT", 7860))

def main():
    """Lance le MLflow tracking server"""
    
    print("="*70)
    print("🚀 Starting MLflow Tracking Server")
    print("="*70)
    print(f"Backend Store: {MLFLOW_BACKEND_STORE_URI[:50] if MLFLOW_BACKEND_STORE_URI else 'NOT SET'}...")
    print(f"Artifact Root: {MLFLOW_ARTIFACT_ROOT}")
    print(f"R2 Endpoint: {R2_ENDPOINT_URL}")
    print(f"Port: {PORT}")
    print("="*70)
    
    # Vérifie que les variables sont configurées
    if not MLFLOW_BACKEND_STORE_URI:
        print("❌ ERROR: NEONDB_MLFLOW not set")
        print("Configure it in HuggingFace Spaces Settings")
        sys.exit(1)
    
    if not MLFLOW_ARTIFACT_ROOT:
        print("❌ ERROR: R2_WR_MLFLOW_URI not set")
        sys.exit(1)
    
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("❌ ERROR: R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY not set")
        sys.exit(1)
    
    if not R2_ENDPOINT_URL:
        print("❌ ERROR: R2_ENDPOINT_URL not set")
        sys.exit(1)
    
    # ✅ Configure les variables d'environnement S3 pour boto3
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = R2_ENDPOINT_URL
    
    print("✅ S3/R2 credentials configured")
    
    # Commande MLflow
    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--backend-store-uri", MLFLOW_BACKEND_STORE_URI,
        "--default-artifact-root", MLFLOW_ARTIFACT_ROOT
    ]
    
    # Lance MLflow
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ MLflow server failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
