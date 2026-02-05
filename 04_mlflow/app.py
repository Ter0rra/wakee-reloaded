"""
MLflow Tracking Server - Wakee Reloaded
Deployed on HuggingFace Spaces
"""

import os
import subprocess
import sys

# Configuration
MLFLOW_BACKEND_STORE_URI = os.getenv("MLFLOW_BACKEND_STORE_URI")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# HF Spaces port
PORT = int(os.getenv("PORT", 7860))

def main():
    """Lance le MLflow tracking server"""
    
    print("="*70)
    print("🚀 Starting MLflow Tracking Server")
    print("="*70)
    print(f"Backend Store: {MLFLOW_BACKEND_STORE_URI[:50]}...")
    print(f"Artifact Root: {MLFLOW_ARTIFACT_ROOT}")
    print(f"Port: {PORT}")
    print("="*70)
    
    # Vérifie que les variables sont configurées
    if not MLFLOW_BACKEND_STORE_URI:
        print("❌ ERROR: MLFLOW_BACKEND_STORE_URI not set")
        print("Configure it in HuggingFace Spaces Settings")
        sys.exit(1)
    
    if not MLFLOW_ARTIFACT_ROOT:
        print("❌ ERROR: MLFLOW_ARTIFACT_ROOT not set")
        sys.exit(1)
    
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
