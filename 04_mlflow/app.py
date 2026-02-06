"""
MLflow Tracking Server - Wakee Reloaded
Deployed on HuggingFace Spaces
SECURED: No secrets in logs
"""

import os
import subprocess
import sys

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
                print(f"✅ Local .env loaded from: {dotenv_path}")
        except ImportError:
            print("⚠️  python-dotenv not installed (OK in production)")

load_env_vars()

# Load environment variables
MLFLOW_BACKEND_STORE_URI = os.getenv("NEONDB_MLFLOW")
MLFLOW_ARTIFACT_ROOT = os.getenv("R2_WR_MLFLOW_URI")
AWS_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ENDPOINT_URL = os.getenv("R2_URI")
PORT = int(os.getenv("PORT", 7860))

def mask_secret(secret: str, show_chars: int = 4) -> str:
    """
    Masque un secret pour l'affichage sécurisé
    
    Args:
        secret (str): Secret à masquer
        show_chars (int): Nombre de caractères à montrer au début/fin
    
    Returns:
        str: Secret masqué (ex: "abc1...xyz9")
    """
    if not secret:
        return "NOT_SET"
    if len(secret) <= show_chars * 2:
        return "***"
    return f"{secret[:show_chars]}...{secret[-show_chars:]}"

def mask_db_uri(uri: str) -> str:
    """
    Masque le password dans une URI PostgreSQL
    
    Args:
        uri (str): URI PostgreSQL
    
    Returns:
        str: URI avec password masqué
    """
    if not uri:
        return "NOT_SET"
    
    try:
        # Format: postgresql://user:password@host/db
        if '@' in uri and ':' in uri:
            before_at = uri.split('@')[0]
            after_at = uri.split('@')[1]
            
            # Split user:password
            if ':' in before_at:
                protocol_user = before_at.rsplit(':', 1)[0]  # postgresql://user
                # password = before_at.rsplit(':', 1)[1]  # On ne l'affiche pas
                
                return f"{protocol_user}:***@{after_at}"
        
        return uri  # Si format non reconnu, retourne tel quel
    except Exception:
        return "MASKED_URI"

def validate_s3_uri(uri: str) -> bool:
    """
    Vérifie que l'URI artifacts est au format S3
    
    Args:
        uri (str): URI à vérifier
    
    Returns:
        bool: True si format valide
    """
    if not uri:
        return False
    
    if not uri.startswith('s3://'):
        print(f"❌ ERROR: Artifact root must start with 's3://'")
        print(f"   Current: {uri}")
        print(f"   Expected: s3://bucket-name/prefix/")
        return False
    
    return True

def main():
    """Lance le MLflow tracking server"""
    
    print("="*70)
    print("🚀 Starting MLflow Tracking Server")
    print("="*70)
    
    # ============================================================================
    # AFFICHAGE SÉCURISÉ DES VARIABLES
    # ============================================================================
    
    # print(f"Backend Store: {mask_db_uri(MLFLOW_BACKEND_STORE_URI)}")
    # print(f"Artifact Root: {MLFLOW_ARTIFACT_ROOT if MLFLOW_ARTIFACT_ROOT else 'NOT_SET'}")
    # print(f"R2 Endpoint: {R2_ENDPOINT_URL if R2_ENDPOINT_URL else 'NOT_SET'}")
    # print(f"R2 Access Key: {mask_secret(AWS_ACCESS_KEY_ID)}")
    # print(f"R2 Secret Key: {mask_secret(AWS_SECRET_ACCESS_KEY)}")
    print(f"Port: {PORT}")
    print("="*70)
    
    # ============================================================================
    # VALIDATION DES VARIABLES
    # ============================================================================
    
    errors = []
    
    if not MLFLOW_BACKEND_STORE_URI:
        errors.append("NEONDB_MLFLOW is not set")
    
    if not MLFLOW_ARTIFACT_ROOT:
        errors.append("R2_WR_MLFLOW_URI is not set")
    elif not validate_s3_uri(MLFLOW_ARTIFACT_ROOT):
        errors.append("R2_WR_MLFLOW_URI must be in s3:// format")
    
    if not AWS_ACCESS_KEY_ID:
        errors.append("R2_ACCESS_KEY_ID is not set")
    
    if not AWS_SECRET_ACCESS_KEY:
        errors.append("R2_SECRET_ACCESS_KEY is not set")
    
    if not R2_ENDPOINT_URL:
        errors.append("R2_ENDPOINT_URL is not set")
    
    # Affiche les erreurs
    if errors:
        print("\n❌ CONFIGURATION ERRORS:")
        for i, error in enumerate(errors, 1):
            print(f"   {i}. {error}")
        print("\n💡 Configure these variables in HuggingFace Spaces Settings")
        print("="*70)
        sys.exit(1)
    
    print("✅ All required variables are set")
    
    # ============================================================================
    # CONFIGURATION S3/R2 POUR BOTO3
    # ============================================================================
    
    # Configure les variables d'environnement pour boto3
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = R2_ENDPOINT_URL
    
    print("✅ S3/R2 credentials configured (boto3)")
    print("="*70)
    
    # ============================================================================
    # LANCEMENT MLFLOW SERVER
    # ============================================================================
    
    cmd = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--backend-store-uri", MLFLOW_BACKEND_STORE_URI,
        "--default-artifact-root", MLFLOW_ARTIFACT_ROOT
    ]
    
    print("🚀 Starting MLflow server...")
    print(f"   Command: mlflow server --host 0.0.0.0 --port {PORT}")
    print("="*70)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("\n" + "="*70)
        print("❌ MLflow server failed to start")
        print("="*70)
        print(f"Error code: {e.returncode}")
        print("\n💡 Common issues:")
        print("   1. Database connection failed (check NEONDB_MLFLOW)")
        print("   2. Database migrations corrupted (DROP + CREATE database)")
        print("   3. R2 connection failed (check R2_ENDPOINT_URL)")
        print("="*70)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 MLflow server stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
