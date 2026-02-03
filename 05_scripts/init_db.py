"""
Initialisation NeonDB (Python 3.11)
Crée les tables pour le pipeline MLOps de ton bébé CNN
"""

import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Vérification Python 3.11
if sys.version_info < (3, 11):
    print(f"❌ Python 3.11+ requis, version actuelle : {sys.version}")
    sys.exit(1)

print(f"✅ Python version : {sys.version.split()[0]}\n")

# ============================================================================
# CONFIGURATION
# ============================================================================
env_path = Path("..") / ".env"
load_dotenv(dotenv_path=env_path)

NEON_DATABASE_URL = os.getenv("NEONDB_WR")

if not NEON_DATABASE_URL:
    print("❌ Variable d'environnement NEONDB_WR non définie\n")
    print("💡 Setup NeonDB :")
    print("   1. Va sur https://neon.tech")
    print("   2. Crée un projet gratuit")
    print("   3. Copie la connection string")
    print("   4. Export: export NEONDB_WR='postgresql://...'\n")
    sys.exit(1)

print(f"🔗 Connexion à NeonDB (Python 3.11)...\n")

# ============================================================================
# DÉFINITION DES TABLES
# ============================================================================

Base = declarative_base()

class EmotionLabel(Base):
    """Stocke prédictions + annotations pour ton bébé CNN"""
    __tablename__ = 'emotion_labels'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Image
    img_name = Column(String(255), unique=True, nullable=False, index=True)
    s3_path = Column(String(500), nullable=False)
    
    # Prédictions de ton CNN (0-3)
    predicted_boredom = Column(Float, nullable=False)
    predicted_confusion = Column(Float, nullable=False)
    predicted_engagement = Column(Float, nullable=False)
    predicted_frustration = Column(Float, nullable=False)
    
    # Annotations utilisateur (0-3)
    user_boredom = Column(Float, nullable=True)
    user_confusion = Column(Float, nullable=True)
    user_engagement = Column(Float, nullable=True)
    user_frustration = Column(Float, nullable=True)
    
    # Métadonnées
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(50), default='app_sourcing')
    is_validated = Column(Boolean, default=False, index=True)
    
    user_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)

class DriftReport(Base):
    """Résultats drift detection sur ton CNN"""
    __tablename__ = 'drift_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(DateTime, default=datetime.utcnow, index=True)
    
    drift_score_boredom = Column(Float)
    drift_score_confusion = Column(Float)
    drift_score_engagement = Column(Float)
    drift_score_frustration = Column(Float)
    drift_score_global = Column(Float)
    
    drift_detected = Column(Boolean, default=False)
    n_samples_analyzed = Column(Integer)
    report_url = Column(Text)
    retrain_triggered = Column(Boolean, default=False)

class ModelVersion(Base):
    """Versioning de ton bébé CNN"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    hf_commit_hash = Column(String(100))
    mlflow_run_id = Column(String(100))
    
    val_mae_boredom = Column(Float)
    val_mae_confusion = Column(Float)
    val_mae_engagement = Column(Float)
    val_mae_frustration = Column(Float)
    val_mae_global = Column(Float)
    
    is_production = Column(Boolean, default=False)
    n_training_samples = Column(Integer)
    training_duration_minutes = Column(Float)

# ============================================================================
# CRÉATION
# ============================================================================

print("=" * 70)
print("🗄️  CRÉATION TABLES NEONDB (Python 3.11)")
print("=" * 70 + "\n")

try:
    engine = create_engine(NEON_DATABASE_URL)
    
    print("📋 Tables à créer :")
    print("   1. emotion_labels (prédictions de ton CNN)")
    print("   2. drift_reports (monitoring)")
    print("   3. model_versions (versioning)\n")
    
    Base.metadata.create_all(engine)
    print("✅ Tables créées !\n")
    
    # Vérification
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"✅ Tables présentes : {tables}\n")
    
    # Test
    print("🧪 Test d'insertion...")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    test_label = EmotionLabel(
        img_name="test_python311.jpg",
        s3_path="test/test_python311.jpg",
        predicted_boredom=1.5,
        predicted_confusion=0.8,
        predicted_engagement=2.1,
        predicted_frustration=0.3,
        source="init_script_py311",
        is_validated=False
    )
    
    session.add(test_label)
    session.commit()
    print(f"✅ Test OK ! ID: {test_label.id}\n")
    
    session.delete(test_label)
    session.commit()
    session.close()
    print("✅ Test nettoyé\n")
    
except Exception as e:
    print(f"❌ ERREUR : {e}\n")
    sys.exit(1)

print("=" * 70)
print("✅ BASE DE DONNÉES PRÊTE ! (Python 3.11)")
print("=" * 70)
print("\n🎉 Ton bébé CNN a maintenant son infrastructure de données !\n")
print("📝 Prochaine étape : API FastAPI (demain)\n")