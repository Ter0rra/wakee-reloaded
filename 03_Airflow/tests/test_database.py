"""
Tests Pytest pour NeonDB
"""

import pytest
from sqlalchemy import create_engine, text
import os

NEON_DATABASE_URL = os.getenv("NEONDB_WR")

pytestmark = pytest.mark.skipif(
    not NEON_DATABASE_URL,
    reason="NEONDB_WR not configured"
)

@pytest.fixture(scope="module")
def db_engine():
    """Crée une connexion à NeonDB"""
    engine = create_engine(NEON_DATABASE_URL)
    yield engine
    engine.dispose()

# ============================================================================
# TESTS Connexion
# ============================================================================

def test_database_connection(db_engine):
    """Vérifie que la connexion fonctionne"""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1

def test_database_version(db_engine):
    """Vérifie la version PostgreSQL"""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        assert 'PostgreSQL' in version

# ============================================================================
# TESTS Tables
# ============================================================================

def test_required_tables_exist(db_engine):
    """Vérifie que toutes les tables requises existent"""
    required_tables = ['emotion_labels', 'drift_reports', 'model_versions']
    
    with db_engine.connect() as conn:
        query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        existing_tables = [row[0] for row in conn.execute(query)]
    
    for table in required_tables:
        assert table in existing_tables

def test_emotion_labels_has_required_columns(db_engine):
    """Vérifie la structure de emotion_labels"""
    with db_engine.connect() as conn:
        query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'emotion_labels'
        """)
        columns = [row[0] for row in conn.execute(query)]
    
    required_columns = [
        'id', 'img_name', 's3_path', 
        'predicted_boredom', 'predicted_confusion', 
        'predicted_engagement', 'predicted_frustration',
        'user_boredom', 'user_confusion', 
        'user_engagement', 'user_frustration',
        'created_at'
    ]
    
    for col in required_columns:
        assert col in columns

# ============================================================================
# TESTS CRUD
# ============================================================================

def test_can_count_emotion_labels(db_engine):
    """Vérifie qu'on peut compter les annotations"""
    with db_engine.connect() as conn:
        query = text("SELECT COUNT(*) FROM emotion_labels")
        count = conn.execute(query).fetchone()[0]
        assert count >= 0

def test_can_filter_by_date(db_engine):
    """Vérifie qu'on peut filtrer par date"""
    with db_engine.connect() as conn:
        query = text("""
            SELECT COUNT(*) 
            FROM emotion_labels 
            WHERE created_at >= NOW() - INTERVAL '30 days'
        """)
        count = conn.execute(query).fetchone()[0]
        assert count >= 0

# ============================================================================
# TESTS Performance
# ============================================================================

def test_query_performance(db_engine):
    """Vérifie que les requêtes sont rapides"""
    import time
    
    with db_engine.connect() as conn:
        start = time.time()
        query = text("SELECT COUNT(*) FROM emotion_labels")
        conn.execute(query).fetchone()
        elapsed = time.time() - start
        
        assert elapsed < 1.0
