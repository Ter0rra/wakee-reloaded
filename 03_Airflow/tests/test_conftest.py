"""
Pytest Configuration & Fixtures
Fixtures partagées pour tous les tests
"""

import pytest
import os
from PIL import Image
import io

# ============================================================================
# FIXTURES - URLs & Configuration
# ============================================================================

@pytest.fixture(scope="session")
def api_url():
    """URL de l'API Wakee"""
    return os.getenv("API_URL", "https://terorra-wakee-api.hf.space")

@pytest.fixture(scope="session")
def sourcing_url():
    """URL de l'app Sourcing"""
    return os.getenv("SOURCING_URL", "https://terorra-wakee-sourcing.hf.space")

@pytest.fixture(scope="session")
def hf_model_repo():
    """Repository HuggingFace du modèle"""
    return os.getenv("HF_MODEL_REPO", "Terorra/wakee-reloaded")

# ============================================================================
# FIXTURES - Database
# ============================================================================

@pytest.fixture(scope="session")
def neon_db_url():
    """URL de la base de données NeonDB"""
    url = os.getenv("NEON_DATABASE_URL")
    if not url:
        pytest.skip("NEON_DATABASE_URL not configured")
    return url

# ============================================================================
# FIXTURES - Storage
# ============================================================================

@pytest.fixture(scope="session")
def r2_credentials():
    """Credentials Cloudflare R2"""
    creds = {
        'account_id': os.getenv("R2_ACCOUNT_ID"),
        'access_key_id': os.getenv("R2_ACCESS_KEY_ID"),
        'secret_access_key': os.getenv("R2_SECRET_ACCESS_KEY"),
        'bucket_name': os.getenv("R2_BUCKET_NAME", "wakee-bucket")
    }
    
    if not all([creds['account_id'], creds['access_key_id'], creds['secret_access_key']]):
        pytest.skip("R2 credentials not configured")
    
    return creds

# ============================================================================
# FIXTURES - Test Data
# ============================================================================

@pytest.fixture
def test_image_small():
    """Image de test petite (224x224 RGB)"""
    img = Image.new('RGB', (224, 224), color='blue')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    return img_buffer

@pytest.fixture
def test_image_large():
    """Image de test grande (1920x1080 RGB)"""
    img = Image.new('RGB', (1920, 1080), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    return img_buffer

@pytest.fixture
def test_file_content():
    """Contenu texte de test"""
    return b"Test content for Wakee pytest - " + os.urandom(32).hex().encode()

# ============================================================================
# PYTEST Configuration
# ============================================================================

def pytest_configure(config):
    """Configuration globale Pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
