"""
Tests Pytest pour Cloudflare R2
"""

import pytest
import boto3
from botocore.exceptions import ClientError
import os

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_WR_IMG_BUCKET_NAME", "wr-img-store")

pytestmark = pytest.mark.skipif(
    not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]),
    reason="R2 credentials not configured"
)

@pytest.fixture(scope="module")
def s3_client():
    """Crée un client S3 pour R2"""
    client = boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )
    return client

# ============================================================================
# TESTS Connexion
# ============================================================================

def test_bucket_exists(s3_client):
    """Vérifie que le bucket existe"""
    try:
        s3_client.head_bucket(Bucket=R2_BUCKET_NAME)
    except ClientError:
        pytest.fail(f"Bucket {R2_BUCKET_NAME} not accessible")

def test_can_list_objects(s3_client):
    """Vérifie qu'on peut lister les objets"""
    response = s3_client.list_objects_v2(
        Bucket=R2_BUCKET_NAME,
        MaxKeys=10
    )
    assert 'Contents' in response or 'KeyCount' in response

# ============================================================================
# TESTS Upload/Download
# ============================================================================

def test_can_upload_file(s3_client, test_file_content):
    """Vérifie qu'on peut uploader un fichier"""
    test_key = f"pytest/test_{os.urandom(4).hex()}.txt"
    
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=test_key,
        Body=test_file_content
    )
    
    # Vérifie que le fichier existe
    response = s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=test_key)
    assert response['ContentLength'] == len(test_file_content)
    
    # Cleanup
    s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=test_key)

def test_can_download_file(s3_client, test_file_content):
    """Vérifie qu'on peut télécharger un fichier"""
    test_key = f"pytest/test_{os.urandom(4).hex()}.txt"
    
    # Upload
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=test_key,
        Body=test_file_content
    )
    
    # Download
    response = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=test_key)
    downloaded_content = response['Body'].read()
    
    assert downloaded_content == test_file_content
    
    # Cleanup
    s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=test_key)

def test_can_delete_file(s3_client, test_file_content):
    """Vérifie qu'on peut supprimer un fichier"""
    test_key = f"pytest/test_{os.urandom(4).hex()}.txt"
    
    # Upload
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=test_key,
        Body=test_file_content
    )
    
    # Delete
    s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=test_key)
    
    # Vérifie que le fichier n'existe plus
    with pytest.raises(ClientError):
        s3_client.head_object(Bucket=R2_BUCKET_NAME, Key=test_key)

# ============================================================================
# TESTS Performance
# ============================================================================

def test_upload_speed(s3_client, test_file_content):
    """Vérifie que l'upload est rapide (< 5s)"""
    import time
    
    test_key = f"pytest/test_{os.urandom(4).hex()}.txt"
    
    start = time.time()
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=test_key,
        Body=test_file_content
    )
    elapsed = time.time() - start
    
    # Cleanup
    s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key=test_key)
    
    assert elapsed < 5.0
