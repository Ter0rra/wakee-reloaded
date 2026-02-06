"""
Data Loader - Download training data
Récupère images depuis R2 et annotations depuis NeonDB
"""

import boto3
from botocore.exceptions import ClientError
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import os
from typing import Tuple, List
import tempfile

# ============================================================================
# CONFIGURATION
# ============================================================================

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
# R2_BUCKET_NAME = os.getenv("R2_WR_IMG_BUCKET_NAME", "wr-img-store")
R2_BUCKET_NAME = "wr-img-store"
R2_URI = os.getenv('R2_URI')
NEON_DATABASE_URL = os.getenv("NEONDB_WR")

# ============================================================================
# DOWNLOAD FROM R2
# ============================================================================

def download_images_from_r2(
    s3_paths: List[str],
    local_dir: str
) -> List[str]:
    """
    Télécharge des images depuis R2
    
    Args:
        s3_paths (list): Liste des chemins S3 (ex: 'img_001.jpg')
        local_dir (str): Répertoire local de destination
    
    Returns:
        list: Chemins locaux des images téléchargées
    """
    print(f"📥 Downloading {len(s3_paths)} images from R2...")
    
    # Créer le client S3
    s3_client = boto3.client(
        's3',
        endpoint_url=R2_URI,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )
    
    # Créer le répertoire local
    os.makedirs(local_dir, exist_ok=True)
    
    local_paths = []
    failed_downloads = []
    
    for i, s3_path in enumerate(s3_paths):
        try:
            # Nom du fichier local
            filename = os.path.basename(s3_path)
            local_path = os.path.join(local_dir, filename)
            
            # Download
            # s3_client.download_file(R2_BUCKET_NAME, s3_path, local_path)
            s3_client.download_file(R2_BUCKET_NAME, s3_path, local_path)
            local_paths.append(local_path)
            
            if (i + 1) % 100 == 0:
                print(f"  Downloaded {i + 1}/{len(s3_paths)} images")
        
        except ClientError as e:
            print(f"  ⚠️  Failed to download {s3_path}: {e}")
            failed_downloads.append(s3_path)
    
    print(f"✅ Downloaded {len(local_paths)}/{len(s3_paths)} images")
    
    if failed_downloads:
        print(f"⚠️  {len(failed_downloads)} downloads failed")
    
    return local_paths

# ============================================================================
# FETCH ANNOTATIONS
# ============================================================================

def fetch_all_validated_annotations() -> pd.DataFrame:
    """
    Récupère toutes les annotations validées depuis NeonDB
    
    Returns:
        pd.DataFrame: Annotations avec user_* labels
    """
    print("📊 Fetching all validated annotations from NeonDB...")
    
    engine = create_engine(NEON_DATABASE_URL)
    
    query = text("""
        SELECT 
            id,
            img_name,
            s3_path,
            user_boredom,
            user_confusion,
            user_engagement,
            user_frustration,
            timestamp
        FROM emotion_labels
        WHERE is_validated = TRUE
        ORDER BY timestamp DESC
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    print(f"✅ Fetched {len(df)} validated annotations")
    
    return df

# ============================================================================
# PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(
    min_samples: int = 100,
    download_dir: str = None
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Prépare les données d'entraînement
    
    Args:
        min_samples (int): Minimum d'échantillons requis
        download_dir (str): Répertoire de téléchargement (temp si None)
    
    Returns:
        tuple: (image_paths, labels, metadata_df)
    """
    # Fetch annotations
    df = fetch_all_validated_annotations()
    
    if len(df) < min_samples:
        raise ValueError(
            f"Not enough training data: {len(df)} samples "
            f"(minimum: {min_samples})"
        )
    
    # Créer répertoire de téléchargement
    if download_dir is None:
        download_dir = tempfile.mkdtemp(prefix='wakee_training_')
    
    print(f"📁 Download directory: {download_dir}")
    
    # Download images
    s3_paths = df['s3_path'].tolist()
    local_paths = download_images_from_r2(s3_paths, download_dir)
    
    # Prépare les labels (N, 4)
    labels = df[['user_boredom', 'user_confusion', 'user_engagement', 'user_frustration']].values
    
    print(f"✅ Training data prepared:")
    print(f"   Images: {len(local_paths)}")
    print(f"   Labels shape: {labels.shape}")
    
    return local_paths, labels, df

# ============================================================================
# SPLIT DATASET
# ============================================================================

def split_dataset(
    image_paths: List[str],
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple:
    """
    Split dataset en train/val/test
    
    Args:
        image_paths (list): Chemins des images
        labels (np.ndarray): Labels (N, 4)
        train_ratio (float): Proportion train
        val_ratio (float): Proportion val
        test_ratio (float): Proportion test
        seed (int): Random seed
    
    Returns:
        tuple: (train_data, val_data, test_data)
               Chaque élément est (image_paths, labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(seed)
    
    n_samples = len(image_paths)
    indices = np.random.permutation(n_samples)
    
    # Calcule les indices de split
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split
    train_images = [image_paths[i] for i in train_indices]
    train_labels = labels[train_indices]
    
    val_images = [image_paths[i] for i in val_indices]
    val_labels = labels[val_indices]
    
    test_images = [image_paths[i] for i in test_indices]
    test_labels = labels[test_indices]
    
    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_images)} samples ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_images)} samples ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_images)} samples ({test_ratio*100:.0f}%)")
    
    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
