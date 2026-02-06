"""
Wakee Airflow - Utils Package
Fonctions utilitaires partagées entre les DAGs
"""

# Database helpers
from .database_helpers import (
    get_db_engine,
    fetch_recent_annotations,
    save_drift_report,
    update_retrain_triggered
)

# Drift calculator
from .drift_calculator import (
    calculate_drift_metrics,
    check_drift_threshold
)

# Data loader (pour DAG 3)
from .data_loader import (
    prepare_training_data,
    split_dataset,
    download_images_from_r2
)

# Model trainer (pour DAG 3)
from .model_trainer import (
    finetune_model,
    load_pretrained_model,
    save_model
)

# ONNX exporter (pour DAG 3)
from .onnx_exporter import (
    export_to_onnx,
    export_and_verify
)

# HF uploader (pour DAG 3)
from .hf_uploader import (
    upload_model_to_hf,
    download_model_from_hf
)

__all__ = [
    # Database
    'get_db_engine',
    'fetch_recent_annotations',
    'save_drift_report',
    'update_retrain_triggered',
    # Drift
    'calculate_drift_metrics',
    'check_drift_threshold',
    # Data
    'prepare_training_data',
    'split_dataset',
    'download_images_from_r2',
    # Training
    'finetune_model',
    'load_pretrained_model',
    'save_model',
    # Export
    'export_to_onnx',
    'export_and_verify',
    # Upload
    'upload_model_to_hf',
    'download_model_from_hf',
]
