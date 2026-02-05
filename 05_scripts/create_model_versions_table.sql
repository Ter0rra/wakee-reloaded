-- Table model_versions
-- Stocke l'historique des versions du modèle

CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version_name VARCHAR(100) NOT NULL UNIQUE,
    
    -- Métadonnées modèle
    model_type VARCHAR(50) DEFAULT 'EfficientNet-B4',
    framework VARCHAR(50) DEFAULT 'PyTorch',
    
    -- Fichiers
    hf_model_bin_url TEXT,
    hf_model_onnx_url TEXT,
    
    -- Métriques d'évaluation
    accuracy FLOAT,
    f1_score FLOAT,
    mae_boredom FLOAT,
    mae_confusion FLOAT,
    mae_engagement FLOAT,
    mae_frustration FLOAT,
    
    -- Training info
    n_samples_train INT,
    n_samples_val INT,
    n_samples_test INT,
    n_epochs INT,
    learning_rate FLOAT,
    batch_size INT,
    
    -- MLflow
    mlflow_run_id VARCHAR(100),
    mlflow_experiment_id VARCHAR(100),
    
    -- Déploiement
    is_production BOOLEAN DEFAULT FALSE,
    deployed_at TIMESTAMP,
    
    -- Drift qui a déclenché ce retrain
    triggered_by_drift_report_id INT REFERENCES drift_reports(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index pour recherche rapide
CREATE INDEX IF NOT EXISTS idx_model_versions_production 
ON model_versions(is_production) 
WHERE is_production = TRUE;

CREATE INDEX IF NOT EXISTS idx_model_versions_created 
ON model_versions(created_at DESC);

-- Commentaires
COMMENT ON TABLE model_versions IS 'Historique des versions du modèle Wakee';
COMMENT ON COLUMN model_versions.version_name IS 'Nom unique de la version (ex: v1.0.0, v1.1.0)';
COMMENT ON COLUMN model_versions.is_production IS 'TRUE si ce modèle est déployé en production';
