-- Table drift_reports
-- Stocke les rapports de drift quotidiens

CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    report_date DATE NOT NULL UNIQUE,
    drift_detected BOOLEAN NOT NULL,
    drift_score FLOAT NOT NULL,
    
    -- MAE par émotion
    mae_boredom FLOAT,
    mae_confusion FLOAT,
    mae_engagement FLOAT,
    mae_frustration FLOAT,
    
    -- MSE par émotion
    mse_boredom FLOAT,
    mse_confusion FLOAT,
    mse_engagement FLOAT,
    mse_frustration FLOAT,
    
    -- Métadonnées
    num_samples INT NOT NULL,
    drift_threshold FLOAT DEFAULT 0.15,
    
    -- Rapport détaillé (JSON Evidently)
    report_json JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index pour recherche rapide par date
CREATE INDEX IF NOT EXISTS idx_drift_reports_date 
ON drift_reports(report_date DESC);

-- Index pour recherche des drifts détectés
CREATE INDEX IF NOT EXISTS idx_drift_reports_detected 
ON drift_reports(drift_detected) 
WHERE drift_detected = TRUE;

-- Commentaires
COMMENT ON TABLE drift_reports IS 'Rapports de drift quotidiens du modèle Wakee';
COMMENT ON COLUMN drift_reports.drift_score IS 'Score de drift moyen (MAE global)';
COMMENT ON COLUMN drift_reports.drift_threshold IS 'Seuil utilisé pour cette détection';
