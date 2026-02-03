"""
Wakee API - Production
ONNX Runtime UNIQUEMENT (pas de PyTorch)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from PIL import Image
import io
import numpy as np
from datetime import datetime
import base64
import os

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import boto3
from botocore.exceptions import ClientError

# ============================================================================
# PREPROCESSING SANS PYTORCH (Pillow + numpy)
# ============================================================================

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocessing identique à ton cnn.py
    SANS dépendance PyTorch (juste Pillow + numpy)
    """
    # 1. Resize to 256x256
    img = pil_image.resize((256, 256), Image.BILINEAR)
    
    # 2. Center crop to 224x224
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    
    # 3. Convert to numpy array [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # 4. ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # 5. Transpose to CHW (channels, height, width)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # 6. Add batch dimension (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

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
                print(f"✅ .env chargé depuis : {dotenv_path}")
        except ImportError:
            print("⚠️  python-dotenv non installé (OK en production)")

load_env_vars()

HF_MODEL_REPO = "Terorra/wakee-reloaded"
MODEL_FILENAME = "model.onnx"

NEON_DATABASE_URL = os.getenv("NEONDB_WR")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_WR_IMG_BUCKET_NAME", "wr-img-store")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PredictionResponse(BaseModel):
    boredom: float = Field(..., ge=0, le=3)
    confusion: float = Field(..., ge=0, le=3)
    engagement: float = Field(..., ge=0, le=3)
    frustration: float = Field(..., ge=0, le=3)
    timestamp: str

class AnnotationInsert(BaseModel):
    image_base64: str
    predicted_boredom: float = Field(..., ge=0, le=3)
    predicted_confusion: float = Field(..., ge=0, le=3)
    predicted_engagement: float = Field(..., ge=0, le=3)
    predicted_frustration: float = Field(..., ge=0, le=3)
    user_boredom: float = Field(..., ge=0, le=3)
    user_confusion: float = Field(..., ge=0, le=3)
    user_engagement: float = Field(..., ge=0, le=3)
    user_frustration: float = Field(..., ge=0, le=3)

class InsertResponse(BaseModel):
    status: str
    message: str
    img_name: str
    s3_url: Optional[str] = None

class LoadResponse(BaseModel):
    total_samples: int
    validated_samples: int
    recent_predictions: List[dict]
    statistics: dict

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Wakee Emotion API",
    description="Multi-label emotion detection (ONNX Runtime)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

onnx_session = None
db_engine = None
s3_client = None

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    global onnx_session, db_engine, s3_client
    
    print("=" * 70)
    print("🚀 DÉMARRAGE API WAKEE (ONNX Runtime)")
    print("=" * 70)
    
    # 1. Download model from HF Model Hub
    try:
        print(f"\n📥 Téléchargement du modèle ONNX...")
        print(f"   Repo : {HF_MODEL_REPO}")
        print(f"   File : {MODEL_FILENAME}")
        
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILENAME,
            cache_dir="/tmp/models"
        )
        
        # Load ONNX session (PAS DE PYTORCH !)
        onnx_session = ort.InferenceSession(model_path)
        
        input_name = onnx_session.get_inputs()[0].name
        input_shape = onnx_session.get_inputs()[0].shape
        
        print(f"✅ Modèle ONNX chargé : {model_path}")
        print(f"   Input : {input_name} {input_shape}\n")
        
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}\n")
        onnx_session = None
    
    # 2. Database
    if NEON_DATABASE_URL:
        try:
            db_engine = create_engine(NEON_DATABASE_URL)
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Connexion NeonDB établie\n")
        except Exception as e:
            print(f"⚠️  NeonDB non disponible : {e}\n")
            db_engine = None
    else:
        print("⚠️  NEON_DATABASE_URL non défini\n")
    
    # 3. Cloudflare R2
    if all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                region_name='auto'
            )
            s3_client.head_bucket(Bucket=R2_BUCKET_NAME)
            print(f"✅ Connexion Cloudflare R2 (bucket: {R2_BUCKET_NAME})\n")
        except Exception as e:
            print(f"⚠️  Cloudflare R2 non disponible : {e}\n")
            s3_client = None
    else:
        print("⚠️  R2 secrets non définis\n")
    
    print("=" * 70)
    print("🎉 API WAKEE PRÊTE !")
    print("=" * 70)
    print(f"📊 Status :")
    print(f"   - Modèle ONNX : {'✅' if onnx_session else '❌'}")
    print(f"   - Database : {'✅' if db_engine else '❌'}")
    print(f"   - Storage : {'✅' if s3_client else '❌'}")
    print("=" * 70 + "\n")

# ============================================================================
# ENDPOINTS (identiques à avant)
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Wakee Emotion API",
        "version": "1.0.0",
        "runtime": "ONNX Runtime (no PyTorch)",
        "model_source": HF_MODEL_REPO
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": onnx_session is not None,
        "runtime": "ONNX",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    if not onnx_session:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess (SANS PyTorch !)
        input_tensor = preprocess_image(image)
        
        # Inference ONNX
        outputs = onnx_session.run(['output'], {'input': input_tensor})
        scores_array = outputs[0][0]
        
        return PredictionResponse(
            boredom=round(float(scores_array[0]), 2),
            confusion=round(float(scores_array[1]), 2),
            engagement=round(float(scores_array[2]), 2),
            frustration=round(float(scores_array[3]), 2),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"❌ Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Prédiction des 4 émotions depuis une image
    
    ⚠️ RIEN N'EST SAUVEGARDÉ à cette étape
    
    L'utilisateur doit ensuite appeler /insert pour sauvegarder
    """
    
    if not onnx_session:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded"
        )
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # 1. Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Preprocessing
        input_tensor = preprocess_image(image)
        
        # 3. Inference ONNX
        outputs = onnx_session.run(['output'], {'input': input_tensor})
        scores_array = outputs[0][0]
        
        # 4. Format résultats
        return PredictionResponse(
            boredom=round(float(scores_array[0]), 2),
            confusion=round(float(scores_array[1]), 2),
            engagement=round(float(scores_array[2]), 2),
            frustration=round(float(scores_array[3]), 2),
            timestamp=datetime.now().isoformat()
        )
        
        # ⚠️ PAS de sauvegarde R2
        # ⚠️ PAS de sauvegarde NeonDB
        # → L'utilisateur décide s'il valide via /insert
    
    except Exception as e:
        print(f"❌ Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert", response_model=InsertResponse)
async def insert_annotation(annotation: AnnotationInsert):
    """
    Insert annotation utilisateur
    
    Ce endpoint fait 2 choses :
    1. Upload image vers Cloudflare R2
    2. Insert labels (predicted + user) dans NeonDB
    
    ✅ Appelé uniquement quand l'utilisateur clique "Valider"
    """
    
    # Vérifications
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database not available")
    
    if not s3_client:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    try:
        # 1. Decode image base64
        try:
            image_bytes = base64.b64decode(annotation.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        
        # 2. Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{timestamp}_{hash(annotation.image_base64) % 10000:04d}.jpg"
        s3_key = f"collected/{img_name}"
        
        # 3. Upload image to Cloudflare R2
        print(f"📤 Upload vers R2 : {s3_key}")
        try:
            s3_client.put_object(
                Bucket=R2_BUCKET_NAME,
                Key=s3_key,
                Body=image_bytes,
                ContentType='image/jpeg'
            )
            print(f"✅ Upload R2 réussi : {img_name}")
        except ClientError as e:
            print(f"❌ Erreur upload R2 : {e}")
            raise HTTPException(status_code=500, detail=f"R2 upload failed: {e}")
        
        # 4. Insert labels in NeonDB
        query = text("""
            INSERT INTO emotion_labels 
            (img_name, s3_path, 
             predicted_boredom, predicted_confusion, predicted_engagement, predicted_frustration,
             user_boredom, user_confusion, user_engagement, user_frustration,
             source, is_validated, timestamp)
            VALUES 
            (:img_name, :s3_path,
             :pred_boredom, :pred_confusion, :pred_engagement, :pred_frustration,
             :user_boredom, :user_confusion, :user_engagement, :user_frustration,
             'app_sourcing', TRUE, :timestamp)
        """)
        
        with db_engine.connect() as conn:
            conn.execute(query, {
                'img_name': img_name,
                's3_path': s3_key,
                'pred_boredom': annotation.predicted_boredom,
                'pred_confusion': annotation.predicted_confusion,
                'pred_engagement': annotation.predicted_engagement,
                'pred_frustration': annotation.predicted_frustration,
                'user_boredom': annotation.user_boredom,
                'user_confusion': annotation.user_confusion,
                'user_engagement': annotation.user_engagement,
                'user_frustration': annotation.user_frustration,
                'timestamp': datetime.now()
            })
            conn.commit()
        
        print(f"✅ Insert NeonDB réussi : {img_name}")
        
        # 5. Generate public URL (si tu as activé l'accès public)
        # public_url = f"https://pub-{R2_ACCOUNT_ID}.r2.dev/{s3_key}"
        # Ou None si pas d'accès public
        public_url = None
        
        return InsertResponse(
            status="success",
            message="Image uploaded to R2 and labels saved to NeonDB",
            img_name=img_name,
            s3_url=public_url
        )
    
    except SQLAlchemyError as e:
        print(f"❌ Erreur NeonDB : {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Erreur insert : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load", response_model=LoadResponse)
async def load_data(limit: int = 10):
    """
    Charge les données depuis NeonDB
    
    Retourne :
    - Nombre total d'échantillons
    - Nombre d'échantillons validés
    - Dernières prédictions (avec corrections utilisateur)
    - Statistiques globales
    """
    
    if not db_engine:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        with db_engine.connect() as conn:
            # Total samples
            total = conn.execute(text(
                "SELECT COUNT(*) FROM emotion_labels"
            )).scalar()
            
            # Validated samples (ceux insérés via /insert)
            validated = conn.execute(text(
                "SELECT COUNT(*) FROM emotion_labels WHERE is_validated = TRUE"
            )).scalar()
            
            # Recent predictions
            recent = conn.execute(text(f"""
                SELECT 
                    img_name,
                    s3_path,
                    predicted_boredom,
                    predicted_confusion,
                    predicted_engagement,
                    predicted_frustration,
                    user_boredom,
                    user_confusion,
                    user_engagement,
                    user_frustration,
                    timestamp
                FROM emotion_labels
                WHERE is_validated = TRUE
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {'limit': limit}).fetchall()
            
            recent_list = [
                {
                    'img_name': row[0],
                    's3_path': row[1],
                    'predicted': {
                        'boredom': float(row[2]),
                        'confusion': float(row[3]),
                        'engagement': float(row[4]),
                        'frustration': float(row[5])
                    },
                    'user_corrected': {
                        'boredom': float(row[6]),
                        'confusion': float(row[7]),
                        'engagement': float(row[8]),
                        'frustration': float(row[9])
                    },
                    'timestamp': row[10].isoformat() if row[10] else None
                }
                for row in recent
            ]
            
            # Statistics (moyennes)
            stats = conn.execute(text("""
                SELECT 
                    AVG(predicted_boredom) as avg_pred_boredom,
                    AVG(predicted_confusion) as avg_pred_confusion,
                    AVG(predicted_engagement) as avg_pred_engagement,
                    AVG(predicted_frustration) as avg_pred_frustration,
                    AVG(user_boredom) as avg_user_boredom,
                    AVG(user_confusion) as avg_user_confusion,
                    AVG(user_engagement) as avg_user_engagement,
                    AVG(user_frustration) as avg_user_frustration
                FROM emotion_labels
                WHERE is_validated = TRUE
            """)).fetchone()
            
            statistics = {
                'predictions': {
                    'boredom': round(float(stats[0] or 0), 2),
                    'confusion': round(float(stats[1] or 0), 2),
                    'engagement': round(float(stats[2] or 0), 2),
                    'frustration': round(float(stats[3] or 0), 2)
                },
                'user_corrections': {
                    'boredom': round(float(stats[4] or 0), 2),
                    'confusion': round(float(stats[5] or 0), 2),
                    'engagement': round(float(stats[6] or 0), 2),
                    'frustration': round(float(stats[7] or 0), 2)
                }
            }
        
        return LoadResponse(
            total_samples=total or 0,
            validated_samples=validated or 0,
            recent_predictions=recent_list,
            statistics=statistics
        )
    
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)