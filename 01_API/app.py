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

# (reste des endpoints /insert et /load identiques)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)