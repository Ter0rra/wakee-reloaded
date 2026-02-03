"""
CNN Module - Détection émotions
Charge le modèle depuis HuggingFace Model Hub
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

HF_MODEL_REPO = "Terorra/wakee-reloaded"
MODEL_FILENAME = "model.onnx"
MODEL_CACHE_DIR = "./model_cache"  # Cache local pour éviter retéléchargement

# ============================================================================
# CHARGEMENT MODÈLE
# ============================================================================

def load_model():
    """
    Charge le modèle ONNX depuis HuggingFace Model Hub
    Utilise un cache local pour éviter de retélécharger à chaque lancement
    """
    try:
        print(f"📥 Chargement du modèle depuis HF Model Hub...")
        print(f"   Repo : {HF_MODEL_REPO}")
        print(f"   File : {MODEL_FILENAME}")
        
        # Télécharge depuis HF (ou utilise le cache)
        model_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILENAME,
            cache_dir=MODEL_CACHE_DIR
        )
        
        print(f"✅ Modèle téléchargé : {model_path}")
        
        # Charge la session ONNX
        session = ort.InferenceSession(model_path)
        
        print(f"✅ Modèle ONNX chargé avec succès")
        return session
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        print(f"⚠️  Tentative de chargement depuis le modèle legacy local...")
        
        # Fallback : modèle local si échec
        try:
            legacy_path = "model_legacy/daisee_model.onnx"
            if os.path.exists(legacy_path):
                print(f"📂 Chargement depuis : {legacy_path}")
                session = ort.InferenceSession(legacy_path)
                print(f"✅ Modèle legacy chargé")
                return session
            else:
                raise FileNotFoundError(f"Modèle legacy introuvable : {legacy_path}")
        except Exception as e2:
            print(f"❌ Erreur fatale : {e2}")
            raise

# Charge le modèle au démarrage du module
onnx_session = load_model()

# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocessing identique à l'API
    """
    # Resize to 256x256
    img = pil_image.resize((256, 256), Image.BILINEAR)
    
    # Center crop to 224x224
    left = (256 - 224) // 2
    top = (256 - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    
    # Convert to array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

# ============================================================================
# PRÉDICTION
# ============================================================================

def predict(image: Image.Image) -> dict:
    """
    Prédit les 4 émotions depuis une image
    
    Args:
        image: PIL Image
    
    Returns:
        dict: {
            "boredom": float (0-3),
            "confusion": float (0-3),
            "engagement": float (0-3),
            "frustration": float (0-3)
        }
    """
    # Preprocessing
    input_tensor = preprocess_image(image)
    
    # Inference
    outputs = onnx_session.run(['output'], {'input': input_tensor})
    scores = outputs[0][0]
    
    return {
        "boredom": float(scores[0]),
        "confusion": float(scores[1]),
        "engagement": float(scores[2]),
        "frustration": float(scores[3])
    }


def get_emotion(pil_image):
    """
    Infers the emotional state from a given PIL image using a pre-trained ONNX model.

    This function loads an ONNX model, preprocesses the input PIL image to match the
    model's expected input format, and then performs an inference to get predictions
    for emotional states.

    Args:
        pil_image (PIL.Image.Image): The input image in PIL format.

    Returns:
        numpy.ndarray: The raw prediction outputs from the ONNX model,
                       typically representing probabilities or logits for different emotion classes.
    """

    # # Charger le modèle ONNX
    # session = ort.InferenceSession("daisee_model.onnx")

     # loading model (lazy loading)
    # session = _load_model()

    # Define the image transformations required by the model

    # Apply transformations, add a batch dimension, and convert to a NumPy array
    input_tensor = preprocess_image(pil_image) # (1, 3, 224, 224)

    # Run the inference on the ONNX model
    # 'output' is the name of the output tensor, 'input' is the name of the input tensor
    # outputs = session.run(['output'], {'input': input_tensor})
    outputs = onnx_session.run(['output'], {'input': input_tensor})
    preds = outputs[0] # Extract the actual predictions from the output list
    

    return preds