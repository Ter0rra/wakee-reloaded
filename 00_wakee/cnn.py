import onnxruntime as ort
import numpy as np
import torchvision.transforms.v2 as transforms

# Global variable to store the session
_session = None

def _load_model():
    """Load the ONNX model"""
    global _session
    if _session is None:
        try:
            _session = ort.InferenceSession("daisee_model.onnx")
            # print("✅ Modèle ONNX chargé avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")
            raise
    return _session

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
    session = _load_model()

    # Define the image transformations required by the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                            [0.229, 0.224, 0.225])
    ])
    # Apply transformations, add a batch dimension, and convert to a NumPy array
    input_tensor = transform(pil_image).unsqueeze(0).numpy()  # (1, 3, 224, 224)

    # Run the inference on the ONNX model
    # 'output' is the name of the output tensor, 'input' is the name of the input tensor
    outputs = session.run(['output'], {'input': input_tensor})
    preds = outputs[0] # Extract the actual predictions from the output list
    

    return preds