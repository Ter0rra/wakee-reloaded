---
license: mit
tags:
- emotion-detection
- daisee
- efficientnet
- pytorch
datasets:
- daisee
---

# Wakee - Emotion Detection Model

Version: **v2026.02.09.0047**

## Model Description

EfficientNet B4 fine-tuned for emotion detection in educational settings.

Predicts 4 emotion intensities (0-3 scale):
- Boredom
- Confusion  
- Engagement
- Frustration

## Training Data

- Base: DAiSEE dataset
- Fine-tuned: User-validated annotations from Wakee app

## Usage

### ONNX (Production)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load model
session = ort.InferenceSession("model.onnx")

# Preprocess image (224x224)
image = Image.open("image.jpg").resize((224, 224))
input_array = np.array(image).transpose(2, 0, 1).astype(np.float32)
input_array = np.expand_dims(input_array, axis=0) / 255.0

# Predict
outputs = session.run(['output'], {'input': input_array})
boredom, confusion, engagement, frustration = outputs[0][0]
```

### PyTorch (Fine-tuning)

```python
import torch
from torchvision import models

# Load checkpoint
model = models.efficientnet_b4()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("model.bin"))
```

## Model Card

- **Architecture**: EfficientNet B4
- **Framework**: PyTorch 2.1.2
- **Input**: RGB images (224x224)
- **Output**: 4 emotion scores (regression)
- **License**: MIT

## Metrics

See model_versions table in database for evaluation metrics.

## Citation

```bibtex
@software{wakee_emotion_detection,
  author = {Terorra},
  title = {Wakee Emotion Detection Model},
  year = {2025},
  version = {v2026.02.09.0047},
}
```
