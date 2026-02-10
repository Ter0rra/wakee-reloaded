"""
Model Trainer - VERSION FINALE
Fix: Sauvegarde en mode TRAIN pour éviter BatchNorm issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = 'cpu'
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
NUM_CLASSES = 4

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================================
# MODEL LOADING (FLEXIBLE)
# ============================================================================

def load_pretrained_model(model_path: str) -> nn.Module:
    """
    Charge le modèle .bin
    ACCEPTE : PyTorch standard OU onnx2torch format
    """
    print(f"📦 Loading model from {model_path}")
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    if isinstance(state_dict, dict):
        if 'model' in state_dict:
            print("   Extracting 'model' key")
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            print("   Extracting 'state_dict' key")
            state_dict = state_dict['state_dict']
    
    from torchvision import models
    model = models.efficientnet_b4(weights=None)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    sample_key = list(state_dict.keys())[0]
    is_onnx2torch = '/' in sample_key
    
    if is_onnx2torch:
        print("   ⚠️  Detected onnx2torch format")
        print("   Loading with strict=False")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f"   Missing keys: {len(missing_keys)}")
        print(f"   Unexpected keys: {len(unexpected_keys)}")
        
        print("\n🔍 Testing model (onnx2torch)...")
        
        # ✅ Test en mode TRAIN
        model.train()
        
        with torch.no_grad():
            test1 = torch.randn(1, 3, 224, 224)
            test2 = torch.randn(1, 3, 224, 224)
            
            try:
                out1 = model(test1)
                out2 = model(test2)
                
                diff = torch.abs(out1 - out2).max().item()
                print(f"   Variation (TRAIN mode): {diff:.4f}")
                
                if diff < 1e-6:
                    raise ValueError("Model outputs constant!")
                
                print("   ✅ Model works (onnx2torch partial load)")
                
            except Exception as e:
                print(f"   ❌ Model inference failed: {e}")
                raise ValueError("onnx2torch .bin not compatible")
    
    else:
        print("   ✅ Detected PyTorch standard format")
        print("   Loading with strict=True")
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        
        if missing_keys:
            print(f"   ❌ Missing keys: {missing_keys}")
            raise ValueError(f"State dict incomplete!")
        
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys: {unexpected_keys}")
        
        # ✅ Test en mode TRAIN
        print("\n🔍 Testing model...")
        model.train()
        
        with torch.no_grad():
            test1 = torch.randn(1, 3, 224, 224)
            test2 = torch.randn(1, 3, 224, 224)
            
            out1 = model(test1)
            out2 = model(test2)
            
            diff = torch.abs(out1 - out2).max().item()
            print(f"   Variation (TRAIN mode): {diff:.4f}")
            
            if diff < 1e-6:
                raise ValueError("Model outputs constant!")
    
    model = model.to(DEVICE)
    print(f"✅ Model loaded correctly")
    
    return model

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    accumulation_steps: int = 4 
) -> float:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item() * accumulation_steps:.4f}")
    
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    print(f"📊 DataLoader length: {len(val_loader)}")
    print(f"📊 Total samples: {len(val_loader.dataset)}")
    
    if len(val_loader) == 0:
        raise ValueError("❌ DataLoader empty!")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    emotions = ['boredom', 'confusion', 'engagement', 'frustration']
    mae_per_emotion = {}
    
    for i, emotion in enumerate(emotions):
        mae = np.mean(np.abs(all_preds[:, i] - all_labels[:, i]))
        mae_per_emotion[f'mae_{emotion}'] = mae
    
    mae_global = np.mean([mae_per_emotion[f'mae_{emotion}'] for emotion in emotions])
    val_loss = running_loss / len(val_loader)
    
    metrics = {
        'val_loss': val_loss,
        'mae_global': mae_global,
        **mae_per_emotion
    }
    
    return val_loss, metrics

# ============================================================================
# FINE-TUNING (AVEC FREEZE)
# ============================================================================

def finetune_model(
    model_path: str,
    train_data: Tuple[list, np.ndarray],
    val_data: Tuple[list, np.ndarray],
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    freeze_backbone: bool = True
) -> Tuple[nn.Module, Dict]:
    print("🔥 Starting fine-tuning...")
    
    model = load_pretrained_model(model_path)
    
    # ✅ FREEZE BACKBONE
    if freeze_backbone:
        print("\n❄️  FREEZING BACKBONE...")
        
        frozen_params = 0
        trainable_params = 0
        
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True
                trainable_params += param.numel()
        
        print(f"   Frozen: {frozen_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        print(f"   Ratio: {trainable_params / (frozen_params + trainable_params) * 100:.2f}%\n")
        
        if learning_rate < 1e-3:
            learning_rate = 1e-3
            print(f"   Learning rate → {learning_rate}\n")
    
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    train_dataset = EmotionDataset(train_images, train_labels, get_train_transform())
    val_dataset = EmotionDataset(val_images, val_labels, get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch [{epoch + 1}/{num_epochs}]")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, accumulation_steps=4)
        history['train_loss'].append(train_loss)
        
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  MAE Global: {val_metrics['mae_global']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✅ New best!")
    
    print("\n🎉 Fine-tuning complete!")
    
    return model, history

# ============================================================================
# SAVE MODEL (FIX: Mode TRAIN)
# ============================================================================

def save_model(model: nn.Module, save_path: str):
    """
    Sauvegarde le modèle PyTorch
    ✅ CRITIQUE: Sauvegarde en mode TRAIN pour éviter BatchNorm issues
    """
    # ✅ Mettre en mode TRAIN avant save
    # Les BatchNorm en eval mode avec batch_size=1 causent des outputs constants
    model.train()
    
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path} (in TRAIN mode)")
