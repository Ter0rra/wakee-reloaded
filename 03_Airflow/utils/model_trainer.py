"""
Model Trainer - PyTorch Fine-tuning
Fine-tune EfficientNet B4 avec nouvelles annotations
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
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_CLASSES = 4  # boredom, confusion, engagement, frustration

# ============================================================================
# DATASET
# ============================================================================

class EmotionDataset(Dataset):
    """Dataset pour les émotions DAiSEE"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): Liste des chemins d'images
            labels (np.array): Labels (N, 4) pour 4 émotions
            transform: Transformations torchvision
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Transform
        if self.transform:
            image = self.transform(image)
        
        # Labels
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_train_transform():
    """Data augmentation pour training"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transform():
    """Transforms pour validation/test"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_pretrained_model(model_path: str) -> nn.Module:
    """
    Charge le modèle .bin depuis HuggingFace
    
    Args:
        model_path (str): Chemin vers model.bin
    
    Returns:
        nn.Module: Modèle PyTorch chargé
    """
    print(f"📦 Loading model from {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Si le state_dict contient 'model', extraire
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Crée le modèle (architecture EfficientNet B4)
    from torchvision import models
    model = models.efficientnet_b4(weights=None)
    
    # Modifie la dernière couche pour 4 sorties
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    # Charge les poids
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(DEVICE)
    print(f"✅ Model loaded on {DEVICE}")
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    accumulation_steps: int = 4 
) -> float:
    """Entraîne le modèle sur une époque"""
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward
        # optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Normalise la loss par accumulation_steps
        loss = loss / accumulation_steps        

        # Backward
        loss.backward()
    #     optimizer.step()
        
    #     running_loss += loss.item()
        
    #     # Log progress
    #     if (batch_idx + 1) % 10 == 0:
    #         print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
    
    # epoch_loss = running_loss / len(train_loader)
    # return epoch_loss
        # ⭐ Update seulement tous les N steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item() * accumulation_steps:.4f}")
    
    # ⭐ Dernier update si nécessaire
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, Dict[str, float]]:
    """Évalue le modèle sur validation set"""
    model.eval()
    running_loss = 0.0
    
    all_preds = []
    all_labels = []
    
        # ✅ DEBUG : Vérifie le DataLoader
    print(f"📊 DataLoader length: {len(val_loader)}")
    print(f"📊 Total samples: {len(val_loader.dataset)}")
    
    if len(val_loader) == 0:
        raise ValueError("❌ DataLoader is empty! No data to evaluate.")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calcule MAE par émotion
    emotions = ['boredom', 'confusion', 'engagement', 'frustration']
    mae_per_emotion = {}
    
    for i, emotion in enumerate(emotions):
        mae = np.mean(np.abs(all_preds[:, i] - all_labels[:, i]))
        mae_per_emotion[f'mae_{emotion}'] = mae
    
    # MAE global
    mae_global = np.mean([mae_per_emotion[f'mae_{emotion}'] for emotion in emotions])
    
    val_loss = running_loss / len(val_loader)
    
    metrics = {
        'val_loss': val_loss,
        'mae_global': mae_global,
        **mae_per_emotion
    }
    
    return val_loss, metrics

# ============================================================================
# FINE-TUNING COMPLET
# ============================================================================

def finetune_model(
    model_path: str,
    train_data: Tuple[list, np.ndarray],
    val_data: Tuple[list, np.ndarray],
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE
) -> Tuple[nn.Module, Dict]:
    """
    Fine-tune le modèle avec nouvelles données
    
    Args:
        model_path (str): Chemin vers model.bin
        train_data (tuple): (image_paths, labels)
        val_data (tuple): (image_paths, labels)
        num_epochs (int): Nombre d'époques
        learning_rate (float): Learning rate
        batch_size (int): Batch size
    
    Returns:
        tuple: (model fine-tuné, historique training)
    """
    print("🔥 Starting fine-tuning...")
    
    # Load model
    model = load_pretrained_model(model_path)
    
    # Créer datasets
    train_images, train_labels = train_data
    val_images, val_labels = val_data
    
    train_dataset = EmotionDataset(train_images, train_labels, get_train_transform())
    val_dataset = EmotionDataset(val_images, val_labels, get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Val samples: {len(val_dataset)}")
    
    # Loss et optimizer
    criterion = nn.MSELoss()  # Régression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch [{epoch + 1}/{num_epochs}]")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, accumulation_steps=4)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  MAE Global: {val_metrics['mae_global']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✅ New best model! (val_loss: {val_loss:.4f})")
    
    print("\n🎉 Fine-tuning complete!")
    
    return model, history

# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model: nn.Module, save_path: str):
    """Sauvegarde le modèle PyTorch"""
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")
