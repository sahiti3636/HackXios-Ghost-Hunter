import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sar_preprocessing import create_standard_preprocessor

# ==========================================
# 1. CONFIG
# ==========================================
IMAGE_FOLDER = "cnn_dataset/images"
JSON_FILE = "cnn_dataset/labels.json"

IMG_SIZE = 64          # Optimized for SAR vessel detection
BATCH_SIZE = 16        # Balanced for memory and gradient stability
LEARNING_RATE = 1e-4   # Lower learning rate for better convergence
EPOCHS = 50            # More epochs with early stopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. STANDARDIZED PREPROCESSING
# ==========================================
# Use standardized preprocessor for consistency
preprocessor = create_standard_preprocessor(img_size=IMG_SIZE)

# ==========================================
# 3. DATASET
# ==========================================
class SARShipDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.root_dir = root_dir
        self.transform = transform

        labels = [x["label"] for x in self.data]
        print(f"\nDataset Loaded:")
        print(f"  Total: {len(labels)}")
        print(f"  Ships (1): {labels.count(1)}")
        print(f"  Noise (0): {labels.count(0)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        img_path = os.path.join(self.root_dir, record["image"])

        image = Image.open(img_path).convert("L")
        label = torch.tensor(record["label"], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 4. STANDARDIZED TRANSFORMS
# ==========================================
# Use standardized preprocessing for training
data_transforms = preprocessor.get_training_transform()

# ==========================================
# 5. ENHANCED CNN MODEL WITH BATCH NORM
# ==========================================
class EnhancedSARCNN(nn.Module):
    def __init__(self, dropout_rate=0.3, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # Feature extraction layers optimized for 64x64 SAR input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batch_norm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2)  # 64x64 -> 32x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2)  # 32x32 -> 16x16
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batch_norm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2)  # 16x16 -> 8x8
        
        # Additional conv layer for better feature extraction
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256) if use_batch_norm else nn.Identity()
        self.pool4 = nn.MaxPool2d(2)  # 8x8 -> 4x4
        
        # Classifier with improved dropout and layer sizes
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 4096 -> 512
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)  # Reduced dropout for deeper layers
        self.fc3 = nn.Linear(128, 1)  # Output layer
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Feature extraction with batch normalization
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        # Classification
        x = self.flatten(x)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)  # Raw logits
        
        return x

# Keep the old model for backward compatibility
class SimpleSARCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # RAW LOGITS
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 6. ENHANCED TRAINING MANAGER
# ==========================================
class TrainingManager:
    def __init__(self, model, device, patience=10, min_delta=0.001):
        self.model = model
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_model_state = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
    
    def setup_data_loaders(self, dataset, batch_size=16, train_split=0.8):
        """Setup training and validation data loaders"""
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_with_early_stopping(self, train_loader, val_loader, epochs=50, 
                                 learning_rate=1e-4, pos_weight=None):
        """Train model with early stopping and learning rate scheduling"""
        
        # Setup loss function with class imbalance handling
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        print(f"\n🚀 Enhanced Training Started")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Early Stopping Patience: {self.patience}")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_acc, val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")
            
            # Early stopping check
            if self._check_early_stopping(val_loss, val_acc):
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Restored best model (Val Acc: {self.best_val_acc:.2f}%)")
        
        return self.training_history
    
    def _train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for imgs, labels in train_loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = self.model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                logits = self.model(imgs)
                loss = criterion(logits, labels)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                total_loss += loss.item()
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100 * correct / total if total > 0 else 0
        
        # Additional metrics
        metrics = {
            'confidence_scores': all_probs,
            'true_labels': all_labels
        }
        
        return val_loss, val_acc, metrics
    
    def _check_early_stopping(self, val_loss, val_acc):
        """Check if early stopping should be triggered"""
        # Check if validation loss improved
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_model_state = self.model.state_dict().copy()
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience
    
    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(imgs)
                probs = torch.sigmoid(logits).squeeze()
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels)
        
        # Confidence analysis
        vessel_confidences = all_probs[all_labels == 1]
        sea_confidences = all_probs[all_labels == 0]
        
        metrics = {
            'accuracy': accuracy,
            'vessel_avg_confidence': np.mean(vessel_confidences) if len(vessel_confidences) > 0 else 0,
            'sea_avg_confidence': np.mean(sea_confidences) if len(sea_confidences) > 0 else 0,
            'vessel_high_confidence_ratio': np.mean(vessel_confidences > 0.7) if len(vessel_confidences) > 0 else 0
        }
        
        return metrics

def train():
    dataset = SARShipDataset(JSON_FILE, IMAGE_FOLDER, data_transforms)

    # Use enhanced model
    model = EnhancedSARCNN(dropout_rate=0.3, use_batch_norm=True).to(device)
    
    # Setup training manager
    trainer = TrainingManager(model, device, patience=10)
    train_loader, val_loader = trainer.setup_data_loaders(dataset, batch_size=BATCH_SIZE)

    # Class imbalance handling
    labels = [x["label"] for x in dataset.data]
    pos_weight = torch.tensor([labels.count(0) / labels.count(1)]).to(device)
    
    # Train with enhanced features
    history = trainer.train_with_early_stopping(
        train_loader, val_loader, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE,
        pos_weight=pos_weight
    )
    
    # Final evaluation
    final_metrics = trainer.evaluate_model(val_loader)
    print(f"\n📊 Final Model Performance:")
    print(f"   Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"   Vessel Avg Confidence: {final_metrics['vessel_avg_confidence']:.3f}")
    print(f"   Vessels >0.7 Confidence: {final_metrics['vessel_high_confidence_ratio']:.3f}")

    torch.save(model.state_dict(), "sar_cnn_model.pth")
    print("\n✅ Enhanced model saved as sar_cnn_model.pth")

# ==========================================
# 7. RUN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(IMAGE_FOLDER) or not os.path.exists(JSON_FILE):
        print("❌ Dataset not found.")
    else:
        train()
