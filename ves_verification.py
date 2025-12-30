import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sar_preprocessing import create_standard_preprocessor

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "sar_cnn_model.pth"       # Your trained model
INPUT_JSON = "candidates.json"         # The output from Step 6 (AIS Cross-check)
OUTPUT_JSON = "verified_output.json"   # The final file with CNN scores added
IMAGE_FOLDER = "data/images"           # Where the 'ship_1.png' files are stored
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.50            # Threshold to say "YES"

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. STANDARDIZED PREPROCESSING & MODEL
# ==========================================
# Use standardized preprocessor for consistency with training
preprocessor = create_standard_preprocessor(img_size=IMG_SIZE)

# Enhanced CNN Model (matches train_cnn.py)
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

# Legacy SimpleSARCNN for backward compatibility
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
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

_model = None

def load_model():
    """Load the trained model with automatic architecture detection"""
    global _model
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        
        # Try to load enhanced model first, fallback to simple model
        try:
            _model = EnhancedSARCNN().to(device)
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ Loaded EnhancedSARCNN model")
        except Exception as e:
            print(f"⚠️ Failed to load EnhancedSARCNN, trying SimpleSARCNN: {e}")
            try:
                _model = SimpleSARCNN().to(device)
                _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print("✅ Loaded SimpleSARCNN model")
            except Exception as e2:
                print(f"❌ Error loading model: {e2}")
                _model = None
                return None
        
        _model.eval()
    
    return _model

def calculate_vessel_size(image):
    """
    Estimates vessel size by counting significant pixels.
    Assumes image is grayscale.
    Returns: approximate pixel count of the object.
    """
    # Simple thresholding to find the object
    # Convert to numpy array for easier counting
    import numpy as np
    img_array = np.array(image)
    # Assuming dark ocean, bright ship. Threshold at 50 (out of 255)
    # Adjust based on your data characteristics
    significant_pixels = np.sum(img_array > 50) 
    return int(significant_pixels)

def verify_snapshot(snapshot_data):
    """
    Verifies a single snapshot using standardized preprocessing.
    Args:
        snapshot_data (dict): {'image': 'filename.png', ...}
    Returns:
        dict: The updated snapshot_data with verification results and sizes.
    """
    model = load_model()
    if model is None:
        snapshot_data['is_vessel_verified'] = "ERROR_MODEL_MISSING"
        return snapshot_data

    img_filename = snapshot_data.get('image')
    # If full path provided, use it, else assume relative to IMAGE_FOLDER
    if os.path.dirname(img_filename):
        img_path = img_filename
    else:
        img_path = os.path.join(IMAGE_FOLDER, img_filename)
        
    cnn_score = 0.0
    verification_status = "ERROR_IMG_MISSING"
    vessel_size = 0

    if os.path.exists(img_path):
        try:
            # Use standardized preprocessing for consistency
            input_tensor = preprocessor.preprocess_image(img_path, training=False)
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
            
            # Load image for size calculation
            image = Image.open(img_path).convert('L')
            vessel_size = calculate_vessel_size(image)
            
            # Inference with consistent preprocessing
            with torch.no_grad():
                logits = model(input_tensor)
                cnn_score = torch.sigmoid(logits).item()
            
            if cnn_score >= CONFIDENCE_THRESHOLD:
                verification_status = "YES"
            else:
                verification_status = "NO"
                
        except Exception as e:
            print(f"⚠️ Error processing {img_filename}: {e}")
            verification_status = "ERROR_PROCESSING"
    else:
        print(f"⚠️ Warning: Image {img_filename} not found at {img_path}")

    # Update Data
    snapshot_data['cnn_confidence'] = round(cnn_score, 4)
    snapshot_data['is_vessel_verified'] = verification_status
    snapshot_data['vessel_size_pixels'] = vessel_size
    
    return snapshot_data

# ==========================================
# 4. MAIN VERIFICATION LOOP
# ==========================================
def verify_vessels():
    print(f"🚀 Starting CNN Verification...")
    
    # 1. Load Input Data
    if not os.path.exists(INPUT_JSON):
        print(f"❌ Error: Input file '{INPUT_JSON}' not found.")
        return

    with open(INPUT_JSON, 'r') as f:
        candidate_list = json.load(f)
    
    print(f"Loaded {len(candidate_list)} candidates from {INPUT_JSON}")

    # 2. Iterate and Verify
    for candidate in candidate_list:
        verify_snapshot(candidate)
        print(f"Checked {candidate.get('image')}: Score {candidate.get('cnn_confidence', 0):.2f} -> {candidate.get('is_vessel_verified')}")

    # 3. Save Updated Data
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(candidate_list, f, indent=4)
    
    print(f"\n✅ Verification Complete. Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    verify_vessels()