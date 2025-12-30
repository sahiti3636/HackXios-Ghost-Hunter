import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageStat
from datetime import datetime
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "sar_cnn_model.pth"
INPUT_FOLDER = "data/test_images"
OUTPUT_JSON = "vessel_report.json"
IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.75

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PIPELINE DATA PLACEHOLDER (THE UPDATE)
# ==========================================
# In your full system, this dictionary comes from Step 1 (Geofencing) & Step 6 (AIS).
# The keys match the filenames in your 'data/test_images' folder.

##########fill in our own json path  

PIPELINE_METADATA = {
    "ship_1.png": {
        "mpa_name": "Galapagos Marine Reserve",
        "ais_status": "OFFLINE (Dark)",
        "lat_approx": 0.523,
        "lon_approx": -90.50
    },
    "ship_2.png": {
        "mpa_name": "Great Barrier Reef",
        "ais_status": "ACTIVE (MMSI: 123456789)",
        "lat_approx": -18.287,
        "lon_approx": 147.69
    }
    # If a file isn't listed here, the code uses the "FALLBACK" values below.
}

FALLBACK_MPA = "PENDING_GEOFENCE_CHECK"  # Placeholder
FALLBACK_AIS = "PENDING_AIS_CROSSCHECK"  # Placeholder

# ==========================================
# 3. MODEL ARCHITECTURE (Must match training)
# ==========================================
class SimpleSARCNN(nn.Module):
    def __init__(self):
        super(SimpleSARCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# ==========================================
# 4. HELPER: FETCH METADATA
# ==========================================
def get_image_metadata(filename):
    """
    Looks up pipeline info for a specific image.
    If missing, returns placeholder values.
    """
    data = PIPELINE_METADATA.get(filename, {})
    
    # Use provided data OR fallback to placeholder
    mpa = data.get("mpa_name", FALLBACK_MPA)
    ais = data.get("ais_status", FALLBACK_AIS)
    
    # Mock coordinates if not provided (for demo purposes)
    lat = data.get("lat_approx", 7.5 + random.uniform(-0.1, 0.1))
    lon = data.get("lon_approx", 95.0 + random.uniform(-0.1, 0.1))
    
    return mpa, ais, lat, lon

def analyze_brightness(image_path):
    img = Image.open(image_path).convert('L')
    stat = ImageStat.Stat(img)
    return stat.mean[0]

# ==========================================
# 5. MAIN INFERENCE LOOP
# ==========================================
def run_inference():
    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleSARCNN().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("❌ Error: Model file not found.")
        return

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    results = []
    
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ Created {INPUT_FOLDER}. Put images here.")
        return

    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.png', '.jpg', '.tif'))]
    print(f"Found {len(image_files)} images to analyze.")

    for img_file in image_files:
        img_path = os.path.join(INPUT_FOLDER, img_file)
        
        # 1. Prediction
        image = Image.open(img_path).convert('L')
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            score = model(input_tensor).item()

        is_vessel = score > CONFIDENCE_THRESHOLD
        status = "CONFIRMED VESSEL" if is_vessel else "NOISE/CLUTTER"

        # 2. Retrieve Pipeline Metadata (Dynamic Step)
        mpa_name, ais_status, lat, lon = get_image_metadata(img_file)

        # 3. Calculate Risk (Simple Logic)
        # If Vessel + Dark (AIS Offline) + Inside MPA = HIGH RISK
        is_dark = "OFFLINE" in ais_status or "Dark" in ais_status
        is_inside_mpa = "PENDING" not in mpa_name
        
        risk_flag = False
        if is_vessel and is_dark and is_inside_mpa:
            risk_flag = True

        # 4. Build JSON
        record = {
            "image_id": img_file,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "detection_data": {
                "cnn_confidence": round(score, 4),
                "classification": status,
                "risk_assessment": "CRITICAL" if risk_flag else "LOW"
            },
            "location_data": {
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "mpa_name": mpa_name  # <--- Now Dynamic
            },
            "radar_metrics": {
                "mean_intensity": round(analyze_brightness(img_path), 2),
            },
            "ais_verification": {
                "status": ais_status  # <--- Now Dynamic
            }
        }
        
        results.append(record)
        print(f"Analyzed {img_file}: {status} | AIS: {ais_status} | MPA: {mpa_name}")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump({"scan_summary": results}, f, indent=4)
    
    print(f"\n✅ Report saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    run_inference()