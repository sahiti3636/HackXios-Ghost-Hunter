import os
import json
from ves_verification import verify_snapshot
from risk_fusion import calculate_risk_score

def check_single_image(image_path):
    print(f"🕵️‍♀️ Inspecting: {image_path}")
    
    if not os.path.exists(image_path):
        print("❌ File not found.")
        return

    # Mock candidate data (since we don't have geospatial info from a PNG)
    candidate = {
        "image": image_path, # Full path
        "location_data": {
            "latitude": 0.0,
            "longitude": 0.0,
            "mpa_name": "Unknown (Manual Upload)" 
        },
        "ais_verification": {
            "status": "Unknown (Manual Check)"
        }
    }

    # 1. Verify
    print("running verification...")
    result = verify_snapshot(candidate)
    
    # 2. Risk Score (Limited, since no AIS/MPA context)
    # We will manually inject a "Dark" status just to see how the code handles it 
    # if the user *intended* it to be a dark ship, but for now let's keep it neutral.
    scored = calculate_risk_score(result)
    
    print("\n--- ANALYSIS REPORT ---")
    print(f"File: {os.path.basename(image_path)}")
    print(f"Is Vessel Verified? : {scored['is_vessel_verified']}")
    print(f"Confidence Score    : {scored.get('cnn_confidence', 0):.4f}")
    print(f"Est. Size (pixels)  : {scored.get('vessel_size_pixels', 0)}")
    print(f"Risk Score          : {scored.get('risk_score', 0)}")
    print(f"Risk Breakdown      : {scored.get('risk_breakdown', [])}")

if __name__ == "__main__":
    check_single_image("data/raw/satellite/quick-look.png")
