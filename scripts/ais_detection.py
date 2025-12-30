#!/usr/bin/env python3
"""
STEP 5: AIS Correlation & Dark Vessel Classification (SIMPLIFIED)

Consumes:
- ship_detection_results_{satellite}.json

Produces:
- ship_detection_with_ais_status_{satellite}.json

Usage:
    python ais_detection.py --satellite-path <path_to_satellite_tiff>
"""

import pandas as pd
import json
import math
import os
import sys
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------
# CONFIG (NO HARDCODING)
# ---------------------------------------------------------
AIS_FOLDER = "data/ais_data/"

AIS_SEARCH_BUFFER_DEG = 1.0
TIME_THRESHOLD_MIN = 60
DIST_THRESHOLD_KM = 2
MAX_AIS_ROWS_PER_FILE = 2000

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def parse_time(t):
    if t.endswith("Z"):
        return datetime.fromisoformat(t.replace("Z", "+00:00"))
    return datetime.fromisoformat(t)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def extract_satellite_name(satellite_file_path: str) -> str:
    """Extract satellite name from file path"""
    try:
        # Extract satellite folder name (e.g., 'sat1', 'sat2', etc.)
        path_parts = satellite_file_path.replace('\\', '/').split('/')
        for i, part in enumerate(path_parts):
            if part.startswith('sat') and part[3:].isdigit():
                return part
        
        # Fallback: use parent directory name
        return Path(satellite_file_path).parent.parent.name
    except:
        # Final fallback: use timestamp
        return f"sat_{datetime.now().strftime('%H%M%S')}"

def main():
    """Main execution function with command line argument support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AIS Correlation & Dark Vessel Classification')
    parser.add_argument('--satellite-path', required=True, 
                       help='Path to satellite TIFF file')
    
    args = parser.parse_args()
    
    # Extract satellite name from path
    satellite_name = extract_satellite_name(args.satellite_path)
    
    print(f"🛰️ AIS DETECTION FOR {satellite_name.upper()}")
    print("=" * 60)
    
    # Define file paths for this satellite
    RADAR_FILE = f"output/json/ship_detection_results_{satellite_name}.json"
    OUTPUT_FILE = f"output/json/ship_detection_with_ais_status_{satellite_name}.json"
    
    # ---------------------------------------------------------
    # LOAD RADAR DATA FOR SPECIFIC SATELLITE
    # ---------------------------------------------------------
    print(f"Loading radar detections for {satellite_name}...")

    if not os.path.exists(RADAR_FILE):
        print(f"❌ Missing radar detection file: {RADAR_FILE}")
        sys.exit(1)

    with open(RADAR_FILE, "r") as f:
        radar_json = json.load(f)

    vessels = radar_json.get("vessel_candidates", [])
    if not vessels:
        print(f"⚠️ No vessel detections found for {satellite_name}")
        # Create empty output file
        empty_output = {
            "satellite_name": satellite_name,
            "summary": {
                "total_radar_detections": 0,
                "normal_vessels": 0,
                "dark_vessels": 0,
                "dark_vessel_percentage": 0,
                "parameters": {
                    "distance_threshold_km": DIST_THRESHOLD_KM,
                    "time_threshold_minutes": TIME_THRESHOLD_MIN,
                    "ais_search_buffer_deg": AIS_SEARCH_BUFFER_DEG
                }
            },
            "vessel_candidates": []
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, "w") as f:
            json.dump(empty_output, f, indent=2)
        
        print(f"✅ Empty AIS classification saved for {satellite_name}")
        print(f"📁 Output → {OUTPUT_FILE}")
        return

    lats = [v["latitude"] for v in vessels if v.get("latitude")]
    lons = [v["longitude"] for v in vessels if v.get("longitude")]

    if not lats or not lons:
        print(f"❌ No valid coordinates found in vessel detections for {satellite_name}")
        sys.exit(1)

    radar_bounds = {
        "min_lat": min(lats) - AIS_SEARCH_BUFFER_DEG,
        "max_lat": max(lats) + AIS_SEARCH_BUFFER_DEG,
        "min_lon": min(lons) - AIS_SEARCH_BUFFER_DEG,
        "max_lon": max(lons) + AIS_SEARCH_BUFFER_DEG
    }

    radar_time = parse_time(
        radar_json.get("processing_timestamp", datetime.utcnow().isoformat())
    )

    print(f"✅ Loaded {len(vessels)} radar vessels for {satellite_name}")

    # ---------------------------------------------------------
    # LOAD AIS DATA (IN-MEMORY ONLY)
    # ---------------------------------------------------------
    ais_points = []

    if os.path.exists(AIS_FOLDER):
        csv_files = glob.glob(os.path.join(AIS_FOLDER, "*.csv"))

        for file in csv_files:
            try:
                df = pd.read_csv(file, nrows=MAX_AIS_ROWS_PER_FILE)
                required = {"date", "cell_ll_lat", "cell_ll_lon", "mmsi"}
                if not required.issubset(df.columns):
                    continue

                for _, row in df.iterrows():
                    lat = float(row["cell_ll_lat"])
                    lon = float(row["cell_ll_lon"])

                    if not (
                        radar_bounds["min_lat"] <= lat <= radar_bounds["max_lat"]
                        and radar_bounds["min_lon"] <= lon <= radar_bounds["max_lon"]
                    ):
                        continue

                    ais_points.append({
                        "lat": lat,
                        "lon": lon,
                        "time": parse_time(str(row["date"]))
                    })

            except Exception:
                continue

    print(f"📡 AIS points loaded for correlation: {len(ais_points)}")

    # ---------------------------------------------------------
    # AIS CORRELATION (BOOLEAN ONLY)
    # ---------------------------------------------------------
    normal_count = 0
    dark_count = 0

    for vessel in vessels:
        vessel["ais_status"] = "DARK"

        for ais in ais_points:
            dist = haversine_km(
                vessel["latitude"], vessel["longitude"],
                ais["lat"], ais["lon"]
            )
            time_diff = abs((radar_time - ais["time"]).total_seconds()) / 60

            if dist <= DIST_THRESHOLD_KM and time_diff <= TIME_THRESHOLD_MIN:
                vessel["ais_status"] = "NORMAL"
                break

        if vessel["ais_status"] == "NORMAL":
            normal_count += 1
        else:
            dark_count += 1

    # ---------------------------------------------------------
    # FINAL SUMMARY FOR SPECIFIC SATELLITE
    # ---------------------------------------------------------
    final_output = {
        "satellite_name": satellite_name,
        "summary": {
            "total_radar_detections": len(vessels),
            "normal_vessels": normal_count,
            "dark_vessels": dark_count,
            "dark_vessel_percentage": round((dark_count / len(vessels)) * 100, 2),
            "parameters": {
                "distance_threshold_km": DIST_THRESHOLD_KM,
                "time_threshold_minutes": TIME_THRESHOLD_MIN,
                "ais_search_buffer_deg": AIS_SEARCH_BUFFER_DEG
            }
        },
        "vessel_candidates": vessels
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n✅ AIS classification complete for {satellite_name.upper()}")
    print(f"📁 Output → {OUTPUT_FILE}")
    print(f"🚨 Dark vessels: {dark_count}/{len(vessels)}")

if __name__ == "__main__":
    main()
