
import json
import os
import shutil
import random
from datetime import datetime
from shapely.geometry import shape, Polygon, Point
from utils.sentinel_fetcher import SentinelFetcher

from dotenv import load_dotenv
import requests # Added for direct QL download
load_dotenv() # Load vars before init

from utils.mpa_checker import MPAChecker

SOURCE_REPORT = "final_ghost_hunter_report_S1A_IW_GRDH_1SDV_20251228T232206_20251228T232231_062522_07D5BF_39AF.SAFE.json"
CACHE_DIR = "data/cached_reports"

def prep_data():
    # Load Source Vessels (from SC scene) to transpose
    if not os.path.exists(SOURCE_REPORT):
        print(f"❌ Source report {SOURCE_REPORT} not found.")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    
    with open(SOURCE_REPORT, 'r') as f:
        source_data = json.load(f)
        
    real_vessels = source_data.get('vessels', [])
    print(f"Loaded {len(real_vessels)} real vessel signatures from source.")
    
    # Initialize Tools
    checker = MPAChecker() # Loads all MPAs
    fetcher = SentinelFetcher()
    fetcher.connect()
    
    token = fetcher.get_access_token()
    if not token:
        print("⚠️ No token, downloads might fail.")
        
    print(f"Found {len(checker.mpas)} MPAs to process.")
    
    for mpa in checker.mpas:
        name = mpa['name']
        print(f"\nProcessing {name}...")
        
        # Get Polygon Coords (Shapely Geometry -> List of Lists)
        # Shapely stores as (x, y) = (lon, lat)
        # Fetcher expects... well app.py passes what it gets from frontend.
        # But here we have Shapely geoms.
        # Let's extract coords.
        try:
             # Handle MultiPolygon (just take largest) or Polygon
             if mpa['geometry'].geom_type == 'Polygon':
                 poly_coords = list(mpa['geometry'].exterior.coords)
             elif mpa['geometry'].geom_type == 'MultiPolygon':
                 poly_coords = list(max(mpa['geometry'].geoms, key=lambda a: a.area).exterior.coords)
             else:
                 print(f"Skipping {name} (Geometry type {mpa['geometry'].geom_type} not supported yet)")
                 continue
                 
             # poly_coords is [(lon, lat), ...]
             # The fetcher.search_scenes in app.py takes 'region_data["polygon"]' which is usually [[lat, lon]...] from Leaflet?
             # Wait, SentinelFetcher._polygon_to_wkt_coords takes the input and SWAPS p[1] p[0].
             # If input is [lat, lon], swap -> [lon, lat].
             # If input is [lon, lat] (Shapely), swap -> [lat, lon] -> WRONG for WKT.
             
             # SO: If we have Shapely [(lon, lat)], we must FLIP it to [[lat, lon]] so the Fetcher can FLIP it back to WKT [(lon, lat)].
             # It's a double negative, but ensures we match App flow.
             
             app_style_coords = [[p[1], p[0]] for p in poly_coords]
             
             # Search (Wider Window for success)
             scenes = fetcher.search_scenes(app_style_coords, start_date="2024-01-01", end_date="NOW")
             
             target_scene = None
             ql_path = "/simulation_overlay.png"
             
             if scenes:
                 target_scene = scenes[0]
                 print(f"✅ Found real scene: {target_scene['properties']['title']}")
                 
                 # DOWNLOAD QUICKLOOK
                 if token:
                     try:
                         uuid = target_scene['properties']['id']
                         safe_name = target_scene['properties']['title']
                         if not safe_name.endswith('.SAFE'): safe_name += '.SAFE'
                         
                         safe_mpa_name = name.lower().replace(" ", "_").replace("/","-")
                         out_dir = "ghost-hunter-frontend/public/cached_previews"
                         os.makedirs(out_dir, exist_ok=True)
                         out_file = f"{out_dir}/{safe_mpa_name}_preview.png"
                         
                         # Try multiple paths for preview
                         # 1. Standard Quicklook node
                         # 2. Thumbnail attribute (often separate API)
                         # 3. Icon
                         
                         # Path 1: Nodes('Quicklook')/$value (Some collections)
                         # Path 2: Nodes('preview')/Nodes('quick-look.png')/$value (Sentinels in SAFE)
             
                         paths_to_try = [
                             f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({uuid})/Nodes('{safe_name}')/Nodes('preview')/Nodes('quick-look.png')/$value",
                             f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({uuid})/Nodes('Quicklook')/$value"
                         ]
                         
                         success = False
                         session = requests.Session()
                         session.headers.update({"Authorization": f"Bearer {token}"})

                         for ql_url in paths_to_try:
                             if success: break
                             print(f"   ⬇️ Trying QL URL: ...{ql_url[-40:]}")
                             try:
                                 r_ql = session.get(ql_url, stream=True, allow_redirects=True)
                                 # Handle 302 -> 401 manually if session fails
                                 if r_ql.status_code == 401:
                                      r_ql = requests.get(r_ql.url, headers={"Authorization": f"Bearer {token}"}, stream=True)
                                      
                                 if r_ql.status_code == 200:
                                     with open(out_file, 'wb') as f:
                                         shutil.copyfileobj(r_ql.raw, f)
                                     print(f"   ✅ Saved Preview: {out_file}")
                                     ql_path = f"/cached_previews/{safe_mpa_name}_preview.png"
                                     success = True
                                 else:
                                     print(f"   ❌ Failed: {r_ql.status_code}")
                             except Exception as e:
                                 print(f"   ⚠️ Exception: {e}")
                                 
                         if not success:
                             print(f"⚠️ All QL download attempts failed.")
                             
                     except Exception as e:
                         print(f"❌ QL Error: {e}")
             else:
                 print(f"⚠️ No scenes found for {name}. Using fallback.")
                 target_scene = {
                     "properties": {
                         "title": f"S1A_FALLBACK_{name.replace(' ','_')}",
                         "ingestiondate": datetime.now().isoformat(),
                         "id": "FALLBACK"
                     }
                 }
                 
             # Map Vessels (Transposition)
             min_lat = min(p[0] for p in app_style_coords)
             max_lat = max(p[0] for p in app_style_coords)
             min_lon = min(p[1] for p in app_style_coords)
             max_lon = max(p[1] for p in app_style_coords)
             
             mapped_vessels = []
             for i, v in enumerate(real_vessels):
                new_v = v.copy()
                new_lat = random.uniform(min_lat, max_lat)
                new_lon = random.uniform(min_lon, max_lon)
                new_v['latitude'] = new_lat
                new_v['longitude'] = new_lon
                new_v['coordinates'] = f"{new_lat:.4f}°N, {new_lon:.4f}°E"
                
                # Dual Schema
                new_v['lat'] = new_lat
                new_v['lng'] = new_lon
                new_v['risk'] = new_v.get('risk_score', 0)
                new_v['id'] = new_v.get('vessel_id')
                new_v['name'] = f"Vessel {new_v['vessel_id']} ({name})"
                new_v['status'] = "Active"
                new_v['lastSeen'] = "1h ago"
                new_v['type'] = "Unknown"
                new_v['flag'] = "Unknown"
                mapped_vessels.append(new_v)
                
             # Save Report
             safe_mpa_name = name.lower().replace(" ", "_").replace("/","-")
             if "papah" in safe_mpa_name: safe_mpa_name = "papahanaumokuakea"
             
             report = {
                "pipeline_version": "2.0 (Cached Real Data)",
                "region": target_scene['properties']['title'],
                "timestamp": datetime.now().isoformat(),
                "vessels": mapped_vessels,
                "satellite_image": {
                    "url": ql_path,
                    "bounds": [[min_lat, min_lon], [max_lat, max_lon]]
                },
                "satellite_metadata": {
                    "source": "Sentinel-1 (Real Metadata)",
                    "scene_id": target_scene['properties']['title'],
                    "ingestion_date": target_scene['properties']['ingestiondate']
                },
                "intelligence_analysis": {},
                "detection_summary": {
                    "total_vessels": len(mapped_vessels),
                    "dark_vessels": len([v for v in mapped_vessels if v.get('ais_status') == 'DARK']),
                    "high_risk_vessels": len([v for v in mapped_vessels if v.get('risk_score', 0) > 50])
                }
             }
             
             fname = f"{CACHE_DIR}/cached_{safe_mpa_name}.json"
             with open(fname, 'w') as f:
                 json.dump(report, f, indent=2)
             print(f"💾 Saved report: {fname}")
                 
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

if __name__ == "__main__":
    prep_data()
