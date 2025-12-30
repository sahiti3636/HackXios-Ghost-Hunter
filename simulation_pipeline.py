import os
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

class SimulationPipeline:
    """
    A lightweight mock pipeline for Ghost Hunter deployment.
    Simulates vessel detection and satellite imagery without heavy ML dependencies.
    """
    
    def __init__(self):
        self.genai_available = True
        print("🚀 SimulationPipeline initialized (Lightweight Mode)")
        
    def check_prerequisites(self):
        return True
        
    def get_satellite_scenes(self) -> List[Dict]:
        """Return a mock scene object to satisfy the API contract"""
        return [{
            'name': 'Ghost_Hunter_Live_Simulation',
            'acquisition_date': datetime.now().strftime('%Y-%m-%d'),
            'properties': {
                'id': 'SIMULATION_MODE_ACTIVE',
                'title': 'Ghost_Hunter_Live_Simulation',
                'ingestiondate': datetime.now().strftime('%Y-%m-%dT%H:%M:%S.000Z')
            }
        }]

    def run_satellite_pipeline(self, target_scene, current_num, total_num) -> bool:
        """Mock pipeline execution matching the signature of the real one"""
        print(f"📡 Simulating analysis for {target_scene['name']}...")
        time.sleep(2) # Fake processing time
        
        # We don't need to do anything here because app_light.py will handle
        # the result generation directly using our process_scene logic equivalent.
        # But to match app.py logic which expects a report file:
        
        region_name = target_scene['name']
        report_path = f"final_ghost_hunter_report_{region_name}.json"
        
        # Generate mock results
        results = self.process_scene(target_scene)
        
        # Save to JSON as app.py expects
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        return True

    def process_scene(self, scene_metadata, mpa_name=None):
        """
        Load cached real data if available, otherwise fallback.
        """
        cache_file = "data/cached_reports/cached_papahanaumokuakea.json" # Default
        
        # If MPA is selected, try to load its specific cached report
        if mpa_name:
             # mpa_name is already sanitized by app_light.py (e.g. "papahanaumokuakea")
             candidate = f"data/cached_reports/cached_{mpa_name}.json"
             if os.path.exists(candidate):
                 cache_file = candidate
                 print(f"📂 Loading cached report for {mpa_name}: {cache_file}")
             else:
                 print(f"⚠️ Cache not found for {mpa_name} at {candidate}, using default.")
        
        # Load the Real Data
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                # Timestamp update to make it feel "fresh"
                data['timestamp'] = datetime.now().isoformat()
                
                # RE-CALCULATE RISK SCORES to include new logic (CNN Suspicion)
                # This ensures any changes to risk_fusion.py are reflected immediately
                from risk_fusion import calculate_risk_score
                
                updated_vessels = []
                for v in data.get('vessels', []):
                    # Ensure confidence is present (from original verification)
                    # If cached data predates this key, it defaults to 0.0 in risk_fusion
                    updated_v = calculate_risk_score(v)
                    updated_vessels.append(updated_v)
                
                # --- USER REQUEST: GUARANTEE 2-3 DARK VESSELS ---
                # Check how many are currently High Risk (>70)
                high_risk_indices = [i for i, v in enumerate(updated_vessels) if v.get('risk_score', 0) > 70]
                target_min = random.randint(2, 3)
                
                if len(high_risk_indices) < target_min:
                    shortfall = target_min - len(high_risk_indices)
                    # Candidates: Low/Med risk vessels
                    candidates = [i for i, v in enumerate(updated_vessels) if v.get('risk_score', 0) <= 70]
                    
                    # Pick random candidates to boost
                    # Use min to avoid error if not enough candidates
                    boost_indices = random.sample(candidates, min(len(candidates), shortfall))
                    
                    for idx in boost_indices:
                        v = updated_vessels[idx]
                        # Boost to High Risk (75-95)
                        new_score = random.randint(75, 95)
                        v['risk_score'] = new_score
                        
                        # FORCE "DARK" STATUS VISUALS
                        v['ais_status'] = "Dark (Simulated)"
                        v['status'] = "High Risk" # Frontend often uses this
                        
                        # Add justification
                        if 'risk_breakdown' not in v: v['risk_breakdown'] = []
                        v['risk_breakdown'].append(f"Correlated Intelligence Signal (+40)")
                        v['risk_breakdown'].append(f"Pattern Anomaly Detected")
                        
                        updated_vessels[idx] = v
                        print(f"⚠️ Boosted vessel {v.get('vessel_id')} to High Risk ({new_score})/Dark to meet quota.")

                data['vessels'] = updated_vessels
                return data
        else:
            # Emergency Fallback if no files exist (should not happen if prep run)
            return {"error": "No cached data available", "vessels": []}

    def run_analysis(self):
        """Legacy fallback"""
        return self.process_scene(self.get_satellite_scenes()[0])
