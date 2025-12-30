#!/usr/bin/env python3
"""
Test batch vessel analysis to ensure all vessels are processed
"""

import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from intelligence_analyzer import IntelligenceAnalyzer

def test_batch_analysis():
    """Test batch analysis with multiple vessels"""
    
    # Load the actual detection report
    with open('final_ghost_hunter_report_sat1.json', 'r') as f:
        detection_data = json.load(f)
    
    print(f"🔍 Testing batch analysis with {len(detection_data['vessels'])} vessels...")
    
    analyzer = IntelligenceAnalyzer()
    
    # Test batch vessel analysis directly
    vessels = detection_data['vessels']
    context = "illegal_fishing"
    
    try:
        vessel_analyses = analyzer._analyze_vessels_batch(vessels, context)
        
        print(f"✅ Batch analysis completed for {len(vessel_analyses)} vessels")
        
        for analysis in vessel_analyses:
            print(f"   • Vessel {analysis['vessel_id']}: {analysis.get('threat_level', 'UNKNOWN')} threat")
            print(f"     Analysis: {analysis['analysis'][:100]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        return False

if __name__ == "__main__":
    test_batch_analysis()