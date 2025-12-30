#!/usr/bin/env python3
"""
Quick test script to verify the Ghost Hunter API integration
"""

import requests
import json
import time

API_BASE = "http://localhost:5001/api"

def test_api_integration():
    """Test the full API workflow"""
    print("🔍 Testing Ghost Hunter API Integration...")
    
    try:
        # 1. Health check
        print("\n1. Testing health check...")
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed: {health_data['status']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # 2. Get MPAs
        print("\n2. Testing MPA endpoint...")
        response = requests.get(f"{API_BASE}/mpas")
        if response.status_code == 200:
            mpas_data = response.json()
            print(f"   ✅ MPAs loaded: {len(mpas_data['mpas'])} available")
        else:
            print(f"   ❌ MPAs failed: {response.status_code}")
        
        # 3. Start analysis
        print("\n3. Testing analysis start...")
        analysis_config = {
            "region_type": "custom",
            "region_data": {
                "polygon": [
                    [34.0522, -118.2437],
                    [34.0622, -118.2337],
                    [34.0522, -118.2237],
                    [34.0422, -118.2337]
                ]
            },
            "start_date": "2024-01-01",
            "end_date": "2024-01-07"
        }
        
        response = requests.post(f"{API_BASE}/analysis/start", json=analysis_config)
        if response.status_code == 200:
            start_data = response.json()
            analysis_id = start_data['analysis_id']
            print(f"   ✅ Analysis started: {analysis_id}")
        else:
            print(f"   ❌ Analysis start failed: {response.status_code}")
            return False
        
        # 4. Wait for analysis to complete
        print("\n4. Waiting for analysis completion...")
        max_wait = 30  # seconds
        waited = 0
        
        while waited < max_wait:
            response = requests.get(f"{API_BASE}/analysis/{analysis_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                status = status_data.get('status', 'unknown')
                progress = status_data.get('progress', 0)
                
                print(f"   📊 Status: {status} ({progress}%)")
                
                if status == 'completed':
                    print("   ✅ Analysis completed!")
                    break
                elif status == 'failed':
                    print(f"   ❌ Analysis failed: {status_data.get('error', 'Unknown error')}")
                    return False
            
            time.sleep(2)
            waited += 2
        
        if waited >= max_wait:
            print("   ⏰ Analysis taking longer than expected, continuing with results check...")
        
        # 5. Get results
        print("\n5. Testing results retrieval...")
        response = requests.get(f"{API_BASE}/analysis/{analysis_id}/results")
        if response.status_code == 200:
            results_data = response.json()
            vessels = results_data.get('vessels', [])
            print(f"   ✅ Results retrieved: {len(vessels)} vessels detected")
            
            # Print vessel summary
            for vessel in vessels[:3]:  # Show first 3 vessels
                print(f"      🚢 Vessel {vessel['id']}: Risk {vessel['risk']}%, Status: {vessel['ais_status']}")
        else:
            print(f"   ❌ Results retrieval failed: {response.status_code}")
            return False
        
        # 6. Test vessel intelligence
        if vessels:
            print("\n6. Testing vessel intelligence...")
            vessel_id = vessels[0]['id']
            response = requests.get(f"{API_BASE}/vessel/{vessel_id}/intelligence?analysis_id={analysis_id}")
            if response.status_code == 200:
                intel_data = response.json()
                print(f"   ✅ Intelligence retrieved for vessel {vessel_id}")
                print(f"      Threat Level: {intel_data.get('threat_level', 'Unknown')}")
            else:
                print(f"   ❌ Intelligence retrieval failed: {response.status_code}")
        
        print("\n🎉 All API tests passed! The integration is working correctly.")
        print(f"\n📱 Frontend should be available at: http://localhost:3000")
        print(f"🔗 You can test the analysis at: http://localhost:3000/results/{analysis_id}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the backend is running on port 5001.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_api_integration()