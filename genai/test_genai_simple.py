#!/usr/bin/env python3
"""
Simple test script for GenAI Intelligence Layer
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from dotenv import load_dotenv

def test_basic_connection():
    """Test basic GenAI connection"""
    print("🧪 Testing GenAI connection...")
    
    try:
        from intelligence_analyzer import IntelligenceAnalyzer
        
        analyzer = IntelligenceAnalyzer()
        print("✅ Intelligence Analyzer initialized successfully")
        
        # Test basic functionality with a simple prompt
        from langchain_core.messages import HumanMessage
        
        test_message = HumanMessage(content="Respond with 'GenAI connection successful' if you can read this.")
        response = analyzer.llm.invoke([test_message])
        
        if "successful" in response.content.lower():
            print("✅ GenAI API connection working")
            print(f"Response: {response.content}")
            return True
        else:
            print(f"⚠️ Unexpected response: {response.content}")
            return False
            
    except Exception as e:
        print(f"❌ GenAI connection failed: {e}")
        return False

def test_intelligence_analysis_simple():
    """Test intelligence analysis with minimal data"""
    print("\n🔬 Testing intelligence analysis with simple data...")
    
    try:
        from intelligence_analyzer import IntelligenceAnalyzer
        
        analyzer = IntelligenceAnalyzer()
        
        # Create simple test data
        simple_detection_data = {
            "metadata": {
                "satellite_name": "test_sat",
                "region": "test_region",
                "analysis_timestamp": "2024-01-01T00:00:00Z"
            },
            "detection_summary": {
                "total_detections": 2,
                "dark_vessels": 1,
                "ais_matched": 1
            },
            "vessels": [
                {
                    "vessel_id": "test_vessel_1",
                    "coordinates": [10.0, 20.0],
                    "confidence": 0.85,
                    "ais_status": "dark",
                    "risk_score": 0.7
                },
                {
                    "vessel_id": "test_vessel_2", 
                    "coordinates": [10.1, 20.1],
                    "confidence": 0.92,
                    "ais_status": "matched",
                    "risk_score": 0.2
                }
            ]
        }
        
        # Test analysis
        print("🔍 Running intelligence analysis on test data...")
        intelligence_data = analyzer._generate_intelligence_analysis(
            simple_detection_data, 
            {"analysis_type": "illegal_fishing"}
        )
        
        # Validate results
        required_fields = ['executive_summary', 'threat_assessment', 'key_findings', 'recommendations']
        
        success = True
        for field in required_fields:
            if field in intelligence_data:
                print(f"✅ {field}")
            else:
                print(f"❌ Missing {field}")
                success = False
        
        if success:
            print("✅ Intelligence analysis test completed successfully")
            
            # Save test results
            with open('test_simple_intelligence.json', 'w') as f:
                json.dump(intelligence_data, f, indent=2)
            
            print("📁 Test file generated: test_simple_intelligence.json")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"❌ Intelligence analysis test failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*50)
    print("GENAI SIMPLE TEST")
    print("="*50)
    
    # Load environment
    load_dotenv()
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("❌ Basic connection test failed")
        return
    
    # Test 2: Simple intelligence analysis
    if not test_intelligence_analysis_simple():
        print("❌ Intelligence analysis test failed")
        return
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("GenAI Intelligence Layer is working correctly")
    print("="*50)

if __name__ == "__main__":
    main()