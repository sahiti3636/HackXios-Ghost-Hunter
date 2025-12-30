#!/usr/bin/env python3
"""
Enhanced Ghost Hunter Pipeline with GenAI Intelligence Layer
Integrates the intelligence analyzer into the main pipeline workflow.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Import the original pipeline
from main_pipeline import GhostHunterPipeline

# Import the intelligence analyzer
try:
    from intelligence_analyzer import IntelligenceAnalyzer
    GENAI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GenAI features not available: {e}")
    print("Install requirements: pip install -r requirements_genai.txt")
    GENAI_AVAILABLE = False

class EnhancedGhostHunterPipeline(GhostHunterPipeline):
    """
    Enhanced Ghost Hunter Pipeline with GenAI Intelligence Analysis
    """
    
    def __init__(self):
        super().__init__()
        self.intelligence_analyzer = None
        
        # Initialize GenAI if available
        if GENAI_AVAILABLE and os.getenv('GOOGLE_API_KEY'):
            try:
                self.intelligence_analyzer = IntelligenceAnalyzer()
                print("✅ GenAI Intelligence Layer initialized")
            except Exception as e:
                print(f"⚠️ GenAI initialization failed: {e}")
                self.intelligence_analyzer = None
        elif GENAI_AVAILABLE:
            print("⚠️ GOOGLE_API_KEY not set - GenAI features disabled")
        
        self.scene_path = None # Allow dynamic override of the scene directory
        
    def get_satellite_scenes(self):
        """
        Override to allow processing a specific downloaded scene path.
        """
        if self.scene_path and os.path.exists(self.scene_path):
            # Dynamic single-scene mode (from real download)
            folder_name = os.path.basename(self.scene_path)
            # Find the TIFF measurement
            tiff_file = None
            measurement_dir = os.path.join(self.scene_path, 'measurement')
            if os.path.exists(measurement_dir):
                for f in os.listdir(measurement_dir):
                    if f.endswith('.tiff') and ('vv' in f.lower() or 'hh' in f.lower()):
                        tiff_file = os.path.join(measurement_dir, f)
                        break
            
            if tiff_file:
                print(f"✅ Using specific scene path: {self.scene_path}")
                return [{
                    'name': folder_name,
                    'path': self.scene_path,
                    'tiff_file': tiff_file
                }]
            else:
                print(f"⚠️ specific scene path set but no measurement TIFF found: {self.scene_path}")
                return []
                
        # Fallback to standard directory scan
        return super().get_satellite_scenes()

    def run_satellite_pipeline(self, satellite_scene, satellite_num, total_satellites):
        """
        Enhanced pipeline with intelligence analysis
        """
        sat_name = satellite_scene['name']
        
        # Run the standard Ghost Hunter pipeline
        success = super().run_satellite_pipeline(satellite_scene, satellite_num, total_satellites)
        
        if not success:
            return False
        
        # Add GenAI Intelligence Analysis
        if self.intelligence_analyzer:
            self.print_banner(f"🧠 STARTING GENAI INTELLIGENCE ANALYSIS: {sat_name.upper()}")
            self._generate_intelligence_analysis(sat_name)
        else:
            print("⚠️ Skipping GenAI analysis - not available")
        
        return True
    
    def _generate_intelligence_analysis(self, sat_name: str):
        """Generate GenAI-powered intelligence analysis"""
        
        report_path = f"final_ghost_hunter_report_{sat_name}.json"
        
        if not os.path.exists(report_path):
            print(f"❌ Detection report not found: {report_path}")
            return
        
        try:
            print(f"🔍 Analyzing detection data with GenAI...")
            
            # Prepare additional context
            context = {
                'mission_type': 'Marine Protected Area Surveillance',
                'priority': 'Illegal Fishing Detection',
                'region_type': 'Marine Protected Area',
                'analysis_focus': 'Dark vessel identification and behavioral analysis'
            }
            
            # Generate intelligence analysis
            intelligence_data = self.intelligence_analyzer.analyze_detection_report(
                report_path, context
            )
            
            # Generate human-readable reports
            markdown_report = self.intelligence_analyzer.generate_human_readable_report(
                intelligence_data, 'markdown'
            )
            text_report = self.intelligence_analyzer.generate_human_readable_report(
                intelligence_data, 'text'
            )
            
            # Save intelligence outputs
            intelligence_json_path = f"intelligence_analysis_{sat_name}.json"
            intelligence_md_path = f"intelligence_report_{sat_name}.md"
            intelligence_txt_path = f"intelligence_report_{sat_name}.txt"
            
            with open(intelligence_json_path, 'w') as f:
                json.dump(intelligence_data, f, indent=2)
            
            with open(intelligence_md_path, 'w') as f:
                f.write(markdown_report)
            
            with open(intelligence_txt_path, 'w') as f:
                f.write(text_report)
            
            # Print intelligence summary
            self._print_intelligence_summary(intelligence_data, sat_name)
            
            print(f"\n✅ Intelligence Analysis Complete for {sat_name.upper()}")
            print(f"📁 Generated Intelligence Reports:")
            print(f"   • {intelligence_json_path} - Structured analysis data")
            print(f"   • {intelligence_md_path} - Markdown report")
            print(f"   • {intelligence_txt_path} - Text report")
            
        except Exception as e:
            print(f"❌ Intelligence analysis failed: {e}")
            print("Continuing with standard detection results...")
    
    def _print_intelligence_summary(self, intelligence_data: Dict, sat_name: str):
        """Print a summary of the intelligence analysis"""
        
        print(f"\n🎯 INTELLIGENCE SUMMARY FOR {sat_name.upper()}:")
        print("=" * 50)
        
        # Threat assessment
        threat_level = intelligence_data.get('threat_assessment', 'UNKNOWN')
        threat_color = {
            'LOW': '🟢',
            'MODERATE': '🟡', 
            'HIGH': '🔴',
            'CRITICAL': '🚨'
        }.get(threat_level, '⚪')
        
        print(f"{threat_color} THREAT LEVEL: {threat_level}")
        
        # Confidence
        confidence = intelligence_data.get('confidence_level', 'UNKNOWN')
        print(f"🎯 CONFIDENCE: {confidence}")
        
        # Key findings
        findings = intelligence_data.get('key_findings', [])
        if findings:
            print(f"\n📋 KEY FINDINGS:")
            for i, finding in enumerate(findings[:3], 1):  # Show top 3
                print(f"   {i}. {finding}")
        
        # Top recommendations
        recommendations = intelligence_data.get('recommendations', [])
        if recommendations:
            print(f"\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:2], 1):  # Show top 2
                print(f"   {i}. {rec}")
        
        # Vessel summary
        vessel_analyses = intelligence_data.get('detailed_vessel_analyses', [])
        if vessel_analyses:
            high_risk_vessels = [v for v in vessel_analyses if v.get('risk_score', 0) > 60]
            print(f"\n🚢 VESSEL SUMMARY:")
            print(f"   • Total analyzed: {len(vessel_analyses)}")
            print(f"   • High risk (>60): {len(high_risk_vessels)}")

def main():
    """Main function with enhanced intelligence capabilities"""
    print("==================================================")
    print("   👻🧠 GHOST HUNTER: AI-ENHANCED VESSEL TRACKING")
    print("==================================================")
    
    # Check GenAI availability
    if GENAI_AVAILABLE and os.getenv('GOOGLE_API_KEY'):
        print("✅ GenAI Intelligence Layer: ENABLED")
    else:
        print("⚠️ GenAI Intelligence Layer: DISABLED")
        if GENAI_AVAILABLE:
            print("   Set GOOGLE_API_KEY environment variable to enable")
        else:
            print("   Install requirements: pip install -r requirements_genai.txt")
    
    pipeline = EnhancedGhostHunterPipeline()
    
    # Check prerequisites
    if not pipeline.check_prerequisites():
        return
    
    # Get scenes
    scenes = pipeline.get_satellite_scenes()
    if not scenes:
        print("❌ No satellite data found.")
        return
    
    # Auto-select first scene for demo
    selected_scene = scenes[0]
    print(f"\n⚡ Auto-selecting Region: {selected_scene['name']}")
    
    # Run enhanced pipeline
    pipeline.run_satellite_pipeline(selected_scene, 1, 1)
    
    print(f"\n🎉 ENHANCED GHOST HUNTER ANALYSIS COMPLETE!")
    print(f"Check the generated intelligence reports for detailed analysis.")

if __name__ == "__main__":
    main()