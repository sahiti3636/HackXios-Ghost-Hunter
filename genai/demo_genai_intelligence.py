#!/usr/bin/env python3
"""
Demo Script for GenAI Intelligence Layer
Demonstrates the intelligence analysis capabilities with mock data and simulated AI responses.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any

class MockIntelligenceAnalyzer:
    """Mock intelligence analyzer for demonstration purposes"""
    
    def __init__(self):
        """Initialize mock analyzer"""
        self.model_name = "gemini-1.5-pro (simulated)"
        print("🤖 Mock Intelligence Analyzer initialized")
        print("   (This demo simulates GenAI responses without requiring API access)")
    
    def analyze_detection_report(self, report_path: str, context: Dict = None) -> Dict[str, Any]:
        """Simulate intelligence analysis of detection report"""
        
        print(f"🔍 Analyzing detection report: {report_path}")
        
        # Load actual detection data
        with open(report_path, 'r') as f:
            detection_data = json.load(f)
        
        vessels = detection_data.get('vessels', [])
        dark_vessels = [v for v in vessels if v.get('ais_status') == 'DARK']
        
        # Generate simulated intelligence analysis
        intelligence_analysis = {
            "executive_summary": f"Maritime surveillance detected {len(vessels)} vessel candidates in the operational area, with {len(dark_vessels)} vessels operating without AIS transponders. This pattern suggests potential illegal fishing activity requiring immediate enforcement attention. The vessels demonstrate coordinated behavior patterns consistent with organized fishing operations in protected waters.",
            
            "threat_assessment": "HIGH" if len(dark_vessels) > 5 else "MODERATE" if len(dark_vessels) > 2 else "LOW",
            
            "key_findings": [
                f"{len(vessels)} total vessel detections in Marine Protected Area",
                f"{len(dark_vessels)} dark vessels operating without AIS identification",
                f"Average risk score: {sum(v.get('risk_score', 0) for v in vessels) / len(vessels) if vessels else 0:.1f}/100",
                "Vessels clustered in high-value fishing areas during restricted season",
                "Strong radar signatures consistent with commercial fishing vessels",
                "Coordinated movement patterns suggesting fleet operations"
            ],
            
            "vessel_analyses": self._generate_vessel_analyses(vessels[:5]),  # Analyze first 5 vessels
            
            "recommendations": [
                "Deploy maritime patrol assets for immediate vessel interdiction",
                "Coordinate with fisheries enforcement agencies for legal action",
                "Document evidence collection for potential prosecution",
                "Increase surveillance coverage in adjacent protected areas",
                "Cross-reference vessel signatures with known fishing fleet databases",
                "Implement real-time monitoring for continued illegal activity"
            ],
            
            "technical_notes": f"Analysis based on multi-sensor fusion including SAR imagery (SBCI detection), AIS tracking, CNN vessel verification, and behavioral pattern analysis. Detection confidence varies by vessel with average CNN confidence of {sum(v.get('cnn_confidence', 0) for v in vessels) / len(vessels) if vessels else 0:.4f}. Strong radar signatures indicate metallic vessels consistent with fishing vessel profiles.",
            
            "confidence_level": "HIGH",
            
            "detailed_vessel_analyses": self._generate_detailed_vessel_analyses(vessels[:3]),
            
            "analysis_metadata": {
                "source_report": report_path,
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzer_version": "1.0 (Demo)",
                "model_used": self.model_name
            }
        }
        
        return intelligence_analysis
    
    def _generate_vessel_analyses(self, vessels):
        """Generate basic vessel analysis summaries"""
        analyses = []
        
        for vessel in vessels:
            risk_score = vessel.get('risk_score', 0)
            ais_status = vessel.get('ais_status', 'UNKNOWN')
            sbci = vessel.get('max_sbci', 0)
            
            # Determine threat level based on risk factors
            if risk_score > 70:
                threat_level = "CRITICAL"
            elif risk_score > 50:
                threat_level = "HIGH"
            elif risk_score > 30:
                threat_level = "MODERATE"
            else:
                threat_level = "LOW"
            
            analyses.append({
                "vessel_id": vessel.get('vessel_id'),
                "threat_level": threat_level,
                "risk_score": risk_score,
                "primary_concerns": [
                    "AIS transponder offline" if ais_status == "DARK" else "AIS operational",
                    f"Strong radar signature (SBCI: {sbci:.1f})" if sbci > 8 else f"Moderate radar signature (SBCI: {sbci:.1f})",
                    "Operating in protected waters",
                    "Vessel size consistent with commercial fishing"
                ]
            })
        
        return analyses
    
    def _generate_detailed_vessel_analyses(self, vessels):
        """Generate detailed analysis for individual vessels"""
        detailed_analyses = []
        
        for vessel in vessels:
            vessel_id = vessel.get('vessel_id')
            lat = vessel.get('latitude', 0)
            lon = vessel.get('longitude', 0)
            risk_score = vessel.get('risk_score', 0)
            ais_status = vessel.get('ais_status', 'UNKNOWN')
            sbci = vessel.get('max_sbci', 0)
            size_pixels = vessel.get('vessel_size_pixels', 0)
            
            # Generate contextual analysis based on vessel characteristics
            analysis_text = f"""
**VESSEL {vessel_id} INTELLIGENCE ASSESSMENT**

**Threat Classification:** {'CRITICAL' if risk_score > 70 else 'HIGH' if risk_score > 50 else 'MODERATE'}

**Technical Signature Analysis:**
This vessel exhibits a strong radar signature with SBCI value of {sbci:.1f}, indicating a metallic structure consistent with commercial fishing vessels. The vessel size of {size_pixels} pixels suggests a medium to large commercial vessel, likely in the 20-40 meter range typical of illegal fishing operations.

**Behavioral Assessment:**
The vessel is operating without AIS identification (dark vessel status), which is a primary indicator of illegal activity in Marine Protected Areas. This behavior pattern is consistent with vessels attempting to avoid detection and regulatory oversight.

**Geographic Context:**
Located at {lat:.4f}°N, {lon:.4f}°E within the Marine Protected Area boundaries. This position is in a high-value fishing zone during a restricted season, further supporting the assessment of illegal fishing activity.

**Risk Factors:**
- AIS transponder deliberately disabled or malfunctioning
- Operating in restricted Marine Protected Area
- Vessel characteristics match illegal fishing fleet profiles
- Timing coincides with peak fishing season restrictions

**Recommended Actions:**
1. Immediate interdiction by maritime patrol assets
2. Evidence collection for legal proceedings
3. Vessel identification and registration verification
4. Crew interviews and cargo inspection
5. Coordination with fisheries enforcement agencies

**Intelligence Confidence:** HIGH - Multiple corroborating indicators support illegal fishing assessment
            """.strip()
            
            detailed_analyses.append({
                'vessel_id': vessel_id,
                'analysis': analysis_text,
                'coordinates': f"{lat:.4f}°N, {lon:.4f}°E",
                'risk_score': risk_score
            })
        
        return detailed_analyses
    
    def generate_human_readable_report(self, intelligence_data: Dict, output_format: str = 'markdown') -> str:
        """Generate human-readable intelligence report"""
        
        if output_format == 'markdown':
            return self._generate_markdown_report(intelligence_data)
        else:
            return self._generate_text_report(intelligence_data)
    
    def _generate_markdown_report(self, data: Dict) -> str:
        """Generate markdown intelligence report"""
        
        report_lines = []
        
        # Header
        report_lines.append("# MARITIME INTELLIGENCE REPORT")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"**Source:** Ghost Hunter Maritime Surveillance System")
        report_lines.append(f"**Analysis Engine:** {self.model_name}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## EXECUTIVE SUMMARY")
        report_lines.append(data.get('executive_summary', 'No summary available'))
        report_lines.append("")
        
        # Threat Assessment
        report_lines.append("## THREAT ASSESSMENT")
        threat_level = data.get('threat_assessment', 'UNKNOWN')
        threat_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🔴', 'CRITICAL': '🚨'}.get(threat_level, '⚪')
        report_lines.append(f"{threat_emoji} **Overall Threat Level:** {threat_level}")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("## KEY FINDINGS")
        for finding in data.get('key_findings', []):
            report_lines.append(f"- {finding}")
        report_lines.append("")
        
        # Vessel Analyses Summary
        vessel_analyses = data.get('vessel_analyses', [])
        if vessel_analyses:
            report_lines.append("## VESSEL THREAT SUMMARY")
            report_lines.append("| Vessel ID | Threat Level | Risk Score | Primary Concerns |")
            report_lines.append("|-----------|--------------|------------|------------------|")
            
            for vessel in vessel_analyses:
                vessel_id = vessel.get('vessel_id', 'Unknown')
                threat_level = vessel.get('threat_level', 'Unknown')
                risk_score = vessel.get('risk_score', 0)
                concerns = ', '.join(vessel.get('primary_concerns', [])[:2])  # First 2 concerns
                
                report_lines.append(f"| {vessel_id} | {threat_level} | {risk_score}/100 | {concerns} |")
            
            report_lines.append("")
        
        # Detailed Vessel Analyses
        detailed_analyses = data.get('detailed_vessel_analyses', [])
        if detailed_analyses:
            report_lines.append("## DETAILED VESSEL ANALYSIS")
            for vessel in detailed_analyses:
                report_lines.append(f"### Vessel {vessel.get('vessel_id')} - Risk Score: {vessel.get('risk_score')}/100")
                report_lines.append(f"**Location:** {vessel.get('coordinates')}")
                report_lines.append(vessel.get('analysis', 'No analysis available'))
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## RECOMMENDATIONS")
        for i, rec in enumerate(data.get('recommendations', []), 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")
        
        # Technical Notes
        if data.get('technical_notes'):
            report_lines.append("## TECHNICAL NOTES")
            report_lines.append(data['technical_notes'])
            report_lines.append("")
        
        # Confidence Assessment
        report_lines.append("## CONFIDENCE ASSESSMENT")
        confidence = data.get('confidence_level', 'UNKNOWN')
        confidence_emoji = {'HIGH': '🎯', 'MEDIUM': '🎲', 'LOW': '❓'}.get(confidence, '❔')
        report_lines.append(f"{confidence_emoji} **Analysis Confidence:** {confidence}")
        
        return "\n".join(report_lines)
    
    def _generate_text_report(self, data: Dict) -> str:
        """Generate plain text intelligence report"""
        
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("MARITIME INTELLIGENCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"Source: Ghost Hunter Maritime Surveillance System")
        report_lines.append(f"Analysis Engine: {self.model_name}")
        report_lines.append("")
        
        report_lines.append("EXECUTIVE SUMMARY:")
        report_lines.append("-" * 20)
        report_lines.append(data.get('executive_summary', 'No summary available'))
        report_lines.append("")
        
        report_lines.append(f"THREAT ASSESSMENT: {data.get('threat_assessment', 'UNKNOWN')}")
        report_lines.append("")
        
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 15)
        for i, finding in enumerate(data.get('key_findings', []), 1):
            report_lines.append(f"{i}. {finding}")
        report_lines.append("")
        
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 18)
        for i, rec in enumerate(data.get('recommendations', []), 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")
        
        report_lines.append(f"CONFIDENCE LEVEL: {data.get('confidence_level', 'UNKNOWN')}")
        
        return "\n".join(report_lines)

def run_demo():
    """Run the GenAI intelligence demo"""
    
    print("=" * 70)
    print("GENAI INTELLIGENCE LAYER - DEMONSTRATION")
    print("=" * 70)
    print("This demo shows the GenAI intelligence capabilities using simulated AI responses.")
    print("No API key required - perfect for testing and evaluation!")
    print("")
    
    # Initialize mock analyzer
    analyzer = MockIntelligenceAnalyzer()
    
    # Check for detection report
    report_path = "final_ghost_hunter_report_sat1.json"
    
    if not os.path.exists(report_path):
        print(f"❌ Detection report not found: {report_path}")
        print("Please run the Ghost Hunter pipeline first:")
        print("   python main_pipeline.py")
        return
    
    print(f"✅ Found detection report: {report_path}")
    
    # Analyze the report
    print("\n🧠 Generating intelligence analysis...")
    
    context = {
        'mission_type': 'Marine Protected Area Surveillance',
        'priority': 'Illegal Fishing Detection',
        'operational_focus': 'Dark vessel identification and enforcement'
    }
    
    intelligence_data = analyzer.analyze_detection_report(report_path, context)
    
    # Generate reports
    print("📝 Generating human-readable reports...")
    
    markdown_report = analyzer.generate_human_readable_report(intelligence_data, 'markdown')
    text_report = analyzer.generate_human_readable_report(intelligence_data, 'text')
    
    # Save outputs
    output_files = {
        'demo_intelligence_analysis.json': intelligence_data,
        'demo_intelligence_report.md': markdown_report,
        'demo_intelligence_report.txt': text_report
    }
    
    for filename, content in output_files.items():
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            with open(filename, 'w') as f:
                f.write(content)
    
    # Print summary
    print("\n" + "=" * 50)
    print("DEMO INTELLIGENCE ANALYSIS COMPLETE")
    print("=" * 50)
    
    print(f"\n🎯 ANALYSIS SUMMARY:")
    print(f"Threat Level: {intelligence_data.get('threat_assessment')}")
    print(f"Confidence: {intelligence_data.get('confidence_level')}")
    print(f"Vessels Analyzed: {len(intelligence_data.get('vessel_analyses', []))}")
    print(f"Key Findings: {len(intelligence_data.get('key_findings', []))}")
    print(f"Recommendations: {len(intelligence_data.get('recommendations', []))}")
    
    print(f"\n📁 GENERATED FILES:")
    for filename in output_files.keys():
        print(f"   • {filename}")
    
    print(f"\n💡 NEXT STEPS:")
    print("1. Review the generated intelligence reports")
    print("2. Set up real GenAI with Google API key for production use")
    print("3. Customize analysis prompts for your specific use case")
    print("4. Integrate with your operational workflows")
    
    # Show sample of the analysis
    print(f"\n📋 SAMPLE INTELLIGENCE OUTPUT:")
    print("-" * 40)
    print(intelligence_data.get('executive_summary', '')[:200] + "...")
    
    print(f"\n✅ Demo completed successfully!")
    print("Check the generated files for full intelligence reports.")

if __name__ == "__main__":
    run_demo()