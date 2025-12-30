#!/usr/bin/env python3
"""
GenAI Intelligence Analyzer for Ghost Hunter
Transforms structured ML detection outputs into explainable intelligence summaries
using LangChain and Google Gemini.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VesselIntelligence:
    """Structured vessel intelligence data"""
    vessel_id: int
    threat_level: str
    confidence_assessment: str
    technical_summary: str
    behavioral_analysis: str
    risk_factors: List[str]
    recommendations: List[str]
    coordinates: str
    detection_quality: str

class IntelligenceReport(BaseModel):
    """Pydantic model for structured intelligence output"""
    executive_summary: str = Field(description="High-level summary for decision makers")
    threat_assessment: str = Field(description="Overall threat level assessment")
    key_findings: List[str] = Field(description="List of key findings")
    vessel_analyses: List[Dict[str, Any]] = Field(description="Individual vessel analyses")
    recommendations: List[str] = Field(description="Actionable recommendations")
    technical_notes: str = Field(description="Technical details for analysts")
    confidence_level: str = Field(description="Overall confidence in the analysis")

class IntelligenceAnalyzer:
    """Main intelligence analysis engine using Gemini"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the intelligence analyzer"""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize Gemini model with timeout
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=4000,
            timeout=60,  # 60 second timeout
            max_retries=2
        )
        
        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=IntelligenceReport)
        
        # Load analysis templates
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates for different analysis types"""
        
        # Main intelligence analysis template
        self.intelligence_template = PromptTemplate(
            input_variables=["detection_data", "context_info", "format_instructions"],
            template="""
You are a senior maritime intelligence analyst.
CONTEXT:
{context_info}

DETECTION DATA:
{detection_data}

TASK:
Provide a high-priority tactical summary of this activity.
CRITICAL FORMATTING REQUIREMENT:
- Output MUST be exactly 5 concise bullet points.
- No introductory text or conversational filler.
- Focus on: Threat Level, Dark Vessel Count, Key Risks, Location Context, and Immediate Recommendation.

{format_instructions}
"""
        )
        
        # Vessel-specific analysis template
        self.vessel_template = PromptTemplate(
            input_variables=["vessel_data", "regional_context"],
            template="""
Analyze this individual vessel detection for intelligence value:

VESSEL DATA: {vessel_data}
REGIONAL CONTEXT: {regional_context}

Provide a focused analysis covering:
1. Threat assessment and classification
2. Technical signature interpretation
3. Behavioral indicators
4. Recommended actions
5. Confidence assessment

Focus on maritime domain expertise and operational implications.
"""
        )
    
    def analyze_detection_report(self, report_path: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze a Ghost Hunter detection report and generate intelligence summary
        
        Args:
            report_path: Path to the JSON detection report
            context: Additional context information
            
        Returns:
            Structured intelligence analysis
        """
        logger.info(f"Analyzing detection report: {report_path}")
        
        # Load detection data
        with open(report_path, 'r') as f:
            detection_data = json.load(f)
        
        # Prepare context information
        context_info = self._prepare_context(detection_data, context)
        
        # Generate intelligence analysis
        intelligence_report = self._generate_intelligence_analysis(detection_data, context_info)
        
        # Add metadata
        intelligence_report['analysis_metadata'] = {
            'source_report': report_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'analyzer_version': '1.0',
            'model_used': 'gemini-1.5-pro'
        }
        
        return intelligence_report
    
    def _prepare_context(self, detection_data: Dict, additional_context: Optional[Dict] = None) -> str:
        """Prepare contextual information for analysis"""
        
        context_parts = []
        
        # Basic mission context
        region = detection_data.get('region', 'Unknown')
        timestamp = detection_data.get('timestamp', 'Unknown')
        vessel_count = len(detection_data.get('vessels', []))
        
        context_parts.append(f"MISSION: Marine surveillance in region {region}")
        context_parts.append(f"TIMESTAMP: {timestamp}")
        context_parts.append(f"DETECTIONS: {vessel_count} vessel candidates identified")
        
        # Analyze detection patterns
        vessels = detection_data.get('vessels', [])
        if vessels:
            dark_vessels = sum(1 for v in vessels if v.get('ais_status') == 'DARK')
            avg_risk = sum(v.get('risk_score', 0) for v in vessels) / len(vessels)
            
            context_parts.append(f"DARK VESSELS: {dark_vessels}/{vessel_count} without AIS")
            context_parts.append(f"AVERAGE RISK SCORE: {avg_risk:.1f}/100")
        
        # Geographic context
        if vessels:
            lats = [v.get('latitude', 0) for v in vessels if v.get('latitude')]
            lons = [v.get('longitude', 0) for v in vessels if v.get('longitude')]
            if lats and lons:
                context_parts.append(f"AREA: {min(lats):.3f}°N to {max(lats):.3f}°N, {min(lons):.3f}°E to {max(lons):.3f}°E")
        
        # Technical context
        context_parts.append("SENSORS: SAR imagery, AIS tracking, CNN verification, behavioral analysis")
        context_parts.append("DETECTION METHOD: SBCI (Ship-to-Background Contrast Index)")
        
        # Additional context
        if additional_context:
            for key, value in additional_context.items():
                context_parts.append(f"{key.upper()}: {value}")
        
        return "\n".join(context_parts)
    
    def _generate_intelligence_analysis(self, detection_data: Dict, context_info: str) -> Dict[str, Any]:
        """Generate the main intelligence analysis using Gemini"""
        
        # Prepare the prompt
        format_instructions = self.output_parser.get_format_instructions()
        
        prompt = self.intelligence_template.format(
            detection_data=json.dumps(detection_data, indent=2),
            context_info=context_info,
            format_instructions=format_instructions
        )
        
        try:
            # Generate analysis
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Parse structured output
            parsed_response = self.output_parser.parse(response.content)
            
            # Convert to dictionary
            intelligence_report = parsed_response.dict()
            
            # Add vessel-specific analyses
            intelligence_report['detailed_vessel_analyses'] = self._analyze_individual_vessels(
                detection_data.get('vessels', []), context_info
            )
            
            return intelligence_report
            
        except Exception as e:
            logger.error(f"Error generating intelligence analysis: {e}")
            # Fallback to basic analysis
            return self._generate_fallback_analysis(detection_data, context_info)
    
    def _analyze_individual_vessels(self, vessels: List[Dict], context: str) -> List[Dict]:
        """Generate detailed analysis for each vessel with timeout and error handling"""
        
        vessel_analyses = []
        max_vessels = min(len(vessels), int(os.getenv('MAX_VESSELS_DETAILED_ANALYSIS', 10)))
        
        logger.info(f"Analyzing {max_vessels} vessels individually...")
        
        # Try batch processing first to save API calls
        if max_vessels > 3:
            try:
                return self._analyze_vessels_batch(vessels[:max_vessels], context)
            except Exception as e:
                logger.warning(f"Batch analysis failed: {e}. Falling back to individual analysis...")
        
        # Individual vessel analysis (fallback)
        for i, vessel in enumerate(vessels[:max_vessels]):
            try:
                logger.info(f"Analyzing vessel {vessel.get('vessel_id')} ({i+1}/{max_vessels})")
                
                vessel_prompt = self.vessel_template.format(
                    vessel_data=json.dumps(vessel, indent=2),
                    regional_context=context
                )
                
                # Add timeout and retry logic
                import time
                start_time = time.time()
                
                response = self.llm.invoke([HumanMessage(content=vessel_prompt)])
                
                elapsed_time = time.time() - start_time
                logger.info(f"Vessel {vessel.get('vessel_id')} analysis completed in {elapsed_time:.1f}s")
                
                vessel_analyses.append({
                    'vessel_id': vessel.get('vessel_id'),
                    'analysis': response.content,
                    'coordinates': f"{vessel.get('latitude', 0):.4f}°N, {vessel.get('longitude', 0):.4f}°E",
                    'risk_score': vessel.get('risk_score', 0)
                })
                
                # Add small delay between requests to avoid rate limiting
                time.sleep(1.0)  # Increased delay to avoid quota issues
                
            except Exception as e:
                logger.warning(f"Error analyzing vessel {vessel.get('vessel_id')}: {e}")
                vessel_analyses.append({
                    'vessel_id': vessel.get('vessel_id'),
                    'analysis': f"Analysis unavailable: {str(e)[:100]}...",
                    'coordinates': f"{vessel.get('latitude', 0):.4f}°N, {vessel.get('longitude', 0):.4f}°E",
                    'risk_score': vessel.get('risk_score', 0)
                })
        
        return vessel_analyses
    
    def _generate_fallback_analysis(self, detection_data: Dict, context_info: str) -> Dict[str, Any]:
        """Generate comprehensive analysis when AI processing fails or quota is exceeded"""
        
        vessels = detection_data.get('vessels', [])
        dark_vessels = [v for v in vessels if v.get('ais_status') == 'DARK']
        high_risk_vessels = [v for v in vessels if v.get('risk_score', 0) > 60]
        
        # Analyze all vessels in fallback mode
        vessel_analyses = []
        for vessel in vessels:
            analysis_text = self._generate_vessel_fallback_analysis(vessel)
            vessel_analyses.append({
                'vessel_id': vessel.get('vessel_id'),
                'analysis': analysis_text,
                'coordinates': f"{vessel.get('latitude', 0):.4f}°N, {vessel.get('longitude', 0):.4f}°E",
                'risk_score': vessel.get('risk_score', 0),
                'threat_level': 'HIGH' if vessel.get('risk_score', 0) > 60 else 'MODERATE' if vessel.get('ais_status') == 'DARK' else 'LOW'
            })
        
        # Determine overall threat level
        if len(high_risk_vessels) > 0:
            threat_level = 'HIGH'
        elif len(dark_vessels) > len(vessels) * 0.5:  # More than 50% dark vessels
            threat_level = 'MODERATE-HIGH'
        elif len(dark_vessels) > 0:
            threat_level = 'MODERATE'
        else:
            threat_level = 'LOW'
        
        return {
            'executive_summary': f"Detected {len(vessels)} vessel candidates in the surveillance area, with {len(dark_vessels)} operating without AIS transponders. {len(high_risk_vessels)} vessels flagged as high risk. All {len(vessels)} vessels have been analyzed for threat assessment and operational significance.",
            'threat_assessment': f'{threat_level}. Primary concerns include {len(dark_vessels)} dark vessels indicating potential intent to avoid detection, commonly associated with illegal fishing or other illicit maritime activities. The presence of vessels in this operational area warrants continued monitoring and potential enforcement action.',
            'key_findings': [
                f"{len(vessels)} total vessel detections analyzed",
                f"{len(dark_vessels)} dark vessels (no AIS) - {(len(dark_vessels)/len(vessels)*100):.1f}% of total",
                f"{len(high_risk_vessels)} high-risk vessels (score >60)",
                f"Average risk score: {sum(v.get('risk_score', 0) for v in vessels) / len(vessels) if vessels else 0:.1f}/100",
                f"Geographic spread: {self._analyze_geographic_distribution(vessels)}",
                f"Detection confidence range: {min(v.get('detection_confidence', 0) for v in vessels):.1f} - {max(v.get('detection_confidence', 0) for v in vessels):.1f}"
            ],
            'recommendations': [
                "Deploy maritime patrol assets to investigate dark vessel activities",
                "Cross-reference detection locations with historical AIS and intelligence data",
                "Prioritize high-risk vessels for immediate follow-up surveillance",
                "Coordinate with regional maritime authorities for enforcement action",
                "Continue monitoring the area for pattern analysis and trend identification"
            ],
            'technical_notes': f"Analysis generated using fallback mode due to API limitations. All {len(vessels)} vessels included in assessment. Detection data processed from Ghost Hunter maritime surveillance system.",
            'confidence_level': 'MODERATE - Based on systematic analysis of detection patterns, risk scores, and operational context',
            'detailed_vessel_analyses': vessel_analyses
        }
    
    def _generate_vessel_fallback_analysis(self, vessel: Dict) -> str:
        """Generate detailed analysis for individual vessel without AI"""
        
        vessel_id = vessel.get('vessel_id', 'Unknown')
        ais_status = vessel.get('ais_status', 'Unknown')
        risk_score = vessel.get('risk_score', 0)
        cnn_confidence = vessel.get('cnn_confidence', 0)
        detection_confidence = vessel.get('detection_confidence', 0)
        behavior = vessel.get('behavior_analysis', {})
        
        analysis_parts = []
        
        # AIS Status Analysis
        if ais_status == 'DARK':
            analysis_parts.append("DARK VESSEL ALERT: Operating without AIS transponder, indicating potential intent to avoid detection. This is a primary indicator of illegal fishing, smuggling, or unauthorized maritime activity.")
        elif ais_status == 'MATCHED':
            analysis_parts.append("AIS MATCHED: Vessel properly broadcasting identification signals, indicating compliance with maritime regulations.")
        
        # Risk Assessment
        if risk_score > 60:
            analysis_parts.append(f"HIGH RISK (Score: {risk_score}/100): Multiple threat indicators present. Immediate investigation recommended.")
        elif risk_score > 30:
            analysis_parts.append(f"MODERATE RISK (Score: {risk_score}/100): Some concerning indicators. Continued monitoring advised.")
        else:
            analysis_parts.append(f"LOW RISK (Score: {risk_score}/100): Minimal threat indicators detected.")
        
        # Detection Confidence
        if detection_confidence > 10:
            analysis_parts.append(f"Strong SAR signature (confidence: {detection_confidence:.1f}) confirms vessel presence.")
        elif detection_confidence > 5:
            analysis_parts.append(f"Moderate SAR signature (confidence: {detection_confidence:.1f}) indicates likely vessel.")
        else:
            analysis_parts.append(f"Weak SAR signature (confidence: {detection_confidence:.1f}) requires verification.")
        
        # CNN Verification
        if cnn_confidence < 0.1:
            analysis_parts.append(f"Low CNN verification confidence ({cnn_confidence:.4f}) suggests unusual vessel characteristics or small size requiring human verification.")
        
        # Behavioral Analysis
        suspicion_level = behavior.get('suspicion_level', 'UNKNOWN')
        flags = behavior.get('flags', [])
        
        if suspicion_level == 'HIGH' or flags:
            analysis_parts.append(f"Behavioral concerns: {suspicion_level} suspicion level with flags: {', '.join(flags) if flags else 'None specified'}.")
        
        return " ".join(analysis_parts)
    
    def _analyze_geographic_distribution(self, vessels: List[Dict]) -> str:
        """Analyze geographic distribution of vessels"""
        
        if not vessels:
            return "No vessels to analyze"
        
        lats = [v.get('latitude', 0) for v in vessels]
        lons = [v.get('longitude', 0) for v in vessels]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        if lat_range < 0.1 and lon_range < 0.1:
            return "Clustered in small area"
        elif lat_range < 0.5 and lon_range < 0.5:
            return "Moderate geographic spread"
        else:
            return "Wide geographic distribution"
    
    def generate_human_readable_report(self, intelligence_data: Dict, output_format: str = 'markdown') -> str:
        """Generate human-readable intelligence report"""
        
        if output_format == 'markdown':
            return self._generate_markdown_report(intelligence_data)
        elif output_format == 'html':
            return self._generate_html_report(intelligence_data)
        else:
            return self._generate_text_report(intelligence_data)
    
    def _generate_markdown_report(self, data: Dict) -> str:
        """Generate markdown intelligence report"""
        
        report_lines = []
        
        # Header
        report_lines.append("# MARITIME INTELLIGENCE REPORT")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"**Source:** Ghost Hunter Maritime Surveillance System")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("## EXECUTIVE SUMMARY")
        report_lines.append(data.get('executive_summary', 'No summary available'))
        report_lines.append("")
        
        # Threat Assessment
        report_lines.append("## THREAT ASSESSMENT")
        threat_level = data.get('threat_assessment', 'UNKNOWN')
        report_lines.append(f"**Overall Threat Level:** {threat_level}")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("## KEY FINDINGS")
        for finding in data.get('key_findings', []):
            report_lines.append(f"- {finding}")
        report_lines.append("")
        
        # Vessel Analyses
        if 'detailed_vessel_analyses' in data:
            report_lines.append("## VESSEL ANALYSIS")
            for vessel in data['detailed_vessel_analyses']:
                report_lines.append(f"### Vessel {vessel.get('vessel_id')} - Risk Score: {vessel.get('risk_score')}/100")
                report_lines.append(f"**Location:** {vessel.get('coordinates')}")
                report_lines.append(vessel.get('analysis', 'No analysis available'))
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## RECOMMENDATIONS")
        for rec in data.get('recommendations', []):
            report_lines.append(f"- {rec}")
        report_lines.append("")
        
        # Technical Notes
        if data.get('technical_notes'):
            report_lines.append("## TECHNICAL NOTES")
            report_lines.append(data['technical_notes'])
            report_lines.append("")
        
        # Confidence Assessment
        report_lines.append("## CONFIDENCE ASSESSMENT")
        report_lines.append(f"**Analysis Confidence:** {data.get('confidence_level', 'UNKNOWN')}")
        
        return "\n".join(report_lines)
    
    def _generate_text_report(self, data: Dict) -> str:
        """Generate plain text intelligence report"""
        
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("MARITIME INTELLIGENCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"Source: Ghost Hunter Maritime Surveillance System")
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

def main():
    """Main function for testing the intelligence analyzer"""
    
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("⚠️ GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key to use the GenAI features")
        return
    
    # Initialize analyzer
    try:
        analyzer = IntelligenceAnalyzer()
        print("✅ Intelligence Analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Test with existing report
    report_path = "final_ghost_hunter_report_sat1.json"
    if os.path.exists(report_path):
        print(f"\n🔍 Analyzing report: {report_path}")
        
        try:
            # Generate intelligence analysis
            intelligence = analyzer.analyze_detection_report(report_path)
            
            # Generate human-readable report
            markdown_report = analyzer.generate_human_readable_report(intelligence, 'markdown')
            
            # Save reports
            with open('intelligence_analysis.json', 'w') as f:
                json.dump(intelligence, f, indent=2)
            
            with open('intelligence_report.md', 'w') as f:
                f.write(markdown_report)
            
            print("✅ Intelligence analysis completed successfully")
            print("📁 Generated files:")
            print("  • intelligence_analysis.json - Structured analysis data")
            print("  • intelligence_report.md - Human-readable report")
            
            # Print summary
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"Threat Level: {intelligence.get('threat_assessment', 'Unknown')}")
            print(f"Confidence: {intelligence.get('confidence_level', 'Unknown')}")
            print(f"Key Findings: {len(intelligence.get('key_findings', []))}")
            print(f"Recommendations: {len(intelligence.get('recommendations', []))}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    else:
        print(f"❌ Report file not found: {report_path}")
        print("Run the Ghost Hunter pipeline first to generate detection data")

if __name__ == "__main__":
    main()