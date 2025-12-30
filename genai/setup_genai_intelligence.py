#!/usr/bin/env python3
"""
Setup and Testing Script for GenAI Intelligence Layer
Handles installation, configuration, and testing of the intelligence analysis system.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking GenAI requirements...")
    
    required_packages = [
        'langchain',
        'langchain-google-genai', 
        'google-generativeai',
        'pydantic',
        'python-dotenv',
        'jinja2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages

def install_requirements():
    """Install missing requirements"""
    print("\n📦 Installing GenAI requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', 'requirements_genai.txt'
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_environment():
    """Setup environment configuration"""
    print("\n⚙️ Setting up environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("📋 Creating .env file from template...")
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created")
            print("⚠️ Please edit .env file and add your GOOGLE_API_KEY")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️ GOOGLE_API_KEY not set in .env file")
        print("Please add your Google API key to the .env file")
        return False
    elif api_key == 'your_google_api_key_here':
        print("⚠️ Please replace the placeholder API key in .env file")
        return False
    else:
        print("✅ GOOGLE_API_KEY configured")
        return True

def test_genai_connection():
    """Test connection to Google Gemini API"""
    print("\n🧪 Testing GenAI connection...")
    
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
            return True
        else:
            print(f"⚠️ Unexpected response: {response.content}")
            return False
            
    except Exception as e:
        print(f"❌ GenAI connection failed: {e}")
        return False

def test_intelligence_analysis():
    """Test intelligence analysis with sample data"""
    print("\n🔬 Testing intelligence analysis...")
    
    # Check if we have a detection report to analyze
    report_path = "final_ghost_hunter_report_sat1.json"
    
    if not os.path.exists(report_path):
        print(f"⚠️ No detection report found at {report_path}")
        print("Run the Ghost Hunter pipeline first to generate test data")
        return False
    
    try:
        from intelligence_analyzer import IntelligenceAnalyzer
        
        analyzer = IntelligenceAnalyzer()
        
        # Test analysis
        print("🔍 Running intelligence analysis...")
        intelligence_data = analyzer.analyze_detection_report(report_path)
        
        # Validate results
        required_fields = ['executive_summary', 'threat_assessment', 'key_findings', 'recommendations']
        
        for field in required_fields:
            if field in intelligence_data:
                print(f"✅ {field}")
            else:
                print(f"❌ Missing {field}")
                return False
        
        # Generate reports
        markdown_report = analyzer.generate_human_readable_report(intelligence_data, 'markdown')
        
        # Save test results
        with open('test_intelligence_analysis.json', 'w') as f:
            json.dump(intelligence_data, f, indent=2)
        
        with open('test_intelligence_report.md', 'w') as f:
            f.write(markdown_report)
        
        print("✅ Intelligence analysis test completed successfully")
        print("📁 Test files generated:")
        print("  • test_intelligence_analysis.json")
        print("  • test_intelligence_report.md")
        
        return True
        
    except Exception as e:
        print(f"❌ Intelligence analysis test failed: {e}")
        return False

def run_enhanced_pipeline_test():
    """Test the enhanced pipeline with GenAI"""
    print("\n🚀 Testing enhanced pipeline...")
    
    try:
        from enhanced_ghost_hunter_pipeline import EnhancedGhostHunterPipeline
        
        pipeline = EnhancedGhostHunterPipeline()
        
        if pipeline.intelligence_analyzer:
            print("✅ Enhanced pipeline with GenAI initialized")
            return True
        else:
            print("⚠️ Enhanced pipeline initialized without GenAI")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced pipeline test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions for the GenAI system"""
    print("\n" + "="*60)
    print("GENAI INTELLIGENCE LAYER - USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n🎯 BASIC USAGE:")
    print("1. Run standard Ghost Hunter pipeline:")
    print("   python main_pipeline.py")
    print("\n2. Run enhanced pipeline with GenAI:")
    print("   python enhanced_ghost_hunter_pipeline.py")
    print("\n3. Analyze existing reports:")
    print("   python intelligence_analyzer.py")
    
    print("\n📊 ANALYSIS TYPES:")
    print("The system supports specialized analysis for:")
    print("• Illegal fishing detection")
    print("• Maritime security threats")
    print("• Environmental protection")
    print("• Tactical operations")
    print("• Pattern analysis")
    print("• Multi-source intelligence fusion")
    print("• Executive briefings")
    
    print("\n📁 OUTPUT FILES:")
    print("• intelligence_analysis_[region].json - Structured analysis")
    print("• intelligence_report_[region].md - Markdown report")
    print("• intelligence_report_[region].txt - Text report")
    
    print("\n⚙️ CONFIGURATION:")
    print("Edit .env file to customize:")
    print("• GOOGLE_API_KEY - Your Google API key")
    print("• GENAI_MODEL - Model to use (default: gemini-1.5-pro)")
    print("• GENAI_TEMPERATURE - Response creativity (default: 0.3)")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• Check GOOGLE_API_KEY is set correctly")
    print("• Ensure all requirements are installed")
    print("• Verify internet connection for API access")
    print("• Check API quota and billing status")

def main():
    """Main setup and testing function"""
    print("="*60)
    print("GENAI INTELLIGENCE LAYER - SETUP & TESTING")
    print("="*60)
    
    # Step 1: Check requirements
    missing_packages = check_requirements()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        
        install_choice = input("\nInstall missing packages? (y/n): ").lower().strip()
        if install_choice == 'y':
            if not install_requirements():
                print("❌ Setup failed - could not install requirements")
                return
        else:
            print("❌ Setup cancelled - requirements not installed")
            return
    
    # Step 2: Setup environment
    if not setup_environment():
        print("❌ Setup failed - environment configuration incomplete")
        print("Please configure your .env file with GOOGLE_API_KEY")
        return
    
    # Step 3: Test GenAI connection
    if not test_genai_connection():
        print("❌ Setup failed - GenAI connection not working")
        return
    
    # Step 4: Test intelligence analysis
    if not test_intelligence_analysis():
        print("⚠️ Intelligence analysis test skipped - no test data available")
        print("Run the Ghost Hunter pipeline first to generate detection data")
    
    # Step 5: Test enhanced pipeline
    if not run_enhanced_pipeline_test():
        print("⚠️ Enhanced pipeline test failed")
    
    # Success!
    print("\n" + "="*60)
    print("✅ GENAI INTELLIGENCE LAYER SETUP COMPLETE!")
    print("="*60)
    
    print_usage_instructions()

if __name__ == "__main__":
    main()