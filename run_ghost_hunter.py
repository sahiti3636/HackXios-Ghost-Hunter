#!/usr/bin/env python3
"""
Ghost Hunter Full Stack Application Launcher
Starts both the Flask backend and provides instructions for the React frontend.
"""

import os
import sys
import subprocess
import time
import threading
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'flask_cors', 'numpy', 'torch', 'langchain', 'google-generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '-r', 'requirements_backend.txt'
            ])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def check_environment():
    """Check environment configuration"""
    print("⚙️ Checking environment configuration...")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️ .env file not found. Creating from template...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ .env file created from template")
            print("📝 Please edit .env file and add your GOOGLE_API_KEY")
        else:
            print("❌ .env.example not found")
            return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check critical environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your_google_api_key_here':
        print("⚠️ GOOGLE_API_KEY not configured in .env file")
        print("Please add your Google API key to enable GenAI functionality")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'uploads',
        'results', 
        'data/raw/satellite',
        'data/raw/mpa_boundaries',
        'output',
        'utils'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting Flask backend server...")
    
    try:
        # Import and run the Flask app
        from app import app, analysis_manager
        
        # Initialize pipeline
        if analysis_manager.initialize_pipeline():
            print("✅ Enhanced pipeline initialized")
        else:
            print("⚠️ Pipeline initialization failed - will retry on first request")
        
        print("🌐 Backend server starting on http://localhost:5000")
        print("📊 API documentation available at http://localhost:5000/api/health")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for production-like behavior
            threaded=True,
            use_reloader=False  # Prevent double startup in development
        )
        
    except Exception as e:
        print(f"❌ Failed to start backend server: {e}")
        return False

def check_frontend():
    """Check if frontend is available and provide instructions"""
    frontend_path = Path('ghost-hunter-frontend')
    
    if not frontend_path.exists():
        print("⚠️ Frontend directory not found")
        print("Please ensure the ghost-hunter-frontend directory is in the same location as this script")
        return False
    
    package_json = frontend_path / 'package.json'
    if not package_json.exists():
        print("⚠️ Frontend package.json not found")
        return False
    
    print("✅ Frontend directory found")
    return True

def print_startup_instructions():
    """Print instructions for starting the full application"""
    print("\n" + "="*80)
    print("🎯 GHOST HUNTER - FULL STACK APPLICATION")
    print("="*80)
    
    print("\n📋 STARTUP INSTRUCTIONS:")
    print("\n1. 🔧 BACKEND (Flask API):")
    print("   The backend server is starting automatically...")
    print("   API will be available at: http://localhost:5000")
    
    print("\n2. 🎨 FRONTEND (React App):")
    print("   Open a new terminal and run:")
    print("   cd ghost-hunter-frontend")
    print("   npm install  # (first time only)")
    print("   npm run dev")
    print("   Frontend will be available at: http://localhost:3000")
    
    print("\n3. 🌐 ACCESS THE APPLICATION:")
    print("   Once both servers are running, open your browser to:")
    print("   http://localhost:3000")
    
    print("\n📡 API ENDPOINTS:")
    print("   • GET  /api/health                    - Health check")
    print("   • POST /api/analysis/start            - Start new analysis")
    print("   • GET  /api/analysis/{id}/status      - Get analysis status")
    print("   • GET  /api/analysis/{id}/results     - Get analysis results")
    print("   • GET  /api/vessel/{id}/intelligence  - Get vessel intelligence")
    print("   • GET  /api/mpas                      - Get available MPAs")
    
    print("\n🔧 CONFIGURATION:")
    print("   • Edit .env file to configure API keys and settings")
    print("   • Check config.py for advanced configuration options")
    
    print("\n📚 DOCUMENTATION:")
    print("   • Backend API: Check app.py for endpoint details")
    print("   • Frontend: Check ghost-hunter-frontend/README.md")
    print("   • GenAI Integration: Check GENAI_INTELLIGENCE_GUIDE.md")
    
    print("\n" + "="*80)

def open_browser_delayed():
    """Open browser after a delay to allow servers to start"""
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:5000/api/health')
        print("🌐 Opened browser to backend health check")
    except:
        pass

def main():
    """Main application launcher"""
    print("🛰️ GHOST HUNTER - Maritime Intelligence Platform")
    print("=" * 60)
    
    # Pre-flight checks
    check_python_version()
    
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)
    
    if not check_environment():
        print("⚠️ Environment check failed - continuing with limited functionality")
    
    setup_directories()
    
    if not check_frontend():
        print("⚠️ Frontend check failed - backend will still start")
    
    # Print instructions
    print_startup_instructions()
    
    # Start browser in background
    browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
    browser_thread.start()
    
    # Start backend server (this will block)
    print("\n🚀 Starting backend server...")
    start_backend()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Ghost Hunter backend stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)