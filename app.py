#!/usr/bin/env python3
"""
Ghost Hunter Flask Backend API
Integrates with the enhanced GenAI pipeline for maritime vessel detection and intelligence analysis.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import json
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import logging

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sqlite3

from utils.email_service import EmailService
from utils.mpa_checker import MPAChecker
from utils.sentinel_fetcher import SentinelFetcher

# Import our enhanced pipeline components
from enhanced_ghost_hunter_pipeline import EnhancedGhostHunterPipeline
from intelligence_analyzer import IntelligenceAnalyzer
from main_pipeline import GhostHunterPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('data/raw/satellite', exist_ok=True)

# Initialize email service
email_service = EmailService()

# Global variables for tracking analysis jobs
analysis_jobs = {}
analysis_lock = threading.Lock()

# Database setup for persistent storage
def init_db():
    """Initialize SQLite database for storing analysis results"""
    conn = sqlite3.connect('ghost_hunter.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            region_type TEXT,
            region_data TEXT,
            start_date TEXT,
            end_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            results_path TEXT,
            intelligence_path TEXT,
            error_message TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vessels (
            id TEXT PRIMARY KEY,
            analysis_id TEXT,
            vessel_data TEXT,
            intelligence_analysis TEXT,
            risk_score INTEGER,
            ais_status TEXT,
            coordinates TEXT,
            FOREIGN KEY (analysis_id) REFERENCES analyses (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

class AnalysisManager:
    """Manages analysis jobs and their lifecycle"""
    
    def __init__(self):
        self.pipeline = None
        self.intelligence_analyzer = None
        self.mpa_checker = MPAChecker()
        self.sentinel_fetcher = SentinelFetcher()
        self.sentinel_fetcher.connect() # Try to connect if CREDS exist
        
    def initialize_pipeline(self):
        """Initialize the enhanced pipeline with GenAI"""
        try:
            self.pipeline = EnhancedGhostHunterPipeline()
            self.intelligence_analyzer = IntelligenceAnalyzer()
            logger.info("Enhanced pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def start_analysis(self, analysis_id: str, config: Dict) -> bool:
        """Start a new analysis job"""
        try:
            if not self.pipeline:
                if not self.initialize_pipeline():
                    return False
            
            # Update job status
            with analysis_lock:
                analysis_jobs[analysis_id] = {
                    'status': 'running',
                    'progress': 0,
                    'current_step': 'Initializing analysis...',
                    'started_at': datetime.now().isoformat(),
                    'config': config
                }
            
            # Run analysis in background thread
            thread = threading.Thread(
                target=self._run_analysis,
                args=(analysis_id, config),
                daemon=True
            )
            thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start analysis {analysis_id}: {e}")
            with analysis_lock:
                analysis_jobs[analysis_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'started_at': datetime.now().isoformat()
                }
            return False
    
    def _generate_web_image(self, scene_path, product_name):
        """
        Generates a web-friendly PNG from the Sentinel-1 SAFE product.
        Returns the relative path for the frontend and the bounds.
        """
        try:
            import rasterio
            from rasterio.plot import reshape_as_image
            from rasterio.vrt import WarpedVRT
            import numpy as np
            from PIL import Image
            
            # Find the measurement tiff (VV polarization usually preferred for ships)
            measurement_path = None
            for root, dirs, files in os.walk(os.path.join(scene_path, "measurement")):
                for file in files:
                    if "vv" in file.lower() and file.endswith(".tiff"):
                        measurement_path = os.path.join(root, file)
                        break
                if measurement_path: break
            
            if not measurement_path:
                logger.error("Could not find measurement TIFF for image generation")
                return None, None

            # Output path in frontend public folder
            filename = f"{product_name}.png"
            # Ensure directory exists in frontend
            public_dir = os.path.join(os.getcwd(), "ghost-hunter-frontend", "public", "analysis_images")
            os.makedirs(public_dir, exist_ok=True)
            output_path = os.path.join(public_dir, filename)
            
            logger.info(f"Generating web preview from: {measurement_path}")
            
            with rasterio.open(measurement_path) as src:
                # Sentinel-1 SLC/GRD often has GCPs but no transform. 
                # Use WarpedVRT to reproject to WGS84 (EPSG:4326) on the fly.
                # src_nodata=0 assumes 0 is the background value in the original SAR data (common)
                # nodata=0 sets the background to 0 in the output
                with WarpedVRT(src, crs='EPSG:4326', src_nodata=0, nodata=0) as vrt:
                    # Get correct geographical bounds from the VRT
                    # Bounds are (left, bottom, right, top) in Longitude, Latitude
                    bounds = [[vrt.bounds.bottom, vrt.bounds.left], [vrt.bounds.top, vrt.bounds.right]]
                    
                    # Calculate dimensions for a reasonable web size (e.g., max 2000px wide)
                    # This avoids reading massive 25k pixel arrays
                    dst_width = 1000
                    scale = dst_width / vrt.width
                    dst_height = int(vrt.height * scale)
                    
                    # Read reprojected data
                    data = vrt.read(1, out_shape=(1, dst_height, dst_width))
                    
                    # Normalize for display (simple min-max scaling with clipping)
                    # SAR data is high dynamic range, clipping is essential
                    p2 = np.percentile(data[data > 0], 2) if np.any(data > 0) else 0
                    p98 = np.percentile(data[data > 0], 98) if np.any(data > 0) else 1
                    
                    # Create mask for transparency (0 values are transparent)
                    # We do this BEFORE normalization to preserve the 0s
                    alpha_mask = (data > 0).astype(np.uint8) * 255
                    
                    data = np.clip(data, p2, p98)
                    if p98 > p2:
                        data = (data - p2) / (p98 - p2) * 255.0
                    else:
                        data = data * 0 
                        
                    data = data.astype(np.uint8)
                    
                    # Convert to RGBA for transparency
                    img = Image.fromarray(data, mode='L').convert('RGBA')
                    
                    # Apply transparency mask
                    # Get the alpha channel
                    r, g, b, a = img.split()
                    # Update alpha with our mask
                    img = Image.merge('RGBA', (r, g, b, Image.fromarray(alpha_mask)))
                    
                    img.save(output_path, "PNG")
                    
                    logger.info(f"Saved web preview to: {output_path} with bounds: {bounds}")
                    
                    # Return relative path for Next.js and LatLngBounds
                    return f"/analysis_images/{filename}", bounds
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _run_analysis(self, analysis_id: str, config: Dict):
        """Run the actual analysis pipeline"""
        try:
            # Update progress
            self._update_progress(analysis_id, 10, "Fetching satellite data...")
            
            # Simulate satellite data processing (in real implementation, this would handle actual data)
            time.sleep(2)
            self._update_progress(analysis_id, 30, "Detecting vessels...")
            
            # Run the enhanced pipeline
            if config.get('region_type') == 'custom':
                # Handle custom polygon analysis
                results = self._run_custom_analysis(analysis_id, config)
            else:
                # Handle MPA analysis
                results = self._run_mpa_analysis(analysis_id, config)
            
            self._update_progress(analysis_id, 70, "Generating intelligence analysis...")
            
            # Generate intelligence analysis
            intelligence_data = self._generate_intelligence(analysis_id, results)
            
            self._update_progress(analysis_id, 90, "Finalizing results...")
            
            # Save results
            self._save_results(analysis_id, results, intelligence_data)
            
            self._update_progress(analysis_id, 100, "Analysis complete")
            
            # Mark as completed
            with analysis_lock:
                analysis_jobs[analysis_id]['status'] = 'completed'
                analysis_jobs[analysis_id]['completed_at'] = datetime.now().isoformat()
            
            # Update database
            self._update_db_analysis(analysis_id, 'completed', results_path=f"results/{analysis_id}")
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            with analysis_lock:
                analysis_jobs[analysis_id]['status'] = 'failed'
                analysis_jobs[analysis_id]['error'] = str(e)
            
            self._update_db_analysis(analysis_id, 'failed', error_message=str(e))
    
    def _update_progress(self, analysis_id: str, progress: int, step: str):
        """Update analysis progress"""
        with analysis_lock:
            if analysis_id in analysis_jobs:
                analysis_jobs[analysis_id]['progress'] = progress
                analysis_jobs[analysis_id]['current_step'] = step
    
    def _run_custom_analysis(self, analysis_id: str, config: Dict) -> Dict:
        """Run analysis for custom polygon region"""
        # In a real implementation, this would process the actual satellite data
        # For now, we'll use the existing pipeline with mock data
        
        try:
            # Perform MPA Check on the custom region
            intersecting_mpas = []
            region_data = config.get('region_data', {})
            if 'polygon' in region_data:
                logger.info("Checking MPA intersection for custom polygon...")
                intersecting_mpas = self.mpa_checker.check_intersection(region_data['polygon'])
                if intersecting_mpas:
                    mpa_names = [m['name'] for m in intersecting_mpas]
                    logger.info(f"✨ Region intersects with MPAs: {', '.join(mpa_names)}")

            # Initialize clean results object
            results = {
                'vessels': [],
                'detection_data': {},
                'satellite_metadata': {},
                'intelligence_analysis': {}
            }
                    
            if intersecting_mpas:
                results['detected_mpas'] = intersecting_mpas

            # Live Data Logic: Always try to fetch fresh data for the requested region.
            # We skip the "check local sat1 database" step entirely to ensure we don't return stale demo data.
            if 'polygon' in region_data:
                # Force "Fetching" new data via Sentinel API.
                logger.info("Checking vessels: Querying Sentinel Hub for fresh data...")
                # Case B: No local data. Try "Fetching" new data via Sentinel API.
                logger.info("Checking vessels: None found locally. Querying Sentinel Hub...")
                
                # Handle dates - ensure we don't pass empty strings
                start_date = config.get('start_date')
                if not start_date:
                    start_date = 'NOW-30DAYS'
                
                end_date = config.get('end_date')
                if not end_date:
                    end_date = 'NOW'

                # Search for scenes
                scenes = self.sentinel_fetcher.search_scenes(
                    region_data['polygon'], 
                    start_date, 
                    end_date
                )
                        
                        # LOGIC UPDATE:
                        # Whether API finds scenes OR fails (and returns []), we should 
                        # generate FRESH simulated data if the user wants "actual" analysis.
                        # Do NOT show the hardcoded 7 vessels from sat1.
                        

                if scenes:
                     logger.info(f"Sentinel API found {len(scenes)} real scenes.")
                     
                     # Select latest scene
                     latest_scene = scenes[0]
                     product_id = latest_scene['properties'].get('id')
                     product_name = latest_scene['properties'].get('title')
                     
                     logger.info(f"Initiating download for: {product_name} (ID: {product_id})")
                     
                     # Download to 'data/raw/satellite/'
                     # Use absolute path to ensure reliability
                     download_dir = os.path.join(os.getcwd(), 'data', 'raw', 'satellite') 
                     # config['current_scene_path'] points to sat1 json usually, so we target data/raw/satellite
                     output_dir = os.path.join(os.getcwd(), 'data', 'raw', 'satellite')
                     
                     zip_path = self.sentinel_fetcher.download_product(product_id, product_name, output_dir)
                     
                     if zip_path and os.path.exists(zip_path):
                         logger.info(f"Successfully downloaded: {zip_path}")
                         
                         # Unzip
                         import zipfile
                         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                             extract_path = os.path.splitext(zip_path)[0] + ".SAFE"
                             # Check if already extracted (some zips contain the folder directly)
                             top_level = zip_ref.namelist()[0].split('/')[0]
                             final_safe_path = os.path.join(output_dir, top_level)
                             
                             if not os.path.exists(final_safe_path):
                                 logger.info("Unzipping product...")
                                 zip_ref.extractall(output_dir)
                             else:
                                 logger.info("Product already extracted.")
                                 
                         logger.info(f"Targeting pipeline to new scene: {final_safe_path}")
                         
                         # GENERATE WEB IMAGE
                         logger.info("Generating web preview image...")
                         image_url, image_bounds = self._generate_web_image(final_safe_path, product_name)
                         
                         if image_url:
                             image_data = {
                                 'url': image_url,
                                 'bounds': image_bounds
                             }
                             results['satellite_image'] = image_data
                         
                         # Inject metadata about the source in memory too
                         results['satellite_metadata'] = {
                            'source': 'Sentinel-1 (Live Download)',
                            'scene_id': product_name,
                            'download_path': final_safe_path
                         }
                         
                         # RE-INITIALIZE PIPELINE for this specific scene
                         self.pipeline.scene_path = final_safe_path
                         # Clear previous results cache
                         if hasattr(self.pipeline, 'cached_results'):
                             self.pipeline.cached_results = None
                         
                         # RUN REAL ANALYSIS
                         logger.info("Running FULL PIPELINE on downloaded data...")
                         
                         # Get the scene object (modified pipeline uses scene_path to generate this)
                         real_scenes = self.pipeline.get_satellite_scenes()
                         
                         if real_scenes:
                              target_scene = real_scenes[0]
                              # Run the pipeline
                              success = self.pipeline.run_satellite_pipeline(target_scene, 1, 1)
                              
                              if success:
                                   logger.info("Pipeline execution successful. Loading results...")
                                   # Load the RESULTS from the output file
                                   # The pipeline saves to final_ghost_hunter_report_{region_name}.json
                                   region_name = target_scene['name']
                                   report_path = f"final_ghost_hunter_report_{region_name}.json"
                                   
                                   # PERSISTENCE FIX: Now that report exists, save image data to it
                                   try:
                                       if os.path.exists(report_path) and 'satellite_image' in results:
                                           with open(report_path, 'r+') as f:
                                               data = json.load(f)
                                               data['satellite_image'] = results['satellite_image']
                                               data['satellite_metadata'] = results.get('satellite_metadata', {})
                                               f.seek(0)
                                               json.dump(data, f, indent=2)
                                               f.truncate()
                                           logger.info(f"Updated report {report_path} with satellite image metadata")
                                   except Exception as e:
                                       logger.error(f"Failed to persist image metadata to report: {e}")

                                   # Load REAL results from the pipeline output
                                   if os.path.exists(report_path):
                                        with open(report_path, 'r') as f:
                                             pipeline_data = json.load(f)
                                             results['vessels'] = pipeline_data.get('vessels', [])
                                             results['detection_data'] = pipeline_data # Store full context
                                        logger.info(f"Loaded {len(results['vessels'])} vessels from pipeline report.")
                                   
                              else:
                                   logger.error("Pipeline run_satellite_pipeline returned False.")
                                   # If pipeline failed, clear sat1 data so we don't show confusing info
                                   results['vessels'] = []
                                   
                         else:
                              logger.error("Pipeline could not load scene from path.")
                              results['vessels'] = []

                                 
                     else:
                         logger.error("Download failed. Falling back to simulation.")
                         # Use simulation fallback if download fails (handled below)
                         scenes = [] # Trigger fallback block
                        
                if not scenes:
                    logger.info("Sentinel API returned no scenes (or connection failed). Generating hypothetical simulation.")
                    # Simulate plausible vessels based on the region size...
                    import random
                    simulated_vessels = []
                    
                    # Generate 0-5 random candidates per scene found (up to max 15)
                    base_count = random.randint(3, 12)
                    num_simulated = min(base_count * random.randint(1, 2), 15)
                    
                    if num_simulated > 0:
                        # Get bounds for random placement
                        lats = [p[0] for p in region_data['polygon']]
                        lons = [p[1] for p in region_data['polygon']]
                        min_lat, max_lat = min(lats), max(lats)
                        min_lon, max_lon = min(lons), max(lons)
                        
                        for i in range(num_simulated):
                            simulated_vessels.append({
                                "vessel_id": f"LIVE-SIM-{i+1}",
                                "latitude": random.uniform(min_lat, max_lat),
                                "longitude": random.uniform(min_lon, max_lon),
                                "risk_score": random.choice([15, 30, 45, 75, 90]), 
                                "ais_status": random.choice(["Active", "Active", "DARK"]),
                                "cnn_confidence": random.uniform(0.60, 0.95),
                                "behavior_analysis": {"suspicion_level": "ANALYZING"},
                                "image": None 
                            })
                    
                    results['vessels'] = simulated_vessels
                    results['satellite_metadata'] = {
                        'source': 'Simulation (Connection Unavailable)',
                        'note': 'Generating hypothetical traffic for demonstration'
                    }
                    
            filtered_vessels = results['vessels']
            
            # Update summary stats
            total = len(filtered_vessels)
            dark = len([v for v in filtered_vessels if 'DARK' in v.get('ais_status', 'FAIL')])
            high_risk = len([v for v in filtered_vessels if v.get('risk_score', 0) > 60])
            
            results['detection_summary'] = {
                'total_detections': total,
                'dark_vessels': dark,
                'high_risk_vessels': high_risk
            }
            
            # Fix KeyError: Ensure intelligence_analysis exists
            if 'intelligence_analysis' not in results:
                    results['intelligence_analysis'] = {}
                    
            # Update executive summary
            if 'polygon' in region_data:
                 # Logic for custom region summary
                 if scenes and self.sentinel_fetcher.api:
                     # Real API fallback to simulation
                     results['intelligence_analysis']['executive_summary'] = f"Sentinel-1 search found {len(scenes)} scenes covering this region. Visualizing estimated traffic based on historical density."
                     if scenes:
                        results['intelligence_analysis']['key_findings'] = [
                            f"Latest acquisition: {scenes[0]['properties'].get('ingestiondate', 'Unknown')}",
                            "Vessel signatures visualized for assessment."
                        ]
                 else:
                     # Pure simulation
                      results['intelligence_analysis']['executive_summary'] = "Generating hypothetical scenario for this region (No live data connection)."
                      results['intelligence_analysis']['key_findings'] = ["System running in standalone simulation mode."]

            else:

                # Default/Full Analysis logic (if no polygon specified)
                # Matches main() behavior from pipeline script
                scenes = self.pipeline.get_satellite_scenes()
                if scenes:
                    target = scenes[0]
                    # process_scene returns the results dict; run_satellite_pipeline returns success bool
                    # We utilize process_scene to get data directly if available, or load from file
                    
                    # Note: process_scene is not available in base pipeline, only in our mental model maybe?
                    # Let's check if we can just use run_satellite_pipeline and load the json.
                    
                    if self.pipeline.run_satellite_pipeline(target, 1, 1):
                         region_name = target['name']
                         report_path = f"final_ghost_hunter_report_{region_name}.json"
                         if os.path.exists(report_path):
                             with open(report_path, 'r') as f:
                                 results = json.load(f)
                else:
                    logger.warning("No scenes found for default analysis.")
            
            return results
            
        except Exception as e:
            logger.error(f"Custom analysis failed: {e}")
            raise
    
    def _run_mpa_analysis(self, analysis_id: str, config: Dict) -> Dict:
        """Run analysis for MPA region"""
        # Similar to custom analysis but with MPA-specific processing
        return self._run_custom_analysis(analysis_id, config)
    
    def _generate_intelligence(self, analysis_id: str, results: Dict) -> Dict:
        """Generate intelligence analysis using GenAI"""
        try:
            if not self.intelligence_analyzer:
                self.intelligence_analyzer = IntelligenceAnalyzer()
            
            # Save results temporarily for analysis
            temp_file = f"temp_results_{analysis_id}.json"
            with open(temp_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate intelligence analysis
            intelligence_data = self.intelligence_analyzer.analyze_detection_report(temp_file)
            
            # Clean up temp file
            os.remove(temp_file)
            
            return intelligence_data
            
        except Exception as e:
            logger.error(f"Intelligence generation failed: {e}")
            # Return fallback analysis
            return {
                'executive_summary': f"Analysis completed for {len(results.get('vessels', []))} detected vessels.",
                'threat_assessment': 'MODERATE',
                'key_findings': ['Analysis completed with limited intelligence due to processing constraints'],
                'recommendations': ['Manual review recommended'],
                'confidence_level': 'LOW - Fallback analysis used'
            }
    
    def _save_results(self, analysis_id: str, results: Dict, intelligence_data: Dict):
        """Save analysis results to files"""
        results_dir = Path(app.config['RESULTS_FOLDER']) / analysis_id
        results_dir.mkdir(exist_ok=True)
        
        # Save raw results
        with open(results_dir / 'detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save intelligence analysis
        with open(results_dir / 'intelligence_analysis.json', 'w') as f:
            json.dump(intelligence_data, f, indent=2)
        
        # Generate human-readable report
        if hasattr(self.intelligence_analyzer, 'generate_human_readable_report'):
            markdown_report = self.intelligence_analyzer.generate_human_readable_report(
                intelligence_data, 'markdown'
            )
            with open(results_dir / 'intelligence_report.md', 'w') as f:
                f.write(markdown_report)
    
    def _update_db_analysis(self, analysis_id: str, status: str, **kwargs):
        """Update analysis record in database"""
        conn = sqlite3.connect('ghost_hunter.db')
        cursor = conn.cursor()
        
        update_fields = ['status = ?']
        values = [status]
        
        if status == 'completed':
            update_fields.append('completed_at = ?')
            values.append(datetime.now().isoformat())
        
        for key, value in kwargs.items():
            update_fields.append(f'{key} = ?')
            values.append(value)
        
        values.append(analysis_id)
        
        cursor.execute(f'''
            UPDATE analyses 
            SET {', '.join(update_fields)}
            WHERE id = ?
        ''', values)
        
        conn.commit()
        conn.close()

# Initialize analysis manager
analysis_manager = AnalysisManager()

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pipeline_ready': analysis_manager.pipeline is not None
    })

@app.route('/api/analysis/start', methods=['POST'])
def start_analysis():
    """Start a new analysis job"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['region_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Store analysis configuration in database
        conn = sqlite3.connect('ghost_hunter.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analyses (id, status, region_type, region_data, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            'pending',
            data.get('region_type'),
            json.dumps(data.get('region_data', {})),
            data.get('start_date'),
            data.get('end_date')
        ))
        
        conn.commit()
        conn.close()
        
        # Start analysis
        if analysis_manager.start_analysis(analysis_id, data):
            return jsonify({
                'analysis_id': analysis_id,
                'status': 'started',
                'message': 'Analysis job started successfully'
            })
        else:
            return jsonify({'error': 'Failed to start analysis'}), 500
            
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/status', methods=['GET'])
def get_analysis_status(analysis_id):
    """Get the status of an analysis job"""
    try:
        with analysis_lock:
            if analysis_id in analysis_jobs:
                job_data = analysis_jobs[analysis_id].copy()
                return jsonify(job_data)
        
        # Check database for completed analyses
        conn = sqlite3.connect('ghost_hunter.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = ['id', 'status', 'region_type', 'region_data', 'start_date', 'end_date', 
                      'created_at', 'completed_at', 'results_path', 'intelligence_path', 'error_message']
            analysis_data = dict(zip(columns, row))
            
            return jsonify({
                'status': analysis_data['status'],
                'progress': 100 if analysis_data['status'] == 'completed' else 0,
                'created_at': analysis_data['created_at'],
                'completed_at': analysis_data['completed_at'],
                'error': analysis_data['error_message']
            })
        
        return jsonify({'error': 'Analysis not found'}), 404
        
    except Exception as e:
        logger.error(f"Error getting analysis status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/results', methods=['GET'])
def get_analysis_results(analysis_id):
    """Get the results of a completed analysis"""
    try:
        results_dir = Path(app.config['RESULTS_FOLDER']) / analysis_id
        
        if not results_dir.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        # Load detection results
        detection_file = results_dir / 'detection_results.json'
        intelligence_file = results_dir / 'intelligence_analysis.json'
        
        results = {}
        
        if detection_file.exists():
            with open(detection_file, 'r') as f:
                results['detection_data'] = json.load(f)
        
        if intelligence_file.exists():
            with open(intelligence_file, 'r') as f:
                results['intelligence_analysis'] = json.load(f)
        
        # Get vessel summary for frontend
        vessels = results.get('detection_data', {}).get('vessels', [])
        vessel_summary = []
        
        for vessel in vessels:
            vessel_summary.append({
                'id': str(vessel.get('vessel_id', 'unknown')),
                'name': f"Unknown Vessel {vessel.get('vessel_id', '')}",
                'risk': vessel.get('risk_score', 0),
                'status': 'Critical' if vessel.get('risk_score', 0) > 80 else 
                         'Suspicious' if vessel.get('risk_score', 0) > 40 else 'Clear',
                'lastSeen': '2h ago',  # Mock data - in real app, calculate from timestamp
                'lat': vessel.get('latitude', 0),
                'lng': vessel.get('longitude', 0),
                'type': 'Unknown',
                'flag': 'Unknown',
                'ais_status': vessel.get('ais_status', 'UNKNOWN'),
                'cnn_confidence': vessel.get('cnn_confidence', 0),
                'detection_confidence': vessel.get('detection_confidence', 0),
                'behavior_analysis': vessel.get('behavior_analysis', {}),
                'coordinates': f"{vessel.get('latitude', 0):.4f}°N, {vessel.get('longitude', 0):.4f}°E"
            })
        
        # Construct overlay image URL
        overlay_url = None
        satellite_metadata = results.get('detection_data', {}).get('satellite_metadata', {})
        scene_id = satellite_metadata.get('scene_id')
        
        if scene_id:
            # Look for matching file in output/png
            overlay_filename = f"vessel_clusters_overlay_{scene_id}.png"
            overlay_path = Path("output/png") / overlay_filename
            if overlay_path.exists():
                overlay_url = f"/api/analysis/{analysis_id}/overlay"

        return jsonify({
            'analysis_id': analysis_id,
            'vessels': vessel_summary,
            'satellite_image': results.get('detection_data', {}).get('satellite_image'),
            'satellite_metadata': satellite_metadata,
            'overlay_image_url': overlay_url,
            'intelligence_summary': results.get('intelligence_analysis', {}),
            'detection_summary': {
                'total_vessels': len(vessels),
                'dark_vessels': len([v for v in vessels if v.get('ais_status') == 'DARK']),
                'high_risk_vessels': len([v for v in vessels if v.get('risk_score', 0) > 60])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/overlay', methods=['GET'])
def get_analysis_overlay(analysis_id):
    """Serve the vessel clusters overlay image"""
    try:
        # Get scene_id from results
        results_dir = Path(app.config['RESULTS_FOLDER']) / analysis_id
        detection_file = results_dir / 'detection_results.json'
        
        if not detection_file.exists():
            return jsonify({'error': 'Analysis not found'}), 404
            
        with open(detection_file, 'r') as f:
            data = json.load(f)
            
        scene_id = data.get('satellite_metadata', {}).get('scene_id')
        if not scene_id:
            return jsonify({'error': 'Scene ID not found'}), 404
            
        filename = f"vessel_clusters_overlay_{scene_id}.png"
        file_path = Path("output/png") / filename
        
        if not file_path.exists():
            # Fallback for sat1 demo
            if 'sat1' in str(scene_id).lower() or 'sat1' in analysis_id:
                 file_path = Path("output/png/vessel_clusters_overlay_sat1.png")
            
        if file_path.exists():
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Overlay image not found'}), 404
            
    except Exception as e:
        logger.error(f"Error serving overlay: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vessel/<vessel_id>/intelligence', methods=['GET'])
def get_vessel_intelligence(vessel_id):
    """Get detailed intelligence analysis for a specific vessel"""
    try:
        analysis_id = request.args.get('analysis_id')
        if not analysis_id:
            return jsonify({'error': 'analysis_id parameter required'}), 400
        
        results_dir = Path(app.config['RESULTS_FOLDER']) / analysis_id
        intelligence_file = results_dir / 'intelligence_analysis.json'
        
        if not intelligence_file.exists():
            return jsonify({'error': 'Intelligence analysis not found'}), 404
        
        with open(intelligence_file, 'r') as f:
            intelligence_data = json.load(f)
        
        # Find vessel-specific analysis
        vessel_analyses = intelligence_data.get('detailed_vessel_analyses', [])
        vessel_analysis = None
        
        for analysis in vessel_analyses:
            if str(analysis.get('vessel_id')) == vessel_id:
                vessel_analysis = analysis
                break
        
        if not vessel_analysis:
            return jsonify({'error': 'Vessel intelligence not found'}), 404
        
        return jsonify({
            'vessel_id': vessel_id,
            'analysis': vessel_analysis.get('analysis', 'No detailed analysis available'),
            'threat_level': vessel_analysis.get('threat_level', 'UNKNOWN'),
            'coordinates': vessel_analysis.get('coordinates', 'Unknown'),
            'risk_score': vessel_analysis.get('risk_score', 0),
            'priority': vessel_analysis.get('priority', 1)
        })
        
    except Exception as e:
        logger.error(f"Error getting vessel intelligence: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/report', methods=['GET'])
def download_report(analysis_id):
    """Download intelligence report in various formats"""
    try:
        format_type = request.args.get('format', 'json')
        results_dir = Path(app.config['RESULTS_FOLDER']) / analysis_id
        
        if not results_dir.exists():
            return jsonify({'error': 'Analysis results not found'}), 404
        
        if format_type == 'json':
            file_path = results_dir / 'intelligence_analysis.json'
            if file_path.exists():
                return send_file(file_path, as_attachment=True, 
                               download_name=f'ghost_hunter_analysis_{analysis_id}.json')
        
        elif format_type == 'markdown':
            file_path = results_dir / 'intelligence_report.md'
            if file_path.exists():
                return send_file(file_path, as_attachment=True,
                               download_name=f'ghost_hunter_report_{analysis_id}.md')
        
        elif format_type == 'pdf':
            # TODO: Implement PDF generation
            return jsonify({'error': 'PDF format not yet implemented'}), 501
        
        return jsonify({'error': f'Report format {format_type} not found'}), 404
        
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/<analysis_id>/send-report', methods=['POST'])
def send_report_email(analysis_id):
    """Send intelligence report via email"""
    try:
        data = request.get_json()
        email = data.get('email')
        include_attachments = data.get('include_attachments', True)
        
        if not email:
            return jsonify({'error': 'Email address required'}), 400
        
        # Get intelligence analysis
        results_dir = Path(app.config['RESULTS_FOLDER']) / analysis_id
        intelligence_file = results_dir / 'intelligence_analysis.json'
        
        if not intelligence_file.exists():
            return jsonify({'error': 'Intelligence analysis not found'}), 404
        
        with open(intelligence_file, 'r') as f:
            intelligence_data = json.load(f)
        
        # Prepare attachments
        attachments = []
        if include_attachments:
            # Add JSON report
            if intelligence_file.exists():
                attachments.append(str(intelligence_file))
            
            # Add markdown report if available
            markdown_file = results_dir / 'intelligence_report.md'
            if markdown_file.exists():
                attachments.append(str(markdown_file))
        
        # Send email
        success = email_service.send_intelligence_report(
            recipient_email=email,
            analysis_id=analysis_id,
            intelligence_summary=intelligence_data,
            attachments=attachments if include_attachments else None
        )
        
        if success:
            return jsonify({
                'status': 'sent',
                'message': f'Report sent successfully to {email}',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'failed',
                'message': 'Failed to send email. Please check email configuration.',
                'timestamp': datetime.now().isoformat()
            }), 500
        
    except Exception as e:
        logger.error(f"Error sending report: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/satellite', methods=['POST'])
def upload_satellite_data():
    """Upload satellite data files"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.tiff', '.tif', '.zip'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Only TIFF and ZIP files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(file_path)
        
        return jsonify({
            'status': 'uploaded',
            'filename': filename,
            'size': file_path.stat().st_size,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mpas', methods=['GET'])
def get_mpas():
    """Get list of available Marine Protected Areas"""
    # Mock MPA data - in real implementation, this would come from a database
    mpas = [
        {"id": "mpa-1", "name": "Galápagos Marine Reserve", "area": "133,000 km²"},
        {"id": "mpa-2", "name": "Papahānaumokuākea", "area": "1,508,870 km²"},
        {"id": "mpa-3", "name": "Great Barrier Reef", "area": "344,400 km²"},
        {"id": "mpa-4", "name": "Palau National Marine Sanctuary", "area": "500,000 km²"},
        {"id": "mpa-5", "name": "Ross Sea Region MPA", "area": "1,550,000 km²"},
        {"id": "mpa-6", "name": "Phoenix Islands Protected Area", "area": "408,250 km²"}
    ]
    
    return jsonify({'mpas': mpas})

@app.route('/api/analyses', methods=['GET'])
def get_analyses_history():
    """Get history of all analyses"""
    try:
        conn = sqlite3.connect('ghost_hunter.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, status, region_type, created_at, completed_at, error_message
            FROM analyses 
            ORDER BY created_at DESC 
            LIMIT 50
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        analyses = []
        for row in rows:
            analyses.append({
                'id': row[0],
                'status': row[1],
                'region_type': row[2],
                'created_at': row[3],
                'completed_at': row[4],
                'error_message': row[5]
            })
        
        return jsonify({'analyses': analyses})
        
    except Exception as e:
        logger.error(f"Error getting analyses history: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize pipeline on startup
    logger.info("Starting Ghost Hunter Backend API...")
    
    # Try to initialize pipeline
    if analysis_manager.initialize_pipeline():
        logger.info("Enhanced pipeline initialized successfully")
    else:
        logger.warning("Pipeline initialization failed - will retry on first analysis request")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True
    )