#!/usr/bin/env python3
"""
Configuration settings for Ghost Hunter Backend API
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ghost-hunter-secret-key-change-in-production'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
    UPLOAD_FOLDER = 'uploads'
    RESULTS_FOLDER = 'results'
    ALLOWED_EXTENSIONS = {'.tiff', '.tif', '.zip', '.nc', '.h5'}
    
    # Database settings
    DATABASE_PATH = 'ghost_hunter.db'
    
    # Pipeline settings
    SATELLITE_DATA_PATH = 'data/raw/satellite'
    MPA_BOUNDARIES_PATH = 'data/raw/mpa_boundaries'
    OUTPUT_PATH = 'output'
    
    # GenAI settings
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    GENAI_MODEL = os.environ.get('GENAI_MODEL', 'gemini-2.5-flash')
    GENAI_TEMPERATURE = float(os.environ.get('GENAI_TEMPERATURE', '0.3'))
    GENAI_MAX_TOKENS = int(os.environ.get('GENAI_MAX_TOKENS', '4000'))
    
    # Intelligence analysis settings
    ANALYSIS_CONFIDENCE_THRESHOLD = float(os.environ.get('ANALYSIS_CONFIDENCE_THRESHOLD', '0.7'))
    MAX_VESSELS_DETAILED_ANALYSIS = int(os.environ.get('MAX_VESSELS_DETAILED_ANALYSIS', '10'))
    REPORT_OUTPUT_FORMAT = os.environ.get('REPORT_OUTPUT_FORMAT', 'markdown')
    
    # Email settings (for report sending)
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
    SMTP_USERNAME = os.environ.get('SMTP_USERNAME')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD')
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'ghost_hunter.log')
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        directories = [
            Config.UPLOAD_FOLDER,
            Config.RESULTS_FOLDER,
            Config.SATELLITE_DATA_PATH,
            Config.MPA_BOUNDARIES_PATH,
            Config.OUTPUT_PATH
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Use more secure settings in production
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-must-set-secret-key-in-production'
    
    # Stricter CORS in production
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_PATH = ':memory:'  # Use in-memory database for tests

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}