# Technology Stack & Build System

## Core Technologies

### Programming Language
- **Python 3.x** - Primary development language for all components

### Key Libraries & Frameworks

#### Scientific Computing & Data Processing
- **NumPy** (>=1.21.0) - Numerical computing and array operations
- **SciPy** (>=1.7.0) - Scientific computing algorithms
- **Pandas** (>=1.3.0) - Data manipulation and analysis

#### Geospatial & Satellite Data
- **Rasterio** (>=1.2.0) - Geospatial raster data I/O and processing
- **NetCDF4** (>=1.5.0) - Network Common Data Form support
- **Shapely** (>=1.8.0) - Geometric operations
- **GeoJSON** (>=2.5.0) - Geographic data format support

#### Computer Vision & Image Processing
- **OpenCV** (>=4.5.0) - Computer vision operations
- **scikit-image** (>=0.18.0) - Image processing algorithms
- **Pillow** (>=8.3.0) - Python Imaging Library

#### Machine Learning
- **PyTorch** - Deep learning framework for CNN model training and inference
- **scikit-learn** (>=1.0.0) - Traditional ML algorithms and clustering
- **torchvision** - Computer vision utilities for PyTorch

#### Visualization
- **Matplotlib** (>=3.4.0) - Plotting and visualization
- **Seaborn** (>=0.11.0) - Statistical data visualization

## Build System & Environment

### Dependency Management
- **pip** with `requirements.txt` files
- Separate requirements for main project and hackxois submodule
- Virtual environment recommended (`venv/` directory present)

### Project Structure
- Modular architecture with separate components
- Pipeline-based execution model
- Hackathon-derived codebase (`hackxois/` subdirectory)

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r hackxois/requirements.txt
```

### Training & Model Operations
```bash
# Train CNN model
python train_cnn.py

# Run inference on trained model
python run_inference.py

# Check image processing
python check_image.py
```

### Pipeline Execution
```bash
# Run complete Ghost Hunter pipeline
python main_pipeline.py

# Run original hackxois pipeline
cd hackxois && python marine_vessel_detection_pipeline.py
```

### Data Processing
```bash
# Individual pipeline components (from hackxois/)
python scripts/sat_mpa_intersection.py
python scripts/ship_detections.py
python scripts/vessel_feature_extraction.py
python scripts/ais_detection.py
python scripts/cnn_dataset_generator.py
```

## Development Notes

### GPU Support
- PyTorch automatically detects CUDA availability
- CPU fallback supported for development/testing
- Model training benefits significantly from GPU acceleration

### Data Requirements
- Sentinel-1 SAR TIFF files in `data/raw/satellite/*/measurement/`
- MPA boundary data in `data/raw/mpa_boundaries/`
- AIS data in CSV format for cross-referencing

### Output Formats
- JSON for structured data and results
- PNG for visualizations and image patches
- PyTorch `.pth` files for trained models