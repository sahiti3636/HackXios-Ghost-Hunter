# Scripts Pipeline Design Document

## Overview

The scripts folder contains the core pipeline components for the Ghost Hunter marine vessel detection system. This pipeline processes satellite imagery through multiple stages to detect, classify, and analyze vessels in marine protected areas (MPAs).

## Architecture

### Pipeline Flow

```
Satellite Data → MPA Intersection → Ship Detection → Feature Extraction → AIS Correlation → CNN Dataset
     ↓                ↓                   ↓               ↓                  ↓              ↓
sat_mpa_intersection  ship_detections  vessel_feature   ais_detection   cnn_dataset_generator
```

### Core Components

#### 1. Satellite-MPA Intersection (`sat_mpa_intersection.py`)
**Purpose**: Determines which satellite tiles intersect with Marine Protected Areas
**Mathematical Foundation**: Process tile ⟺ Area(Tile∩MPA) > 0

**Key Features**:
- Loads satellite tile boundaries from Sentinel-1 data
- Loads MPA boundaries from GeoJSON files
- Computes polygon-tile intersections using Shapely
- Extracts MPA-specific regions for processing
- Handles multiple coordinate systems and transformations

**Design Patterns**:
- **Geometric Processing**: Uses Shapely for polygon operations
- **Multi-format Support**: Handles TIFF, NetCDF, and manifest files
- **Adaptive Bounds Extraction**: Multiple methods for coordinate extraction
- **Memory Management**: Downsampling for large images

#### 2. Ship Detection (`ship_detections.py`)
**Purpose**: Physics-based vessel detection using Sea-Background Contrast Index (SBCI)
**Algorithm**: SBCI = pixel_intensity / local_sea_background_mean

**Processing Steps**:
1. **Radiometric Calibration**: Converts raw SAR signal → σ⁰
2. **Speckle Filtering**: Reduces radar noise using median filters
3. **Local Sea Normalization**: Computes local mean backscatter
4. **SBCI Calculation**: Ships → high SBCI, Sea clutter → low SBCI
5. **Binary Detection**: Threshold-based vessel identification
6. **Post-processing**: Morphological operations and connected components

**Design Patterns**:
- **Physics-based Detection**: Explainable vessel detection method
- **Adaptive Processing**: Auto-calculated window sizes
- **Memory Optimization**: Downsampling and chunked processing
- **Multi-format Input**: TIFF and NetCDF support

#### 3. Vessel Feature Extraction (`vessel_feature_extraction.py`)
**Purpose**: Extract vessel characteristics and perform unsupervised clustering

**Feature Set**:
- **Geometric**: Area, aspect ratio, compactness, bounding box
- **Radiometric**: Mean/max backscatter, SBCI values
- **Spatial**: Pixel and geographic coordinates

**Clustering Algorithm**:
- **K-Means**: Unsupervised vessel classification
- **Feature Standardization**: StandardScaler preprocessing
- **Adaptive Clusters**: Adjusts cluster count based on vessel count

**Design Patterns**:
- **Feature Engineering**: Physics-based vessel characteristics
- **Visualization**: SAR overlay with cluster analysis
- **Data Pipeline**: Seamless integration with detection results

#### 4. AIS Correlation (`ais_detection.py`)
**Purpose**: Cross-reference detected vessels with AIS data to identify "dark vessels"

**Correlation Logic**:
- **Spatial Matching**: Haversine distance calculation
- **Temporal Matching**: Time-based correlation windows
- **Classification**: NORMAL (AIS present) vs DARK (no AIS)

**Parameters**:
- Distance threshold: 2 km
- Time threshold: 60 minutes
- Search buffer: 1.0 degrees

**Design Patterns**:
- **Efficient Search**: Bounded spatial-temporal queries
- **Memory Management**: Limited AIS rows per file
- **Batch Processing**: Handles multiple AIS CSV files

#### 5. CNN Dataset Generator (`cnn_dataset_generator.py`)
**Purpose**: Generate CNN training dataset from physics-based detections

**Dataset Structure**:
- **Ship Patches**: 64x64 SAR patches centered on detected vessels
- **Sea Patches**: 64x64 patches from vessel-free ocean areas
- **Labels**: JSON mapping with metadata (AIS status, cluster ID, etc.)

**Generation Strategy**:
- **Fixed Sea Patches**: 10 sea patches per satellite (consistent ratio)
- **Exclusion Zones**: Avoid vessel areas for sea patches
- **Incremental Building**: Supports multi-satellite dataset growth
- **Normalization**: Percentile-based patch normalization

## Data Flow Architecture

### Input Data
- **Satellite Images**: Sentinel-1 SAR TIFF files
- **MPA Boundaries**: GeoJSON polygon files
- **AIS Data**: CSV files with vessel positions

### Intermediate Data
- **Intersection Results**: `sat_mpa_intersections_{satellite}.json`
- **Detection Results**: `ship_detection_results_{satellite}.json`
- **Feature Data**: `vessel_features_{satellite}.json`
- **AIS Status**: `ship_detection_with_ais_status_{satellite}.json`

### Output Data
- **CNN Dataset**: Image patches + labels in `cnn_dataset/`
- **Visualizations**: PNG plots and overlays
- **Statistics**: Processing metrics and summaries

## Design Principles

### 1. Satellite-Specific Processing
- Each script processes individual satellite files
- Satellite name extraction from file paths
- Output files tagged with satellite identifiers

### 2. Physics-Based Approach
- SBCI method for explainable vessel detection
- Radiometric calibration and speckle filtering
- Local sea background normalization

### 3. Modular Architecture
- Independent script execution
- JSON-based inter-module communication
- Loose coupling between components

### 4. Memory Efficiency
- Adaptive downsampling for large images
- Chunked processing for memory management
- Efficient spatial data structures

### 5. Error Handling
- Graceful degradation with fallback data
- Comprehensive error logging
- Partial pipeline execution support

## Technology Stack

### Core Libraries
- **Rasterio**: Geospatial raster I/O
- **Shapely**: Geometric operations
- **NumPy/SciPy**: Numerical computing
- **scikit-learn**: Machine learning (K-means)
- **OpenCV**: Image processing
- **Matplotlib**: Visualization

### Data Formats
- **Input**: TIFF (Sentinel-1), GeoJSON (MPAs), CSV (AIS)
- **Intermediate**: JSON (structured data)
- **Output**: PNG (images), JSON (labels), CSV (features)

## Performance Considerations

### Memory Management
- Automatic downsampling for images > 1GB
- Efficient array operations with NumPy
- Limited AIS data loading (2000 rows/file)

### Processing Optimization
- Vectorized operations for SBCI calculation
- Spatial indexing for intersection queries
- Parallel-ready architecture (satellite-specific)

### Scalability
- Incremental dataset building
- Satellite-specific output files
- Modular pipeline execution

## Quality Assurance

### Validation Methods
- Physics-based detection validation
- Statistical analysis of results
- Visual inspection through plots

### Error Recovery
- Fallback to mock data when needed
- Partial pipeline execution
- Comprehensive logging

### Testing Support
- Mock data generation
- Incremental processing
- Visualization for debugging