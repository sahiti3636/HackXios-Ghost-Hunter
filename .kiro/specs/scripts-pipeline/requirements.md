# Scripts Pipeline Requirements

## Functional Requirements

### FR-1: Satellite-MPA Intersection Analysis
**Priority**: Critical
**Description**: The system must determine which satellite tiles intersect with Marine Protected Areas

#### FR-1.1: Spatial Intersection Detection
- **Requirement**: Implement mathematical condition: Process tile ⟺ Area(Tile∩MPA) > 0
- **Input**: Sentinel-1 SAR TIFF files, MPA GeoJSON boundaries
- **Output**: Intersection analysis results with overlap percentages
- **Acceptance Criteria**:
  - Correctly identify all satellite tiles that overlap with MPAs
  - Calculate precise overlap areas and coverage percentages
  - Handle multiple coordinate systems (WGS84, UTM, etc.)
  - Support minimum overlap threshold filtering (default: 0.1%)

#### FR-1.2: Geographic Bounds Extraction
- **Requirement**: Extract geographic bounds from satellite data using multiple methods
- **Methods**: Direct bounds, CRS transformation, SAFE manifest parsing, TIFF geotags
- **Acceptance Criteria**:
  - Successfully extract bounds from 95% of Sentinel-1 files
  - Handle corrupted or incomplete metadata gracefully
  - Provide fallback mechanisms for edge cases

#### FR-1.3: Processing Region Extraction
- **Requirement**: Extract satellite data regions for each MPA intersection
- **Features**: Buffer zones (default: 10km), memory-efficient processing
- **Acceptance Criteria**:
  - Generate processing regions for all valid intersections
  - Apply configurable buffer zones around MPA boundaries
  - Optimize memory usage for large satellite images

### FR-2: Physics-Based Ship Detection
**Priority**: Critical
**Description**: Detect vessels using Sea-Background Contrast Index (SBCI) method

#### FR-2.1: SBCI Calculation
- **Requirement**: Implement SBCI = pixel_intensity / local_sea_background_mean
- **Algorithm**: Physics-based, explainable vessel detection
- **Acceptance Criteria**:
  - Calculate SBCI for all valid pixels in satellite image
  - Use adaptive window sizes for local background estimation
  - Handle edge cases and invalid data gracefully
  - Achieve detection threshold configurability (default: 5.0)

#### FR-2.2: Preprocessing Pipeline
- **Requirement**: Apply radiometric calibration and noise reduction
- **Steps**: Speckle filtering, local sea normalization, invalid value handling
- **Acceptance Criteria**:
  - Reduce speckle noise while preserving vessel signatures
  - Normalize for varying sea conditions and wind states
  - Handle NaN and infinite values appropriately

#### FR-2.3: Post-Processing
- **Requirement**: Clean detections and extract vessel candidates
- **Operations**: Morphological operations, connected components, size filtering
- **Acceptance Criteria**:
  - Remove noise and false positives
  - Extract vessel properties (area, centroid, backscatter statistics)
  - Apply minimum vessel size threshold (default: 3 pixels)

### FR-3: Vessel Feature Extraction and Clustering
**Priority**: High
**Description**: Extract vessel characteristics and perform unsupervised classification

#### FR-3.1: Feature Engineering
- **Requirement**: Extract physics-based vessel features
- **Features**: Area, aspect ratio, compactness, backscatter statistics, SBCI values
- **Acceptance Criteria**:
  - Calculate geometric properties for all detected vessels
  - Extract radiometric characteristics from SAR data
  - Provide spatial coordinates (pixel and geographic)

#### FR-3.2: Unsupervised Clustering
- **Requirement**: Classify vessels using K-means clustering
- **Algorithm**: Standardized features, adaptive cluster count
- **Acceptance Criteria**:
  - Perform K-means clustering with configurable cluster count (default: 3)
  - Adapt cluster count based on available vessels
  - Standardize features before clustering

#### FR-3.3: Visualization Generation
- **Requirement**: Create cluster analysis visualizations
- **Output**: SAR overlay with vessel markers and cluster assignments
- **Acceptance Criteria**:
  - Generate high-quality visualization plots
  - Show vessel locations with cluster color coding
  - Include statistics and metadata in visualizations

### FR-4: AIS Correlation and Dark Vessel Detection
**Priority**: High
**Description**: Cross-reference detected vessels with AIS data to identify dark vessels

#### FR-4.1: Spatial-Temporal Correlation
- **Requirement**: Match radar detections with AIS positions
- **Algorithm**: Haversine distance calculation with time windows
- **Parameters**: Distance threshold (2km), time threshold (60min)
- **Acceptance Criteria**:
  - Accurately calculate distances between radar and AIS positions
  - Apply temporal correlation within specified time windows
  - Handle timezone and timestamp format variations

#### FR-4.2: Dark Vessel Classification
- **Requirement**: Classify vessels as NORMAL (AIS present) or DARK (no AIS)
- **Logic**: Binary classification based on correlation results
- **Acceptance Criteria**:
  - Correctly identify vessels without corresponding AIS signals
  - Provide dark vessel percentage statistics
  - Generate detailed classification reports

#### FR-4.3: AIS Data Processing
- **Requirement**: Efficiently process large AIS datasets
- **Optimization**: Spatial filtering, memory limits, batch processing
- **Acceptance Criteria**:
  - Process AIS CSV files with memory constraints (2000 rows/file)
  - Apply spatial bounding box filtering
  - Handle multiple AIS file formats

### FR-5: CNN Dataset Generation
**Priority**: Medium
**Description**: Generate training dataset for CNN vessel verification

#### FR-5.1: Patch Extraction
- **Requirement**: Extract 64x64 image patches from SAR data
- **Types**: Ship patches (centered on detections), sea patches (vessel-free areas)
- **Acceptance Criteria**:
  - Generate one ship patch per detected vessel
  - Create fixed number of sea patches per satellite (10)
  - Apply exclusion zones around vessels for sea patches

#### FR-5.2: Dataset Labeling
- **Requirement**: Create comprehensive labels with metadata
- **Metadata**: AIS status, cluster ID, vessel properties, coordinates
- **Acceptance Criteria**:
  - Generate JSON labels file with all vessel metadata
  - Support incremental dataset building
  - Include geographic and pixel coordinates

#### FR-5.3: Data Normalization
- **Requirement**: Normalize patches for CNN training
- **Method**: Percentile-based normalization to [0,1] range
- **Acceptance Criteria**:
  - Apply robust normalization handling outliers
  - Convert to 8-bit PNG format for storage
  - Handle NaN values appropriately

## Non-Functional Requirements

### NFR-1: Performance
- **Memory Usage**: Handle satellite images up to 10GB with automatic downsampling
- **Processing Time**: Complete pipeline execution within 30 minutes per satellite
- **Scalability**: Support parallel processing of multiple satellites

### NFR-2: Reliability
- **Error Handling**: Graceful degradation with comprehensive error logging
- **Data Validation**: Input validation for all file formats and parameters
- **Recovery**: Fallback mechanisms for corrupted or missing data

### NFR-3: Maintainability
- **Code Quality**: Modular architecture with clear separation of concerns
- **Documentation**: Comprehensive docstrings and inline comments
- **Testing**: Support for unit testing and integration testing

### NFR-4: Usability
- **Command Line Interface**: Consistent argument parsing across all scripts
- **Progress Reporting**: Detailed progress indicators and status messages
- **Output Organization**: Structured output files with satellite-specific naming

### NFR-5: Compatibility
- **File Formats**: Support for TIFF, NetCDF, GeoJSON, CSV formats
- **Coordinate Systems**: Handle multiple CRS with automatic transformation
- **Platform Independence**: Cross-platform compatibility (Windows, Linux, macOS)

## Data Requirements

### DR-1: Input Data Specifications
- **Satellite Data**: Sentinel-1 SAR TIFF files with VV polarization
- **MPA Boundaries**: GeoJSON format with polygon geometries
- **AIS Data**: CSV files with timestamp, position, and MMSI columns

### DR-2: Output Data Specifications
- **JSON Results**: Structured data with satellite-specific naming
- **Visualizations**: High-resolution PNG plots (300 DPI)
- **CNN Dataset**: 64x64 grayscale PNG patches with JSON labels

### DR-3: Data Quality Requirements
- **Completeness**: Handle missing or incomplete data gracefully
- **Accuracy**: Maintain spatial accuracy within 100m for vessel positions
- **Consistency**: Standardized output formats across all satellites

## Integration Requirements

### IR-1: Pipeline Integration
- **Sequential Processing**: Each script consumes output from previous stage
- **Error Propagation**: Failed stages should not break downstream processing
- **Data Validation**: Validate input data before processing

### IR-2: External Dependencies
- **Geospatial Libraries**: Rasterio, Shapely, GeoPandas integration
- **Scientific Computing**: NumPy, SciPy, scikit-learn compatibility
- **Visualization**: Matplotlib, OpenCV integration

### IR-3: Configuration Management
- **Parameter Configuration**: Configurable thresholds and parameters
- **Environment Variables**: Support for environment-based configuration
- **Default Values**: Sensible defaults for all parameters

## Security Requirements

### SR-1: Data Protection
- **Input Validation**: Sanitize all input files and parameters
- **Path Security**: Prevent directory traversal attacks
- **Memory Safety**: Prevent buffer overflows and memory leaks

### SR-2: Access Control
- **File Permissions**: Respect system file permissions
- **Output Security**: Create output files with appropriate permissions
- **Temporary Files**: Secure handling of temporary data

## Compliance Requirements

### CR-1: Data Standards
- **Geospatial Standards**: Comply with OGC standards for spatial data
- **Metadata Standards**: Include comprehensive metadata in outputs
- **Format Standards**: Use standard file formats for interoperability

### CR-2: Scientific Reproducibility
- **Deterministic Results**: Ensure reproducible results with same inputs
- **Version Control**: Track algorithm versions and parameters
- **Documentation**: Maintain detailed processing logs