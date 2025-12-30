# Project Structure & Organization

## Root Directory Layout

```
ghost-hunter/
├── main_pipeline.py           # Main Ghost Hunter pipeline orchestrator
├── train_cnn.py              # CNN model training script
├── run_inference.py          # Model inference runner
├── behavior_analysis.py      # Behavioral pattern analysis
├── risk_fusion.py           # Risk scoring and fusion logic
├── ves_verification.py      # Vessel verification using CNN
├── check_image.py           # Image processing utilities
├── requirements.txt         # Python dependencies (currently empty)
├── sar_cnn_model.pth       # Trained PyTorch model
├── mock_data.json          # Test/demo data
└── final_ghost_hunter_report_*.json  # Output reports
```

## Data Organization

### Input Data Structure
```
data/
├── raw/
│   ├── satellite/
│   │   └── sat*/
│   │       └── measurement/     # Sentinel-1 SAR TIFF files (*vv*.tiff)
│   └── mpa_boundaries/
│       └── Combined_MPA_Boundaries.geojson
└── processed/
    └── ais_detection_input_*.json
```

### CNN Training Dataset
```
cnn_dataset/
├── images/                    # Training image patches
│   ├── ship_*.png            # Positive samples (vessels)
│   └── sea_*.png             # Negative samples (sea/noise)
├── labels.json               # Training labels and metadata
├── dataset_stats.json        # Dataset statistics
└── sample_patches.png        # Visualization of training data
```

## Hackxois Submodule

The `hackxois/` directory contains the original hackathon codebase that forms the foundation:

```
hackxois/
├── marine_vessel_detection_pipeline.py  # Original pipeline orchestrator
├── requirements.txt                     # Hackxois dependencies
├── data/
│   └── ais_data/                       # Daily AIS CSV files
├── scripts/                            # Individual pipeline components
│   ├── sat_mpa_intersection.py         # SAR-MPA intersection analysis
│   ├── ship_detections.py              # SBCI vessel detection
│   ├── vessel_feature_extraction.py    # Feature extraction & clustering
│   ├── ais_detection.py                # AIS cross-referencing
│   └── cnn_dataset_generator.py        # Training data generation
└── output/                             # Hackxois pipeline outputs
```

## Output Structure

```
output/
├── json/                              # Structured data outputs
│   ├── ship_detection_results*.json   # Raw detection results
│   ├── vessel_features*.json          # Extracted vessel features
│   ├── ship_detection_with_ais_status*.json  # AIS-enhanced results
│   └── multi_satellite_pipeline_summary.json
├── png/                               # Visualizations
│   └── vessel_clusters_overlay*.png   # Cluster visualization overlays
└── chips/                             # Extracted vessel image patches
    └── sat*/                          # Per-satellite chip directories
```

## Architecture Patterns

### Pipeline-Based Processing
- **Sequential Steps**: Each pipeline step depends on outputs from previous steps
- **Multi-Satellite Support**: Process multiple satellite scenes in parallel or sequence
- **Modular Components**: Individual scripts can be run independently for debugging

### Data Flow Pattern
1. **Raw Data Ingestion**: Satellite TIFF files and MPA boundaries
2. **Detection Phase**: SBCI-based vessel detection
3. **Feature Extraction**: Vessel characteristics and clustering
4. **Cross-Reference**: AIS data matching for dark vessel identification
5. **Verification**: CNN-based vessel confirmation
6. **Analysis**: Behavioral pattern analysis and risk assessment
7. **Output**: JSON reports and visualizations

### Configuration Management
- **Environment Variables**: Used for satellite paths and names
- **JSON Configuration**: Dataset statistics and pipeline parameters
- **Hardcoded Constants**: Thresholds and parameters in individual modules

## File Naming Conventions

### Satellite-Specific Files
- Pattern: `*_{satellite_name}.json` (e.g., `ship_detection_results_sat1.json`)
- Enables multi-satellite processing and result aggregation

### Image Files
- **Training Data**: `ship_{id}.png`, `sea_{id}.png`
- **Vessel Chips**: `vessel_{vessel_id}.png`
- **Visualizations**: Descriptive names with satellite identifier

### Model Files
- **PyTorch Models**: `.pth` extension
- **Training Data**: JSON format for labels and metadata

## Development Guidelines

### Module Organization
- **Single Responsibility**: Each script handles one pipeline step
- **Loose Coupling**: Modules communicate via JSON files
- **Error Handling**: Graceful degradation with fallback to mock data

### Path Management
- **Relative Paths**: All paths relative to project root
- **Cross-Platform**: Handle both Windows and Unix path separators
- **Environment Awareness**: Detect and adapt to runtime environment

### Testing & Demo Support
- **Mock Data**: Fallback data for testing without full satellite datasets
- **Incremental Processing**: Support for partial pipeline execution
- **Verbose Logging**: Detailed progress reporting for debugging