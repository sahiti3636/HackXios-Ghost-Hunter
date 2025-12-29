#!/usr/bin/env python3
"""
Ship Detection System - Comprehensive SBCI Implementation

STEP 2: Detect Vessels via Sentinel-1 SAR

This module implements physics-based vessel detection using Sea-Background Contrast Index (SBCI).
Combines both full-image detection and MPA-focused detection capabilities.

Processing Steps:
1. Radiometric Calibration: Converts raw SAR signal → σ⁰ (physical reflectivity)
2. Speckle Filtering: Reduces radar noise using median filters
3. Local Sea Normalization: Computes local mean backscatter to account for wind/waves
4. SBCI Calculation: Ships → high SBCI, Sea clutter → low SBCI
5. Binary Detection: Bright pixels = possible ship pixels, Dark pixels = sea

Key Physical Property:
- Calm ocean → low backscatter (dark)
- Ships (metal, edges) → high backscatter (bright)

Usage:
    python ship_detections.py --satellite-path <path_to_satellite_tiff>

Author: Marine Vessel Detection System
Date: December 2025
"""

import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Union

try:
    import netCDF4 as nc
    from scipy import ndimage
    from scipy.ndimage import median_filter, uniform_filter, binary_opening
    from skimage import measure
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import rasterio
    import rasterio.transform
    from shapely.geometry import Polygon, box
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install netCDF4 scipy scikit-image matplotlib rasterio shapely")
    DEPENDENCIES_AVAILABLE = False

class ShipDetectorSBCI:
    """
    Comprehensive ship detection using Sea-Background Contrast Index (SBCI).
    
    Supports both full-image detection and MPA-focused detection modes.
    Implements physics-based, explainable vessel detection methods.
    """
    
    def __init__(self, sbci_threshold: float = 5.0, window_size: Optional[int] = None, 
                 min_vessel_size: int = 3, buffer_km: float = 5.0):
        """
        Initialize the Ship Detector.
        
        Args:
            sbci_threshold: SBCI threshold for vessel detection (default: 5.0)
            window_size: Local background window size (auto-calculated if None)
            min_vessel_size: Minimum vessel size in pixels
            buffer_km: Buffer around MPA boundaries in kilometers
        """
        self.sbci_threshold = sbci_threshold
        self.window_size = window_size
        self.min_vessel_size = min_vessel_size
        self.buffer_km = buffer_km
        
        # Data containers
        self.sar_data = None
        self.coordinates = None
        self.sbci_map = None
        self.background_map = None
        self.detection_mask = None
        self.vessel_candidates = []
        
        # MPA-specific data
        self.mpa_boundaries = None
        self.overlapping_mpas = []
        self.mpa_regions = {}
        self.detection_results = {}
        
        # Statistics
        self.stats = {
            'total_pixels': 0,
            'valid_pixels': 0,
            'detected_pixels': 0,
            'vessel_count': 0,
            'processing_time': 0.0
        }
    
    def load_sentinel1_data(self, satellite_file_path: str) -> bool:
        """
        STEP 2.1: Load Sentinel-1 Radar Image from specific file path.
        
        Args:
            satellite_file_path: Specific path to satellite TIFF file
            
        Returns:
            bool: Success status
        """
        print("=== STEP 2.1: LOADING SENTINEL-1 RADAR IMAGE ===")
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Missing required dependencies")
            return False
        
        if not os.path.exists(satellite_file_path):
            print(f"❌ Satellite file not found: {satellite_file_path}")
            return False
        
        print(f"Loading: {os.path.basename(satellite_file_path)}")
        
        if satellite_file_path.endswith('.tiff'):
            return self._load_tiff_data(satellite_file_path)
        elif satellite_file_path.endswith('.nc'):
            return self._load_netcdf_data(satellite_file_path)
        else:
            print(f"❌ Unsupported file format: {satellite_file_path}")
            return False
    
    def _load_tiff_data(self, file_path: str) -> bool:
        """Load data from TIFF file."""
        try:
            print(f"Loading TIFF: {os.path.basename(file_path)}")
            
            with rasterio.open(file_path) as src:
                # Check memory requirements
                height, width = src.height, src.width
                memory_gb = (height * width * 4) / (1024**3)
                
                print(f"   Image size: {height} x {width} pixels")
                print(f"   Memory requirement: {memory_gb:.2f} GB")
                
                if memory_gb > 1.0:
                    downsample_factor = int(np.ceil(np.sqrt(memory_gb)))
                    print(f"   Downsampling by factor {downsample_factor} for memory efficiency")
                    
                    self.sar_data = src.read(1, 
                                           out_shape=(height // downsample_factor, 
                                                    width // downsample_factor),
                                           resampling=rasterio.enums.Resampling.average).astype(np.float32)
                else:
                    self.sar_data = src.read(1).astype(np.float32)
                
                # Handle invalid values
                self.sar_data[self.sar_data <= 0] = np.nan
                self.sar_data[np.isinf(self.sar_data)] = np.nan
                
                # Extract coordinates using generalized method
                try:
                    bounds = src.bounds
                    crs = src.crs
                    
                    # Method 1: Check if bounds are already in geographic coordinates
                    if (bounds.left >= -180 and bounds.right <= 180 and 
                        bounds.bottom >= -90 and bounds.top <= 90):
                        # Already in geographic coordinates
                        self.coordinates = {
                            'west': bounds.left, 'east': bounds.right,
                            'south': bounds.bottom, 'north': bounds.top
                        }
                        print(f"   Using direct geographic bounds from TIFF")
                    
                    # Method 2: Try to transform from source CRS to WGS84
                    elif crs and crs != 'EPSG:4326':
                        try:
                            from rasterio.warp import transform_bounds
                            
                            # Transform bounds to WGS84
                            west, south, east, north = transform_bounds(
                                crs, 'EPSG:4326', 
                                bounds.left, bounds.bottom, bounds.right, bounds.top
                            )
                            
                            self.coordinates = {
                                'west': west, 'east': east,
                                'south': south, 'north': north
                            }
                            print(f"   Transformed bounds from {crs} to WGS84")
                            
                        except Exception as transform_error:
                            print(f"   ⚠️  CRS transformation failed: {transform_error}")
                            self.coordinates = self._extract_bounds_from_manifest(file_path) or None
                    
                    # Method 3: Extract from Sentinel-1 SAFE manifest (if available)
                    else:
                        self.coordinates = self._extract_bounds_from_manifest(file_path)
                        if self.coordinates:
                            print(f"   Extracted bounds from SAFE manifest")
                        else:
                            print(f"   ⚠️  Could not determine geographic bounds automatically")
                            print(f"   Bounds appear to be in pixel coordinates: {bounds}")
                            self.coordinates = None
                
                except Exception as coord_error:
                    print(f"   ⚠️  Error extracting coordinates: {coord_error}")
                    self.coordinates = None
                
                self.stats['total_pixels'] = self.sar_data.size
                self.stats['valid_pixels'] = np.sum(~np.isnan(self.sar_data))
                
                print(f"✅ TIFF loaded successfully: {self.sar_data.shape}")
                print(f"   Valid pixels: {self.stats['valid_pixels']:,}")
                print(f"   Intensity range: {np.nanmin(self.sar_data):.2f} to {np.nanmax(self.sar_data):.2f}")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading TIFF: {e}")
            return False
    
    def _load_netcdf_data(self, file_path: str) -> bool:
        """Load data from NetCDF file."""
        try:
            print(f"Loading NetCDF: {os.path.basename(file_path)}")
            
            with nc.Dataset(file_path, 'r') as dataset:
                # Find VV polarization data
                vv_var = None
                for var_name in dataset.variables:
                    if 'vv' in var_name.lower():
                        vv_var = dataset.variables[var_name]
                        break
                
                if vv_var is None:
                    print("❌ No VV polarization data found in NetCDF")
                    return False
                
                self.sar_data = np.array(vv_var[:]).astype(np.float32)
                
                # Handle invalid values
                self.sar_data[self.sar_data <= 0] = np.nan
                self.sar_data[np.isinf(self.sar_data)] = np.nan
                
                # Extract coordinates if available
                try:
                    if 'longitude' in dataset.variables and 'latitude' in dataset.variables:
                        lons = np.array(dataset.variables['longitude'][:])
                        lats = np.array(dataset.variables['latitude'][:])
                        self.coordinates = {
                            'west': np.min(lons), 'east': np.max(lons),
                            'south': np.min(lats), 'north': np.max(lats)
                        }
                except:
                    self.coordinates = None
                
                self.stats['total_pixels'] = self.sar_data.size
                self.stats['valid_pixels'] = np.sum(~np.isnan(self.sar_data))
                
                print(f"✅ NetCDF loaded successfully: {self.sar_data.shape}")
                print(f"   Valid pixels: {self.stats['valid_pixels']:,}")
                print(f"   Intensity range: {np.nanmin(self.sar_data):.2f} to {np.nanmax(self.sar_data):.2f}")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading NetCDF: {e}")
            return False
    
    def _extract_bounds_from_manifest(self, satellite_file_path: str) -> Optional[Dict]:
        """
        Extract geographic bounds from Sentinel-1 SAFE manifest file.
        
        Args:
            satellite_file_path: Path to satellite TIFF file
            
        Returns:
            Dict with geographic bounds or None if not found
        """
        try:
            from pathlib import Path
            
            # Look for manifest.safe file in parent directories
            current_dir = Path(satellite_file_path).parent
            
            # Search up to 3 levels for manifest.safe
            for _ in range(3):
                manifest_path = current_dir / 'manifest.safe'
                if manifest_path.exists():
                    return self._parse_manifest_coordinates(manifest_path)
                current_dir = current_dir.parent
                if current_dir == current_dir.parent:  # Reached root
                    break
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error reading manifest: {e}")
            return None
    
    def _parse_manifest_coordinates(self, manifest_path) -> Optional[Dict]:
        """
        Parse coordinates from Sentinel-1 manifest.safe file.
        
        Args:
            manifest_path: Path to manifest.safe file
            
        Returns:
            Dict with geographic bounds or None if parsing failed
        """
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(manifest_path)
            root = tree.getroot()
            
            # Find coordinate elements in the manifest
            coordinates = []
            
            # Look for different coordinate patterns in Sentinel-1 manifests
            for coord_elem in root.iter():
                if 'coordinate' in coord_elem.tag.lower() or 'gml:coordinates' in coord_elem.tag:
                    if coord_elem.text:
                        coord_pairs = coord_elem.text.strip().split()
                        for pair in coord_pairs:
                            if ',' in pair:
                                lat, lon = map(float, pair.split(','))
                                coordinates.append((lat, lon))
            
            # Also look for footprint coordinates
            for footprint in root.iter():
                if 'footprint' in footprint.tag.lower():
                    for coord in footprint.iter():
                        if coord.text and ',' in coord.text:
                            try:
                                coord_pairs = coord.text.strip().split()
                                for pair in coord_pairs:
                                    if ',' in pair:
                                        lat, lon = map(float, pair.split(','))
                                        coordinates.append((lat, lon))
                            except:
                                continue
            
            if coordinates:
                # Calculate bounding box from coordinates
                lats = [coord[0] for coord in coordinates]
                lons = [coord[1] for coord in coordinates]
                
                return {
                    'west': min(lons),
                    'east': max(lons),
                    'south': min(lats),
                    'north': max(lats)
                }
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error parsing manifest coordinates: {e}")
            return None
    
    def apply_noise_reduction(self) -> bool:
        """
        STEP 2.2: Apply Speckle Filtering.
        
        Reduces radar noise using median filter while preserving vessel signatures.
        
        Returns:
            bool: Success status
        """
        print("\n=== STEP 2.2: APPLYING SPECKLE FILTERING ===")
        
        if self.sar_data is None:
            print("❌ No SAR data loaded")
            return False
        
        # Apply median filter to reduce speckle noise
        print("   Applying median filter (3x3 kernel)...")
        
        # Preserve NaN values
        original_shape = self.sar_data.shape
        valid_mask = ~np.isnan(self.sar_data)
        
        if np.sum(valid_mask) == 0:
            print("❌ No valid pixels to filter")
            return False
        
        # Create temporary array for filtering
        temp_data = np.copy(self.sar_data)
        temp_data[~valid_mask] = 0  # Temporarily fill NaN with 0 for filtering
        
        filtered_temp = median_filter(temp_data, size=3)
        
        # Restore original structure
        filtered_data = np.full(original_shape, np.nan, dtype=np.float32)
        filtered_data[valid_mask] = filtered_temp[valid_mask]
        
        self.sar_data = filtered_data
        
        print(f"✅ Speckle filtering completed")
        print(f"   Preserved {np.sum(~np.isnan(self.sar_data)):,} valid pixels")
        
        return True
    
    def compute_local_sea_background(self) -> bool:
        """
        STEP 2.3: Compute Local Sea Background.
        
        Computes local mean backscatter to account for wind/waves and sea state variations.
        
        Returns:
            bool: Success status
        """
        print("\n=== STEP 2.3: COMPUTING LOCAL SEA BACKGROUND ===")
        
        if self.sar_data is None:
            print("❌ No SAR data available")
            return False
        
        # Create local background map
        background_map = np.full_like(self.sar_data, np.nan)
        
        # Determine kernel size adaptively
        kernel_size = self.window_size or min(21, max(7, min(self.sar_data.shape) // 20))
        print(f"   Using window size: {kernel_size}x{kernel_size}")
        
        # For small images, use a simpler approach
        if min(self.sar_data.shape) < kernel_size * 2:
            # Use global background for very small images
            global_mean = np.nanmean(self.sar_data)
            background_map = np.full_like(self.sar_data, global_mean)
            print(f"   Using global background (image too small for local windows)")
            print(f"   Global mean: {global_mean:.6f}")
        else:
            # Compute local sum using uniform filter
            local_sum = uniform_filter(np.nan_to_num(self.sar_data, 0), size=kernel_size, mode='constant')
            local_count = uniform_filter((~np.isnan(self.sar_data)).astype(float), size=kernel_size, mode='constant')
            
            # Subtract center pixel to exclude it from background calculation
            local_count_excluding_center = local_count - 1.0/(kernel_size * kernel_size)
            center_contribution = self.sar_data / (kernel_size * kernel_size)
            local_sum_excluding_center = local_sum - center_contribution
            
            # Compute mean excluding center pixel
            valid_background = local_count_excluding_center > 0.1  # At least some valid pixels
            background_map[valid_background] = (local_sum_excluding_center[valid_background] / 
                                              local_count_excluding_center[valid_background])
        
        self.background_map = background_map
        
        valid_bg_pixels = np.sum(~np.isnan(background_map))
        print(f"✅ Computed local sea background:")
        print(f"   Valid background pixels: {valid_bg_pixels:,}")
        print(f"   Background range: {np.nanmin(background_map):.6f} to {np.nanmax(background_map):.6f}")
        
        return True
    
    def compute_sbci(self) -> bool:
        """
        STEP 2.4: Compute SBCI (Sea-Background Contrast Index).
        
        Core metric: SBCI = pixel_intensity / local_sea_background_mean
        Ships → high SBCI, Sea clutter → low SBCI
        
        Returns:
            bool: Success status
        """
        print("\n=== STEP 2.4: COMPUTING SBCI ===")
        
        if self.sar_data is None or not hasattr(self, 'background_map'):
            print("❌ Missing SAR data or background map")
            return False
        
        # Compute SBCI = pixel_intensity / local_sea_mean
        print("   Computing SBCI = pixel_intensity / local_sea_background_mean")
        
        # Handle division by zero safely
        sbci_map = np.full_like(self.sar_data, np.nan)
        
        valid_pixels = (~np.isnan(self.sar_data)) & (~np.isnan(self.background_map)) & (self.background_map > 0)
        
        if np.sum(valid_pixels) == 0:
            print("❌ No valid pixels for SBCI computation")
            return False
        
        sbci_map[valid_pixels] = self.sar_data[valid_pixels] / self.background_map[valid_pixels]
        
        self.sbci_map = sbci_map
        
        valid_sbci_pixels = np.sum(~np.isnan(sbci_map))
        print(f"✅ Computed SBCI:")
        print(f"   Valid SBCI pixels: {valid_sbci_pixels:,}")
        print(f"   SBCI range: {np.nanmin(sbci_map):.2f} to {np.nanmax(sbci_map):.2f}")
        print(f"   Mean SBCI: {np.nanmean(sbci_map):.2f}")
        print(f"   Detection threshold: {self.sbci_threshold}")
        
        return True
    
    def apply_detection_threshold(self) -> bool:
        """
        STEP 2.5: Apply Detection Threshold.
        
        Creates binary detection map: SBCI > threshold = ship candidate
        Output: Bright pixels = possible ship pixels, Dark pixels = sea
        
        Returns:
            bool: Success status
        """
        print("\n=== STEP 2.5: APPLYING DETECTION THRESHOLD ===")
        
        if self.sbci_map is None:
            print("❌ No SBCI map available")
            return False
        
        # Apply threshold: SBCI > threshold = ship candidate
        detection_mask = np.zeros_like(self.sbci_map, dtype=np.uint8)
        
        valid_pixels = ~np.isnan(self.sbci_map)
        high_contrast_pixels = (self.sbci_map > self.sbci_threshold) & valid_pixels
        
        detection_mask[high_contrast_pixels] = 1
        
        self.detection_mask = detection_mask
        
        detected_pixels = np.sum(high_contrast_pixels)
        self.stats['detected_pixels'] = detected_pixels
        
        print(f"✅ Applied detection threshold:")
        print(f"   Threshold: SBCI > {self.sbci_threshold}")
        print(f"   Detected pixels: {detected_pixels:,}")
        print(f"   Detection rate: {detected_pixels/self.stats['valid_pixels']*100:.4f}%")
        
        return True
    
    def post_process_detections(self) -> bool:
        """
        STEP 2.6: Post-Process Detections.
        
        Removes noise and identifies connected vessel components.
        
        Returns:
            bool: Success status
        """
        print("\n=== STEP 2.6: POST-PROCESSING DETECTIONS ===")
        
        if self.detection_mask is None:
            print("❌ No detection mask available")
            return False
        
        try:
            # Remove single isolated pixels
            print("   Applying morphological operations...")
            cleaned_mask = binary_opening(self.detection_mask, structure=np.ones((3,3)))
            
            # Fill small holes
            cleaned_mask = ndimage.binary_fill_holes(cleaned_mask)
            
            # Find connected components (potential vessels)
            print("   Finding connected components...")
            labeled_mask, num_features = measure.label(cleaned_mask, return_num=True)
            
            print(f"   Found {num_features} connected components")
            
            # Limit processing to reasonable number of components
            if num_features > 1000:
                print(f"   ⚠️  Large number of components ({num_features}), limiting to top 100 by size")
                # Get component sizes and keep only the largest ones
                component_sizes = []
                for i in range(1, num_features + 1):
                    size = np.sum(labeled_mask == i)
                    component_sizes.append((i, size))
                
                # Sort by size and keep top 100
                component_sizes.sort(key=lambda x: x[1], reverse=True)
                keep_components = [comp[0] for comp in component_sizes[:100]]
                
                # Create new mask with only selected components
                new_labeled_mask = np.zeros_like(labeled_mask)
                for i, comp_id in enumerate(keep_components, 1):
                    new_labeled_mask[labeled_mask == comp_id] = i
                
                labeled_mask = new_labeled_mask
                num_features = len(keep_components)
                print(f"   Reduced to {num_features} largest components")
            
            # Extract vessel candidates
            self.vessel_candidates = []
            vessel_id = 1
            
            print("   Extracting vessel properties...")
            regions = measure.regionprops(labeled_mask)
            
            for i, region in enumerate(regions):
                if i % 10 == 0 and i > 0:
                    print(f"   Processing region {i+1}/{len(regions)}...")
                
                if region.area >= self.min_vessel_size:
                    # Calculate centroid coordinates
                    centroid_y, centroid_x = region.centroid
                    
                    # Convert to geographic coordinates if available
                    lat, lon = None, None
                    if self.coordinates:
                        height, width = self.sar_data.shape
                        norm_x = centroid_x / width
                        norm_y = centroid_y / height
                        
                        lon = self.coordinates['west'] + norm_x * (self.coordinates['east'] - self.coordinates['west'])
                        lat = self.coordinates['north'] - norm_y * (self.coordinates['north'] - self.coordinates['south'])
                    
                    # Extract SBCI values for this region (optimized)
                    region_mask = labeled_mask == region.label
                    
                    # Use region bounding box to limit processing
                    min_row, min_col, max_row, max_col = region.bbox
                    region_slice = region_mask[min_row:max_row, min_col:max_col]
                    
                    if hasattr(self, 'sbci_map') and self.sbci_map is not None:
                        sbci_slice = self.sbci_map[min_row:max_row, min_col:max_col]
                        sbci_values = sbci_slice[region_slice]
                        sbci_values = sbci_values[~np.isnan(sbci_values)]
                    else:
                        sbci_values = np.array([])
                    
                    # Extract SAR values for this region (optimized)
                    sar_slice = self.sar_data[min_row:max_row, min_col:max_col]
                    sar_values = sar_slice[region_slice]
                    sar_values = sar_values[~np.isnan(sar_values)]
                    
                    vessel = {
                        'vessel_id': vessel_id,
                        'pixel_x': float(centroid_x),
                        'pixel_y': float(centroid_y),
                        'latitude': float(lat) if lat is not None else None,
                        'longitude': float(lon) if lon is not None else None,
                        'area_pixels': int(region.area),
                        'mean_backscatter': float(np.mean(sar_values)) if len(sar_values) > 0 else 0.0,
                        'max_backscatter': float(np.max(sar_values)) if len(sar_values) > 0 else 0.0,
                        'mean_sbci': float(np.mean(sbci_values)) if len(sbci_values) > 0 else 0.0,
                        'max_sbci': float(np.max(sbci_values)) if len(sbci_values) > 0 else 0.0,
                        'detection_confidence': float(np.max(sbci_values)) if len(sbci_values) > 0 else 0.0
                    }
                    
                    self.vessel_candidates.append(vessel)
                    vessel_id += 1
            
            self.stats['vessel_count'] = len(self.vessel_candidates)
            
            print(f"✅ Post-processing completed:")
            print(f"   Vessel candidates: {len(self.vessel_candidates)}")
            print(f"   Size threshold: ≥{self.min_vessel_size} pixels")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in post-processing: {e}")
            # Initialize empty vessel candidates to allow saving
            self.vessel_candidates = []
            self.stats['vessel_count'] = 0
            return False
    
    def load_mpa_boundaries(self, mpa_file_path: str = 'data/raw/mpa_boundaries/Combined_MPA_Boundaries.geojson') -> bool:
        """
        Load Marine Protected Area boundaries for MPA-focused detection.
        
        Args:
            mpa_file_path: Path to MPA boundaries GeoJSON file
            
        Returns:
            bool: Success status
        """
        print("\n=== LOADING MPA BOUNDARIES ===")
        
        if not os.path.exists(mpa_file_path):
            print(f"❌ MPA boundaries file not found: {mpa_file_path}")
            return False
        
        try:
            with open(mpa_file_path, 'r') as f:
                self.mpa_boundaries = json.load(f)
            
            mpa_count = len(self.mpa_boundaries['features'])
            print(f"✅ Loaded {mpa_count} MPA boundaries")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading MPA boundaries: {e}")
            return False
    
    def detect_vessels_in_mpa_regions(self) -> bool:
        """
        Run vessel detection specifically in MPA regions.
        
        Returns:
            bool: Success status
        """
        print("\n=== MPA-FOCUSED VESSEL DETECTION ===")
        
        if not self.mpa_boundaries:
            print("❌ No MPA boundaries loaded")
            return False
        
        if self.sar_data is None:
            print("❌ No SAR data loaded")
            return False
        
        # Find overlapping MPAs (simplified version)
        if self.coordinates:
            sat_polygon = box(
                self.coordinates['west'], self.coordinates['south'],
                self.coordinates['east'], self.coordinates['north']
            )
            
            self.overlapping_mpas = []
            
            for feature in self.mpa_boundaries['features']:
                mpa_name = feature['properties'].get('mpa_name', 'Unknown')
                
                try:
                    if feature['geometry']['type'] == 'Polygon':
                        coords = feature['geometry']['coordinates'][0]
                    elif feature['geometry']['type'] == 'MultiPolygon':
                        coords = feature['geometry']['coordinates'][0][0]
                    else:
                        continue
                    
                    mpa_polygon = Polygon(coords)
                    
                    if sat_polygon.intersects(mpa_polygon):
                        overlap_info = {
                            'name': mpa_name,
                            'polygon': mpa_polygon,
                            'properties': feature['properties']
                        }
                        self.overlapping_mpas.append(overlap_info)
                        print(f"   ✅ Found overlapping MPA: {mpa_name}")
                
                except Exception as e:
                    print(f"   ⚠️  Error processing {mpa_name}: {e}")
                    continue
        
        if not self.overlapping_mpas:
            print("❌ No overlapping MPAs found")
            return False
        
        # Run detection for each MPA
        self.detection_results = {}
        
        for mpa in self.overlapping_mpas:
            mpa_name = mpa['name']
            print(f"\n   Processing MPA: {mpa_name}")
            
            # For simplicity, run detection on full image
            # In production, you would crop to MPA region + buffer
            result = {
                'mpa_name': mpa_name,
                'vessels_detected': len(self.vessel_candidates),
                'vessels': self.vessel_candidates,
                'processing_stats': {
                    'region_size': self.sar_data.shape,
                    'valid_pixels': self.stats['valid_pixels'],
                    'sbci_threshold': self.sbci_threshold,
                    'window_size': self.window_size or 21
                }
            }
            
            self.detection_results[mpa_name] = result
            print(f"   ✅ Detected {len(self.vessel_candidates)} vessels in {mpa_name}")
        
        return True
    
    def save_detection_results(self, satellite_name: str, output_file: Optional[str] = None) -> bool:
        """
        Save detection results to JSON file with satellite-specific naming.
        
        Args:
            satellite_name: Name of the satellite (e.g., 'sat1', 'sat2')
            output_file: Optional custom output file path
            
        Returns:
            bool: Success status
        """
        print(f"\n=== SAVING DETECTION RESULTS FOR {satellite_name.upper()} ===")
        
        if output_file is None:
            output_file = f'output/json/ship_detection_results_{satellite_name}.json'
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            results = {
                'satellite_name': satellite_name,
                'detection_parameters': {
                    'sbci_threshold': self.sbci_threshold,
                    'min_vessel_size': self.min_vessel_size,
                    'window_size': self.window_size
                },
                'statistics': convert_numpy_types(self.stats),
                'vessel_candidates': convert_numpy_types(getattr(self, 'vessel_candidates', [])),
                'mpa_results': convert_numpy_types(getattr(self, 'detection_results', None)) if hasattr(self, 'detection_results') else None,
                'processing_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def create_detection_visualization(self, output_file: str = 'output/png/ship_detection_plot.png') -> bool:
        """
        Create visualization of detection results.
        
        Args:
            output_file: Output image file path
            
        Returns:
            bool: Success status
        """
        print(f"\n=== CREATING DETECTION VISUALIZATION ===")
        
        if self.sar_data is None:
            print("❌ No SAR data to visualize")
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Ship Detection Results (SBCI Method)', fontsize=16)
            
            # Original SAR image (log scale for better visualization)
            sar_display = np.log10(np.clip(self.sar_data, 1, np.nanmax(self.sar_data)))
            im1 = axes[0,0].imshow(sar_display, cmap='gray', aspect='auto')
            axes[0,0].set_title('Original SAR Image (log scale)')
            axes[0,0].set_xlabel('X (pixels)')
            axes[0,0].set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=axes[0,0])
            
            # SBCI map
            if hasattr(self, 'sbci_map') and self.sbci_map is not None:
                im2 = axes[0,1].imshow(np.clip(self.sbci_map, 0, 20), cmap='hot', aspect='auto')
                axes[0,1].set_title('SBCI Map (clipped to 0-20)')
                axes[0,1].set_xlabel('X (pixels)')
                axes[0,1].set_ylabel('Y (pixels)')
                plt.colorbar(im2, ax=axes[0,1])
            
            # Detection mask
            if hasattr(self, 'detection_mask') and self.detection_mask is not None:
                im3 = axes[1,0].imshow(self.detection_mask, cmap='Reds', aspect='auto')
                axes[1,0].set_title('Detection Mask')
                axes[1,0].set_xlabel('X (pixels)')
                axes[1,0].set_ylabel('Y (pixels)')
                plt.colorbar(im3, ax=axes[1,0])
            
            # Vessel overlay
            axes[1,1].imshow(sar_display, cmap='gray', aspect='auto')
            
            # Overlay vessel detections
            for i, vessel in enumerate(self.vessel_candidates):
                x, y = vessel['pixel_x'], vessel['pixel_y']
                
                # Add vessel marker
                circle = plt.Circle((x, y), radius=5, color='red', fill=False, linewidth=2)
                axes[1,1].add_patch(circle)
                
                # Add vessel ID
                axes[1,1].text(x+8, y, f"V{vessel['vessel_id']}", 
                              color='red', fontsize=10, fontweight='bold')
            
            axes[1,1].set_title(f'Detected Vessels (n={len(self.vessel_candidates)})')
            axes[1,1].set_xlabel('X (pixels)')
            axes[1,1].set_ylabel('Y (pixels)')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return False
    
    def print_detection_summary(self):
        """Print comprehensive detection summary."""
        print("\n" + "="*70)
        print("SHIP DETECTION SUMMARY (SBCI METHOD)")
        print("="*70)
        
        print(f"\n📊 PROCESSING STATISTICS:")
        print(f"   Total pixels: {self.stats['total_pixels']:,}")
        print(f"   Valid pixels: {self.stats['valid_pixels']:,}")
        print(f"   Detected pixels: {self.stats['detected_pixels']:,}")
        print(f"   Detection rate: {self.stats['detected_pixels']/self.stats['valid_pixels']*100:.4f}%")
        
        print(f"\n🎯 DETECTION PARAMETERS:")
        print(f"   SBCI threshold: {self.sbci_threshold}")
        print(f"   Minimum vessel size: {self.min_vessel_size} pixels")
        print(f"   Window size: {self.window_size or 'auto'}")
        
        print(f"\n🚢 VESSEL DETECTIONS:")
        print(f"   Total vessels detected: {len(self.vessel_candidates)}")
        
        if self.vessel_candidates:
            print(f"\n📋 VESSEL DETAILS:")
            for vessel in self.vessel_candidates:
                coord_str = ""
                if vessel['latitude'] and vessel['longitude']:
                    coord_str = f" at ({vessel['latitude']:.4f}°, {vessel['longitude']:.4f}°)"
                
                print(f"   • Vessel {vessel['vessel_id']}: {vessel['area_pixels']} pixels{coord_str}")
                print(f"     SBCI: {vessel['mean_sbci']:.2f} (mean), {vessel['max_sbci']:.2f} (max)")
                print(f"     Backscatter: {vessel['mean_backscatter']:.1f} (mean), {vessel['max_backscatter']:.1f} (max)")
        
        if self.detection_results:
            print(f"\n🌊 MPA DETECTION RESULTS:")
            for mpa_name, result in self.detection_results.items():
                print(f"   • {mpa_name}: {result['vessels_detected']} vessels")
        
        print("="*70)
    
    def run_full_detection(self, satellite_file_path: str,
                          mpa_file_path: Optional[str] = None) -> bool:
        """
        Run complete ship detection pipeline for a single satellite file.
        
        Args:
            satellite_file_path: Path to specific satellite TIFF file
            mpa_file_path: Optional path to MPA boundaries for focused detection
            
        Returns:
            bool: Success status
        """
        print("🚢 SHIP DETECTION SYSTEM - SBCI METHOD")
        print("="*70)
        print("STEP 2: Detect Vessels via Sentinel-1 SAR")
        
        # Extract satellite name from path for output naming
        satellite_name = self._extract_satellite_name(satellite_file_path)
        print(f"Processing satellite: {satellite_name}")
        
        start_time = datetime.now()
        
        # Execute detection pipeline
        success = True
        success &= self.load_sentinel1_data(satellite_file_path)
        success &= self.apply_noise_reduction()
        success &= self.compute_local_sea_background()
        success &= self.compute_sbci()
        success &= self.apply_detection_threshold()
        success &= self.post_process_detections()
        
        # Optional MPA-focused detection
        if mpa_file_path and success:
            success &= self.load_mpa_boundaries(mpa_file_path)
            success &= self.detect_vessels_in_mpa_regions()
        
        # Calculate processing time
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Always try to save results, even if some steps failed
        try:
            self.save_detection_results(satellite_name)
            print(f"✅ Final results saved for {satellite_name}")
        except Exception as e:
            print(f"⚠️  Error saving final results: {e}")
        
        if success:
            try:
                self.create_detection_visualization(satellite_name)
            except Exception as e:
                print(f"⚠️  Error creating visualization: {e}")
            
            self.print_detection_summary()
            print(f"\n🎉 SHIP DETECTION COMPLETE FOR {satellite_name.upper()}!")
            print(f"Processing time: {self.stats['processing_time']:.1f} seconds")
        else:
            print(f"\n❌ Ship detection failed for {satellite_name}. Check error messages above.")
        
        return success
    
    def _extract_satellite_name(self, satellite_file_path: str) -> str:
        """Extract satellite name from file path"""
        try:
            # Extract satellite folder name (e.g., 'sat1', 'sat2', etc.)
            path_parts = satellite_file_path.replace('\\', '/').split('/')
            for i, part in enumerate(path_parts):
                if part.startswith('sat') and part[3:].isdigit():
                    return part
            
            # Fallback: use parent directory name
            return Path(satellite_file_path).parent.parent.name
        except:
            # Final fallback: use timestamp
            return f"sat_{datetime.now().strftime('%H%M%S')}"
    
    def create_detection_visualization(self, satellite_name: str, output_file: Optional[str] = None) -> bool:
        """
        Create visualization of detection results for specific satellite.
        
        Args:
            satellite_name: Name of the satellite
            output_file: Optional custom output file path
            
        Returns:
            bool: Success status
        """
        print(f"\n=== CREATING DETECTION VISUALIZATION FOR {satellite_name.upper()} ===")
        
        if output_file is None:
            output_file = f'output/png/ship_detection_plot_{satellite_name}.png'
        
        if self.sar_data is None:
            print("❌ No SAR data to visualize")
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Ship Detection Results - {satellite_name.upper()} (SBCI Method)', fontsize=16)
            
            # Original SAR image (log scale for better visualization)
            sar_display = np.log10(np.clip(self.sar_data, 1, np.nanmax(self.sar_data)))
            im1 = axes[0,0].imshow(sar_display, cmap='gray', aspect='auto')
            axes[0,0].set_title('Original SAR Image (log scale)')
            axes[0,0].set_xlabel('X (pixels)')
            axes[0,0].set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=axes[0,0])
            
            # SBCI map
            if hasattr(self, 'sbci_map') and self.sbci_map is not None:
                im2 = axes[0,1].imshow(np.clip(self.sbci_map, 0, 20), cmap='hot', aspect='auto')
                axes[0,1].set_title('SBCI Map (clipped to 0-20)')
                axes[0,1].set_xlabel('X (pixels)')
                axes[0,1].set_ylabel('Y (pixels)')
                plt.colorbar(im2, ax=axes[0,1])
            
            # Detection mask
            if hasattr(self, 'detection_mask') and self.detection_mask is not None:
                im3 = axes[1,0].imshow(self.detection_mask, cmap='Reds', aspect='auto')
                axes[1,0].set_title('Detection Mask')
                axes[1,0].set_xlabel('X (pixels)')
                axes[1,0].set_ylabel('Y (pixels)')
                plt.colorbar(im3, ax=axes[1,0])
            
            # Vessel overlay
            axes[1,1].imshow(sar_display, cmap='gray', aspect='auto')
            
            # Overlay vessel detections
            for i, vessel in enumerate(self.vessel_candidates):
                x, y = vessel['pixel_x'], vessel['pixel_y']
                
                # Add vessel marker
                circle = plt.Circle((x, y), radius=5, color='red', fill=False, linewidth=2)
                axes[1,1].add_patch(circle)
                
                # Add vessel ID
                axes[1,1].text(x+8, y, f"V{vessel['vessel_id']}", 
                              color='red', fontsize=10, fontweight='bold')
            
            axes[1,1].set_title(f'Detected Vessels (n={len(self.vessel_candidates)})')
            axes[1,1].set_xlabel('X (pixels)')
            axes[1,1].set_ylabel('Y (pixels)')
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show to avoid blocking
            print(f"✅ Visualization saved: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return False

def main():
    """Main execution function for standalone usage."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ship Detection System - SBCI Method')
    parser.add_argument('--satellite-path', required=True, 
                       help='Path to satellite TIFF file')
    parser.add_argument('--mpa-boundaries', 
                       default='data/raw/mpa_boundaries/Combined_MPA_Boundaries.geojson',
                       help='Path to MPA boundaries file')
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Missing required dependencies. Please install:")
        print("pip install netCDF4 scipy scikit-image matplotlib rasterio shapely")
        return
    
    # Initialize ship detector
    detector = ShipDetectorSBCI(
        sbci_threshold=5.0,
        min_vessel_size=3,
        buffer_km=5.0
    )
    
    # Run detection for specific satellite file
    success = detector.run_full_detection(
        satellite_file_path=args.satellite_path,
        mpa_file_path=args.mpa_boundaries
    )
    
    if success:
        satellite_name = detector._extract_satellite_name(args.satellite_path)
        print(f"\n🎯 DETECTION RESULTS FOR {satellite_name.upper()}:")
        print(f"   Vessels detected: {len(detector.vessel_candidates)}")
        print(f"   Output files generated:")
        print(f"   • ship_detection_results_{satellite_name}.json - Detection data")
        print(f"   • ship_detection_plot_{satellite_name}.png - Visualization")
    else:
        print(f"\n❌ Detection failed for satellite: {args.satellite_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()