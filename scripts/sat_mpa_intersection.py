#!/usr/bin/env python3
"""
Satellite-MPA Intersection Module

Implements the mathematical condition: Process tile ⟺ Area(Tile∩MPA) > 0

This module handles:
1. Loading satellite tile boundaries from Sentinel-1 data
2. Loading MPA boundaries from GeoJSON files
3. Computing polygon-tile intersections
4. Determining which tiles overlap MPAs for further processing
5. Extracting MPA-specific regions from satellite data

Mathematical Foundation:
- Tile = Satellite coverage polygon
- MPA = Marine Protected Area polygon  
- ∩ = Geometric intersection
- Area(Tile∩MPA) > 0 = Non-zero overlap area

Usage:
    python sat_mpa_intersection.py --satellite-path <path_to_satellite_tiff>

Author: Marine Vessel Detection System
Date: December 2025
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

try:
    import rasterio
    from rasterio.transform import from_bounds
    from shapely.geometry import Polygon, box
    from shapely.ops import transform
    import geopandas as gpd
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install rasterio shapely geopandas")
    DEPENDENCIES_AVAILABLE = False

class SatelliteMPAIntersector:
    """
    Handles intersection analysis between satellite tiles and Marine Protected Areas.
    
    Implements the core mathematical condition:
    Process tile ⟺ Area(Tile∩MPA) > 0
    """
    
    def __init__(self, buffer_km: float = 10.0, min_overlap_percent: float = 0.1):
        """
        Initialize the Satellite-MPA Intersector.
        
        Args:
            buffer_km: Buffer distance around MPAs in kilometers
            min_overlap_percent: Minimum overlap percentage to consider processing
        """
        self.buffer_km = buffer_km
        self.min_overlap_percent = min_overlap_percent
        
        # Data containers
        self.satellite_tiles = []
        self.mpa_boundaries = None
        self.intersecting_pairs = []
        self.processing_regions = {}
        
        # Statistics
        self.stats = {
            'total_tiles': 0,
            'total_mpas': 0,
            'intersecting_tiles': 0,
            'processing_regions': 0,
            'total_overlap_area': 0.0
        }
    
    def load_mpa_boundaries(self, mpa_file_path: str = 'data/raw/mpa_boundaries/Combined_MPA_Boundaries.geojson') -> bool:
        """
        Load Marine Protected Area boundaries from GeoJSON file.
        
        Args:
            mpa_file_path: Path to MPA boundaries GeoJSON file
            
        Returns:
            bool: Success status
        """
        print("=== LOADING MPA BOUNDARIES ===")
        
        if not os.path.exists(mpa_file_path):
            print(f"❌ MPA boundaries file not found: {mpa_file_path}")
            return False
        
        try:
            with open(mpa_file_path, 'r') as f:
                self.mpa_boundaries = json.load(f)
            
            mpa_count = len(self.mpa_boundaries['features'])
            self.stats['total_mpas'] = mpa_count
            
            print(f"✅ Loaded {mpa_count} MPA boundaries")
            
            # Display MPA summary
            for feature in self.mpa_boundaries['features']:
                mpa_name = feature['properties'].get('mpa_name', 'Unknown')
                ocean_region = feature['properties'].get('ocean_region', 'Unknown')
                print(f"   • {mpa_name} ({ocean_region})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading MPA boundaries: {e}")
            return False
    
    def extract_satellite_bounds(self, satellite_file_path: str) -> Optional[Dict]:
        """
        Extract spatial bounds from satellite data file.
        
        Handles both geographic and pixel coordinate systems by attempting
        multiple methods to extract proper geographic bounds.
        
        Args:
            satellite_file_path: Path to satellite TIFF file
            
        Returns:
            Dict with spatial bounds or None if failed
        """
        try:
            with rasterio.open(satellite_file_path) as src:
                bounds = src.bounds
                crs = src.crs
                
                # Method 1: Check if bounds are already in geographic coordinates
                if (bounds.left >= -180 and bounds.right <= 180 and 
                    bounds.bottom >= -90 and bounds.top <= 90):
                    # Already in geographic coordinates
                    satellite_bounds = {
                        'west': bounds.left,
                        'east': bounds.right,
                        'south': bounds.bottom,
                        'north': bounds.top,
                        'file_path': satellite_file_path,
                        'crs': str(crs)
                    }
                    print(f"   Using direct geographic bounds from file")
                    return satellite_bounds
                
                # Method 2: Try to transform from source CRS to WGS84
                if crs and crs != 'EPSG:4326':
                    try:
                        from rasterio.warp import transform_bounds
                        
                        # Transform bounds to WGS84
                        west, south, east, north = transform_bounds(
                            crs, 'EPSG:4326', 
                            bounds.left, bounds.bottom, bounds.right, bounds.top
                        )
                        
                        satellite_bounds = {
                            'west': west,
                            'east': east,
                            'south': south,
                            'north': north,
                            'file_path': satellite_file_path,
                            'crs': 'EPSG:4326'
                        }
                        print(f"   Transformed bounds from {crs} to WGS84")
                        return satellite_bounds
                        
                    except Exception as transform_error:
                        print(f"   ⚠️  CRS transformation failed: {transform_error}")
                
                # Method 3: Extract from Sentinel-1 SAFE manifest (if available)
                bounds_from_manifest = self._extract_bounds_from_manifest(satellite_file_path)
                if bounds_from_manifest:
                    print(f"   Extracted bounds from SAFE manifest")
                    return bounds_from_manifest
                
                # Method 4: Extract from TIFF geotags
                bounds_from_geotags = self._extract_bounds_from_geotags(src)
                if bounds_from_geotags:
                    print(f"   Extracted bounds from TIFF geotags")
                    return bounds_from_geotags
                
                # Method 5: Fallback - assume pixel coordinates and warn user
                print(f"   ⚠️  Could not determine geographic bounds automatically")
                print(f"   Bounds appear to be in pixel coordinates: {bounds}")
                print(f"   Please provide geographic bounds manually or check file metadata")
                
                return None
                
        except Exception as e:
            print(f"❌ Error extracting bounds from {satellite_file_path}: {e}")
            return None
    
    def _extract_bounds_from_manifest(self, satellite_file_path: str) -> Optional[Dict]:
        """
        Extract geographic bounds from Sentinel-1 SAFE manifest file.
        
        Args:
            satellite_file_path: Path to satellite TIFF file
            
        Returns:
            Dict with geographic bounds or None if not found
        """
        try:
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
    
    def _parse_manifest_coordinates(self, manifest_path: Path) -> Optional[Dict]:
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
                                coordinates.append((lon, lat))  # Convert to (lon, lat)
            
            if len(coordinates) >= 4:
                # Calculate bounding box from coordinates
                lons = [coord[0] for coord in coordinates]
                lats = [coord[1] for coord in coordinates]
                
                satellite_bounds = {
                    'west': min(lons),
                    'east': max(lons),
                    'south': min(lats),
                    'north': max(lats),
                    'file_path': str(manifest_path.parent),
                    'crs': 'EPSG:4326'
                }
                
                return satellite_bounds
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error parsing manifest coordinates: {e}")
            return None
    
    def _extract_bounds_from_geotags(self, src) -> Optional[Dict]:
        """
        Extract geographic bounds from TIFF geotags.
        
        Args:
            src: Opened rasterio dataset
            
        Returns:
            Dict with geographic bounds or None if not found
        """
        try:
            # Check for geographic metadata in tags
            tags = src.tags()
            
            # Look for common geographic metadata tags
            geo_keys = ['AREA_OR_POINT', 'GeoTIFF_Information', 'Coordinate_System']
            
            for key, value in tags.items():
                if any(geo_key in key for geo_key in geo_keys):
                    print(f"   Found geo tag: {key} = {value}")
            
            # Try to get corner coordinates if available
            if hasattr(src, 'gcps') and src.gcps[0]:
                gcps = src.gcps[0]
                if len(gcps) >= 4:
                    lons = [gcp.x for gcp in gcps]
                    lats = [gcp.y for gcp in gcps]
                    
                    satellite_bounds = {
                        'west': min(lons),
                        'east': max(lons),
                        'south': min(lats),
                        'north': max(lats),
                        'file_path': src.name,
                        'crs': 'EPSG:4326'
                    }
                    
                    return satellite_bounds
            
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error extracting geotags: {e}")
            return None
    
    def load_satellite_tile(self, satellite_file_path: str) -> bool:
        """
        Load satellite tile information from specific file.
        
        Args:
            satellite_file_path: Path to specific satellite TIFF file
            
        Returns:
            bool: Success status
        """
        print(f"\n=== LOADING SATELLITE TILE ===")
        print(f"Processing: {os.path.basename(satellite_file_path)}")
        
        if not os.path.exists(satellite_file_path):
            print(f"❌ Satellite file not found: {satellite_file_path}")
            return False
        
        # Extract spatial bounds
        bounds = self.extract_satellite_bounds(satellite_file_path)
        
        if bounds:
            tile_info = {
                'file_path': satellite_file_path,
                'bounds': bounds,
                'polygon': box(bounds['west'], bounds['south'], 
                             bounds['east'], bounds['north']),
                'area': (bounds['east'] - bounds['west']) * (bounds['north'] - bounds['south'])
            }
            
            self.satellite_tiles = [tile_info]  # Single tile
            
            print(f"   ✅ Bounds: {bounds['west']:.3f}°E to {bounds['east']:.3f}°E, "
                  f"{bounds['south']:.3f}°N to {bounds['north']:.3f}°N")
            
            self.stats['total_tiles'] = 1
            print(f"\n✅ Loaded satellite tile: {os.path.basename(satellite_file_path)}")
            return True
        else:
            print(f"   ❌ Failed to extract bounds")
            return False
    
    def compute_intersections(self) -> bool:
        """
        Compute intersections between satellite tiles and MPAs.
        
        Implements: Process tile ⟺ Area(Tile∩MPA) > 0
        
        Returns:
            bool: Success status
        """
        print("\n=== COMPUTING TILE-MPA INTERSECTIONS ===")
        
        if not self.satellite_tiles or not self.mpa_boundaries:
            print("❌ Missing satellite tiles or MPA boundaries")
            return False
        
        self.intersecting_pairs = []
        total_overlap_area = 0.0
        
        for tile_idx, tile in enumerate(self.satellite_tiles):
            tile_polygon = tile['polygon']
            tile_intersections = []
            
            print(f"\nTile {tile_idx + 1}: {os.path.basename(tile['file_path'])}")
            
            for feature in self.mpa_boundaries['features']:
                mpa_name = feature['properties'].get('mpa_name', 'Unknown')
                ocean_region = feature['properties'].get('ocean_region', 'Unknown')
                
                try:
                    # Create MPA polygon
                    if feature['geometry']['type'] == 'Polygon':
                        coords = feature['geometry']['coordinates'][0]
                    elif feature['geometry']['type'] == 'MultiPolygon':
                        coords = feature['geometry']['coordinates'][0][0]
                    else:
                        continue
                    
                    mpa_polygon = Polygon(coords)
                    
                    # 🎯 CORE MATHEMATICAL CONDITION: Area(Tile∩MPA) > 0
                    if tile_polygon.intersects(mpa_polygon):
                        # Calculate intersection area
                        intersection = tile_polygon.intersection(mpa_polygon)
                        overlap_area = intersection.area
                        
                        # Only process if overlap area > 0 (mathematical condition)
                        if overlap_area > 0:
                            mpa_area = mpa_polygon.area
                            tile_area = tile_polygon.area
                            
                            # Calculate coverage percentages
                            mpa_coverage_percent = (overlap_area / mpa_area) * 100
                            tile_coverage_percent = (overlap_area / tile_area) * 100
                            
                            # Check minimum overlap threshold
                            if mpa_coverage_percent >= self.min_overlap_percent:
                                intersection_info = {
                                    'tile_index': tile_idx,
                                    'tile_file': tile['file_path'],
                                    'mpa_name': mpa_name,
                                    'ocean_region': ocean_region,
                                    'mpa_polygon': mpa_polygon,
                                    'tile_polygon': tile_polygon,
                                    'intersection_polygon': intersection,
                                    'overlap_area': overlap_area,
                                    'mpa_area': mpa_area,
                                    'tile_area': tile_area,
                                    'mpa_coverage_percent': mpa_coverage_percent,
                                    'tile_coverage_percent': tile_coverage_percent,
                                    'mpa_properties': feature['properties']
                                }
                                
                                tile_intersections.append(intersection_info)
                                total_overlap_area += overlap_area
                                
                                print(f"   ✅ {mpa_name} ({ocean_region}): "
                                      f"{mpa_coverage_percent:.2f}% MPA coverage, "
                                      f"{tile_coverage_percent:.2f}% tile coverage")
                            else:
                                print(f"   ⚠️  {mpa_name}: {mpa_coverage_percent:.2f}% coverage "
                                      f"(below {self.min_overlap_percent}% threshold)")
                        
                except Exception as e:
                    print(f"   ❌ Error processing {mpa_name}: {e}")
                    continue
            
            if tile_intersections:
                self.intersecting_pairs.extend(tile_intersections)
                print(f"   📊 Total intersections for this tile: {len(tile_intersections)}")
            else:
                print(f"   ❌ No MPA intersections found for this tile")
        
        # Update statistics
        self.stats['intersecting_tiles'] = len(set(pair['tile_index'] for pair in self.intersecting_pairs))
        self.stats['processing_regions'] = len(self.intersecting_pairs)
        self.stats['total_overlap_area'] = total_overlap_area
        
        print(f"\n🎯 INTERSECTION SUMMARY:")
        print(f"   Total satellite tiles: {self.stats['total_tiles']}")
        print(f"   Total MPAs: {self.stats['total_mpas']}")
        print(f"   Tiles with MPA intersections: {self.stats['intersecting_tiles']}")
        print(f"   Total processing regions: {self.stats['processing_regions']}")
        print(f"   Total overlap area: {total_overlap_area:.6f} square degrees")
        
        return len(self.intersecting_pairs) > 0
    
    def extract_processing_regions(self) -> bool:
        """
        Extract satellite data regions for each MPA intersection.
        
        Returns:
            bool: Success status
        """
        print("\n=== EXTRACTING PROCESSING REGIONS ===")
        
        if not self.intersecting_pairs:
            print("❌ No intersecting pairs to process")
            return False
        
        self.processing_regions = {}
        
        for pair in self.intersecting_pairs:
            mpa_name = pair['mpa_name']
            tile_file = pair['tile_file']
            mpa_polygon = pair['mpa_polygon']
            
            print(f"\nExtracting region: {mpa_name}")
            print(f"   Satellite file: {os.path.basename(tile_file)}")
            
            try:
                # Load satellite data
                with rasterio.open(tile_file) as src:
                    # Check memory requirements
                    height, width = src.height, src.width
                    memory_gb = (height * width * 4) / (1024**3)
                    
                    if memory_gb > 1.0:
                        downsample_factor = int(np.ceil(np.sqrt(memory_gb)))
                        print(f"   Downsampling by factor {downsample_factor} for memory efficiency")
                        
                        satellite_data = src.read(1, 
                                                out_shape=(height // downsample_factor, 
                                                         width // downsample_factor),
                                                resampling=rasterio.enums.Resampling.average).astype(np.float32)
                        
                        # Adjust transform
                        transform = src.transform * src.transform.scale(downsample_factor, downsample_factor)
                    else:
                        satellite_data = src.read(1).astype(np.float32)
                        transform = src.transform
                    
                    # Handle invalid values
                    satellite_data[satellite_data <= 0] = np.nan
                    satellite_data[np.isinf(satellite_data)] = np.nan
                    
                    # Get MPA bounding box with buffer
                    minx, miny, maxx, maxy = mpa_polygon.bounds
                    
                    # Add buffer (convert km to degrees approximately)
                    buffer_deg = self.buffer_km / 111.32  # Rough conversion
                    minx -= buffer_deg
                    maxx += buffer_deg
                    miny -= buffer_deg
                    maxy += buffer_deg
                    
                    # Store processing region information
                    region_info = {
                        'mpa_name': mpa_name,
                        'satellite_file': tile_file,
                        'satellite_data': satellite_data,
                        'transform': transform,
                        'mpa_polygon': mpa_polygon,
                        'buffered_bounds': (minx, miny, maxx, maxy),
                        'buffer_km': self.buffer_km,
                        'intersection_info': pair,
                        'data_shape': satellite_data.shape,
                        'valid_pixels': np.sum(~np.isnan(satellite_data))
                    }
                    
                    self.processing_regions[mpa_name] = region_info
                    
                    print(f"   ✅ Region extracted: {satellite_data.shape}")
                    print(f"   Valid pixels: {np.sum(~np.isnan(satellite_data)):,}")
                    print(f"   Buffer: {self.buffer_km} km ({buffer_deg:.4f}°)")
                    
            except Exception as e:
                print(f"   ❌ Error extracting region for {mpa_name}: {e}")
                continue
        
        print(f"\n✅ Extracted {len(self.processing_regions)} processing regions")
        return len(self.processing_regions) > 0
    
    def get_tiles_for_processing(self) -> List[str]:
        """
        Get list of satellite tiles that should be processed.
        
        Based on mathematical condition: Area(Tile∩MPA) > 0
        
        Returns:
            List of satellite file paths that intersect with MPAs
        """
        if not self.intersecting_pairs:
            return []
        
        # Get unique tile files that have MPA intersections
        processing_tiles = list(set(pair['tile_file'] for pair in self.intersecting_pairs))
        
        return processing_tiles
    
    def get_mpa_processing_info(self, mpa_name: str) -> Optional[Dict]:
        """
        Get processing information for a specific MPA.
        
        Args:
            mpa_name: Name of the MPA
            
        Returns:
            Processing region information or None if not found
        """
        return self.processing_regions.get(mpa_name)
    
    def save_intersection_results(self, output_file: str = 'sat_mpa_intersections.json') -> bool:
        """
        Save intersection analysis results to JSON file.
        
        Args:
            output_file: Output file path
            
        Returns:
            bool: Success status
        """
        print(f"\n=== SAVING INTERSECTION RESULTS ===")
        
        try:
            # Prepare serializable data
            results = {
                'analysis_parameters': {
                    'buffer_km': self.buffer_km,
                    'min_overlap_percent': self.min_overlap_percent
                },
                'statistics': self.stats,
                'intersecting_pairs': []
            }
            
            # Convert intersection pairs to serializable format
            for pair in self.intersecting_pairs:
                serializable_pair = {
                    'tile_index': pair['tile_index'],
                    'tile_file': pair['tile_file'],
                    'mpa_name': pair['mpa_name'],
                    'ocean_region': pair['ocean_region'],
                    'overlap_area': pair['overlap_area'],
                    'mpa_area': pair['mpa_area'],
                    'tile_area': pair['tile_area'],
                    'mpa_coverage_percent': pair['mpa_coverage_percent'],
                    'tile_coverage_percent': pair['tile_coverage_percent'],
                    'mpa_properties': pair['mpa_properties']
                }
                results['intersecting_pairs'].append(serializable_pair)
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def print_processing_summary(self):
        """Print comprehensive processing summary."""
        print("\n" + "="*70)
        print("SATELLITE-MPA INTERSECTION ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\n📊 MATHEMATICAL CONDITION: Process tile ⟺ Area(Tile∩MPA) > 0")
        print(f"\n📈 STATISTICS:")
        print(f"   Total satellite tiles loaded: {self.stats['total_tiles']}")
        print(f"   Total MPA boundaries loaded: {self.stats['total_mpas']}")
        print(f"   Tiles with MPA intersections: {self.stats['intersecting_tiles']}")
        print(f"   Total processing regions: {self.stats['processing_regions']}")
        print(f"   Total overlap area: {self.stats['total_overlap_area']:.6f} square degrees")
        
        if self.intersecting_pairs:
            print(f"\n🎯 PROCESSING REGIONS:")
            for pair in self.intersecting_pairs:
                print(f"   • {pair['mpa_name']} ({pair['ocean_region']})")
                print(f"     Tile: {os.path.basename(pair['tile_file'])}")
                print(f"     MPA Coverage: {pair['mpa_coverage_percent']:.2f}%")
                print(f"     Tile Coverage: {pair['tile_coverage_percent']:.2f}%")
                print(f"     Overlap Area: {pair['overlap_area']:.6f} sq degrees")
        
        processing_tiles = self.get_tiles_for_processing()
        if processing_tiles:
            print(f"\n📁 TILES FOR PROCESSING:")
            for tile in processing_tiles:
                print(f"   • {os.path.basename(tile)}")
        
        print(f"\n✅ READY FOR VESSEL DETECTION IN {len(self.processing_regions)} MPA REGIONS")
        print("="*70)
    
    def run_analysis_for_satellite(self, 
                                  satellite_file_path: str,
                                  mpa_file_path: str = 'data/raw/mpa_boundaries/Combined_MPA_Boundaries.geojson') -> bool:
        """
        Run satellite-MPA intersection analysis for specific satellite file.
        
        Args:
            satellite_file_path: Path to specific satellite TIFF file
            mpa_file_path: Path to MPA boundaries file
            
        Returns:
            bool: Success status
        """
        satellite_name = self._extract_satellite_name(satellite_file_path)
        
        print(f"🛰️ SATELLITE-MPA INTERSECTION ANALYZER - {satellite_name.upper()}")
        print("="*70)
        print("Implementing: Process tile ⟺ Area(Tile∩MPA) > 0")
        
        # Execute analysis pipeline
        success = True
        success &= self.load_mpa_boundaries(mpa_file_path)
        success &= self.load_satellite_tile(satellite_file_path)
        success &= self.compute_intersections()
        success &= self.extract_processing_regions()
        
        if success:
            self.save_intersection_results(satellite_name)
            self.print_processing_summary()
            print(f"\n🎉 SATELLITE-MPA INTERSECTION ANALYSIS COMPLETE FOR {satellite_name.upper()}!")
        else:
            print(f"\n❌ Intersection analysis failed for {satellite_name}. Check error messages above.")
        
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
            from datetime import datetime
            return f"sat_{datetime.now().strftime('%H%M%S')}"
    
    def save_intersection_results(self, satellite_name: str, output_file: Optional[str] = None) -> bool:
        """
        Save intersection analysis results to JSON file with satellite-specific naming.
        
        Args:
            satellite_name: Name of the satellite
            output_file: Optional custom output file path
            
        Returns:
            bool: Success status
        """
        if output_file is None:
            output_file = f'output/json/sat_mpa_intersections_{satellite_name}.json'
        
        print(f"\n=== SAVING INTERSECTION RESULTS FOR {satellite_name.upper()} ===")
        
        try:
            # Prepare serializable data
            results = {
                'satellite_name': satellite_name,
                'analysis_parameters': {
                    'buffer_km': self.buffer_km,
                    'min_overlap_percent': self.min_overlap_percent
                },
                'statistics': self.stats,
                'intersecting_pairs': []
            }
            
            # Convert intersection pairs to serializable format
            for pair in self.intersecting_pairs:
                serializable_pair = {
                    'tile_index': pair['tile_index'],
                    'tile_file': pair['tile_file'],
                    'mpa_name': pair['mpa_name'],
                    'ocean_region': pair['ocean_region'],
                    'overlap_area': pair['overlap_area'],
                    'mpa_area': pair['mpa_area'],
                    'tile_area': pair['tile_area'],
                    'mpa_coverage_percent': pair['mpa_coverage_percent'],
                    'tile_coverage_percent': pair['tile_coverage_percent'],
                    'mpa_properties': pair['mpa_properties']
                }
                results['intersecting_pairs'].append(serializable_pair)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save to file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Results saved to: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False

def main():
    """Main execution function with command line argument support."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Satellite-MPA Intersection Analysis')
    parser.add_argument('--satellite-path', required=True, 
                       help='Path to satellite TIFF file')
    parser.add_argument('--mpa-boundaries', 
                       default='data/raw/mpa_boundaries/Combined_MPA_Boundaries.geojson',
                       help='Path to MPA boundaries file')
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        print("❌ Missing required dependencies. Please install:")
        print("pip install rasterio shapely geopandas")
        return
    
    # Initialize intersector
    intersector = SatelliteMPAIntersector(
        buffer_km=10.0,
        min_overlap_percent=0.1
    )
    
    # Run analysis for specific satellite
    success = intersector.run_analysis_for_satellite(
        satellite_file_path=args.satellite_path,
        mpa_file_path=args.mpa_boundaries
    )
    
    if success:
        satellite_name = intersector._extract_satellite_name(args.satellite_path)
        
        # Example: Get tiles that need processing
        processing_tiles = intersector.get_tiles_for_processing()
        print(f"\n🎯 RESULT FOR {satellite_name.upper()}: {len(processing_tiles)} tiles require processing")
        
        # Example: Get specific MPA processing info
        for mpa_name in intersector.processing_regions.keys():
            region_info = intersector.get_mpa_processing_info(mpa_name)
            if region_info:
                print(f"   {mpa_name}: {region_info['data_shape']} pixels ready for vessel detection")
    else:
        print(f"\n❌ Analysis failed for satellite: {args.satellite_path}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # New command-line mode
        main()
    else:
        # Legacy mode - process first available satellite
        print("🛰️ SATELLITE-MPA INTERSECTION ANALYZER (LEGACY MODE)")
        print("="*70)
        print("⚠️  Running in legacy mode - processing first available satellite")
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Missing required dependencies. Please install:")
            print("pip install rasterio shapely geopandas")
            sys.exit(1)
        
        # Initialize intersector
        intersector = SatelliteMPAIntersector(
            buffer_km=10.0,
            min_overlap_percent=0.1
        )
        
        # Try to find first available satellite
        import glob
        tiff_files = glob.glob('data/raw/satellite/*/measurement/*vv*.tiff')
        if tiff_files:
            satellite_file_path = tiff_files[0]
            success = intersector.run_analysis_for_satellite(satellite_file_path)
            
            if success:
                satellite_name = intersector._extract_satellite_name(satellite_file_path)
                processing_tiles = intersector.get_tiles_for_processing()
                print(f"\n🎯 RESULT FOR {satellite_name.upper()}: {len(processing_tiles)} tiles require processing")
        else:
            print("❌ No satellite files found")
            sys.exit(1)