#!/usr/bin/env python3
"""
CNN Dataset Generator for Vessel Detection Confirmation

Generates a CNN-ready dataset from physics-based SBCI vessel detections:
- Ship patches: 64x64 SAR image patches centered on detected vessels
- Sea patches: 64x64 SAR image patches from vessel-free ocean areas
- Labels file: JSON mapping images to classes with metadata

This provides ground truth data for CNN training based on explainable physics detections.

Usage:
    python cnn_dataset_generator.py --satellite-path <path_to_satellite_tiff>
"""
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
import random
from datetime import datetime

try:
    import rasterio
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install rasterio opencv-python pillow matplotlib")
    DEPENDENCIES_AVAILABLE = False

class CNNDatasetGenerator:
    """Generate CNN training dataset from physics-based vessel detections"""
    
    def __init__(self, patch_size=64, output_dir='cnn_dataset', sea_patches_per_satellite=10):
        """
        Initialize CNN dataset generator
        
        Args:
            patch_size: Size of image patches (patch_size x patch_size)
            output_dir: Output directory for CNN dataset
            sea_patches_per_satellite: Fixed number of sea patches to generate per satellite (default: 10)
        """
        self.patch_size = patch_size
        self.output_dir = output_dir
        self.half_patch = patch_size // 2
        self.sea_patches_per_satellite = sea_patches_per_satellite
        
        # Data containers
        self.sar_data = None
        self.vessel_detections = []
        self.ship_patches = []
        self.sea_patches = []
        self.labels = []
        
        # Statistics
        self.stats = {
            'total_ships': 0,
            'total_sea_patches': 0,
            'target_sea_patches': sea_patches_per_satellite,
            'patch_size': patch_size,
            'generation_time': None
        }
    
    def load_detection_results(self, 
                             satellite_file_path: str,
                             satellite_name: str):
        """Load SAR data and vessel detection results with AIS status for specific satellite"""
        print(f"=== LOADING DETECTION RESULTS FOR {satellite_name.upper()} ===")
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Missing required dependencies")
            return False
        
        # Load SAR data from specific satellite file
        if not os.path.exists(satellite_file_path):
            print(f"❌ Satellite file not found: {satellite_file_path}")
            return False
        
        print(f"Loading SAR data: {os.path.basename(satellite_file_path)}")
        
        try:
            with rasterio.open(satellite_file_path) as src:
                # Check if we need to downsample
                height, width = src.height, src.width
                memory_gb = (height * width * 4) / (1024**3)
                
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
                
                print(f"✅ SAR data loaded: {self.sar_data.shape}")
                print(f"   Valid pixels: {np.sum(~np.isnan(self.sar_data))}")
                print(f"   Intensity range: {np.nanmin(self.sar_data):.2f} to {np.nanmax(self.sar_data):.2f}")
        
        except Exception as e:
            print(f"❌ Error loading SAR data: {e}")
            return False
        
        # Load vessel detection results with AIS status for specific satellite
        ais_status_file = f'output/json/ship_detection_with_ais_status_{satellite_name}.json'
        if not os.path.exists(ais_status_file):
            print(f"❌ AIS status file not found: {ais_status_file}")
            return False
        
        try:
            with open(ais_status_file, 'r') as f:
                ais_status_data = json.load(f)
            
            # Extract vessel candidates with AIS status
            self.vessel_detections = ais_status_data.get('vessel_candidates', [])
            
            # Add cluster assignments based on vessel characteristics
            # Since we don't have explicit clustering, we'll create clusters based on AIS status and size
            for vessel in self.vessel_detections:
                ais_status = vessel.get('ais_status', 'UNKNOWN')
                area_pixels = vessel.get('area_pixels', 0)
                
                # Simple clustering based on AIS status and vessel size
                if ais_status == 'DARK':
                    if area_pixels <= 10:
                        vessel['cluster_id'] = 0  # Small dark vessels
                    elif area_pixels <= 20:
                        vessel['cluster_id'] = 1  # Medium dark vessels
                    else:
                        vessel['cluster_id'] = 2  # Large dark vessels
                elif ais_status == 'NORMAL':
                    vessel['cluster_id'] = 3  # Normal vessels with AIS
                else:
                    vessel['cluster_id'] = 4  # Unknown status
                
                print(f"   Vessel {vessel['vessel_id']}: AIS={ais_status}, Size={area_pixels}px, Cluster={vessel['cluster_id']}")
            
            # Print summary
            summary = ais_status_data.get('summary', {})
            total_vessels = summary.get('total_radar_detections', len(self.vessel_detections))
            dark_vessels = summary.get('dark_vessels', 0)
            normal_vessels = summary.get('normal_vessels', 0)
            
            print(f"✅ Loaded {len(self.vessel_detections)} vessel detections with AIS status for {satellite_name}")
            print(f"   Total vessels: {total_vessels}")
            print(f"   Dark vessels: {dark_vessels}")
            print(f"   Normal vessels: {normal_vessels}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading AIS status file: {e}")
            return False
    
    def extract_ship_patches(self):
        """Extract 64x64 patches centered on detected vessels"""
        print("\n=== EXTRACTING SHIP PATCHES ===")
        
        if self.sar_data is None:
            print("❌ Missing SAR data")
            return False
        
        if not self.vessel_detections:
            print("No vessel detections found - will generate sea-only dataset")
            self.ship_patches = []
            self.stats['total_ships'] = 0
            return True
        
        self.ship_patches = []
        height, width = self.sar_data.shape
        
        for i, vessel in enumerate(self.vessel_detections):
            center_x = int(vessel['pixel_x'])
            center_y = int(vessel['pixel_y'])
            
            # Check if patch fits within image bounds
            if (center_x - self.half_patch >= 0 and center_x + self.half_patch < width and
                center_y - self.half_patch >= 0 and center_y + self.half_patch < height):
                
                # Extract patch
                patch = self.sar_data[center_y - self.half_patch:center_y + self.half_patch,
                                    center_x - self.half_patch:center_x + self.half_patch]
                
                # Check if patch has valid data
                if not np.all(np.isnan(patch)) and patch.shape == (self.patch_size, self.patch_size):
                    # Normalize patch to [0, 1]
                    patch_normalized = self._normalize_patch(patch)
                    
                    if patch_normalized is not None:
                        patch_info = {
                            'patch': patch_normalized,
                            'vessel_id': vessel['vessel_id'],
                            'ais_status': vessel.get('ais_status', 'UNKNOWN'),
                            'cluster_id': vessel.get('cluster_id', -1),
                            'center_x': center_x,
                            'center_y': center_y,
                            'area_pixels': vessel.get('area_pixels', 0),
                            'mean_sbci': vessel.get('mean_sbci', 0.0),
                            'max_sbci': vessel.get('max_sbci', 0.0),
                            'detection_confidence': vessel.get('detection_confidence', 0.0),
                            'latitude': vessel.get('latitude'),
                            'longitude': vessel.get('longitude'),
                            'satellite_source': getattr(self, 'current_satellite', 'unknown')
                        }
                        
                        self.ship_patches.append(patch_info)
                        print(f"✅ Ship patch {i+1}: Vessel {vessel['vessel_id']} - AIS: {vessel.get('ais_status', 'UNKNOWN')}")
                else:
                    print(f"⚠️  Invalid patch for vessel {vessel['vessel_id']}")
            else:
                print(f"⚠️  Vessel {vessel['vessel_id']} too close to image edge")
        
        print(f"✅ Extracted {len(self.ship_patches)} ship patches")
        self.stats['total_ships'] = len(self.ship_patches)
        
        return True
    
    def extract_sea_patches(self):
        """Extract sea-only patches (negative samples) - Fixed 10 patches per satellite"""
        print("\n=== EXTRACTING SEA PATCHES ===")
        
        if self.sar_data is None:
            print("❌ Missing SAR data")
            return False
        
        num_ship_patches = len(self.ship_patches)
        
        # Fixed number of sea patches per satellite image
        target_sea_patches = self.sea_patches_per_satellite
        
        print(f"Generating {target_sea_patches} sea patches per satellite (fixed)")
        print(f"Ship patches detected: {num_ship_patches}")
        print(f"Sea:Ship ratio will be {target_sea_patches}:{num_ship_patches}")
        
        height, width = self.sar_data.shape
        self.sea_patches = []
        
        # Create exclusion zones around detected vessels (if any)
        exclusion_zones = []
        for vessel in self.vessel_detections:
            center_x = int(vessel['pixel_x'])
            center_y = int(vessel['pixel_y'])
            
            # Create larger exclusion zone (2x patch size) around each vessel
            exclusion_size = self.patch_size
            exclusion_zones.append({
                'x_min': max(0, center_x - exclusion_size),
                'x_max': min(width, center_x + exclusion_size),
                'y_min': max(0, center_y - exclusion_size),
                'y_max': min(height, center_y + exclusion_size)
            })
        
        print(f"   Created {len(exclusion_zones)} exclusion zones around detected vessels")
        
        # Generate sea patches
        attempts = 0
        max_attempts = target_sea_patches * 20  # More attempts since we have a fixed target
        
        while len(self.sea_patches) < target_sea_patches and attempts < max_attempts:
            attempts += 1
            
            # Random location
            center_x = random.randint(self.half_patch, width - self.half_patch - 1)
            center_y = random.randint(self.half_patch, height - self.half_patch - 1)
            
            # Check if location overlaps with any vessel exclusion zone
            overlaps_vessel = False
            for zone in exclusion_zones:
                if (zone['x_min'] <= center_x <= zone['x_max'] and
                    zone['y_min'] <= center_y <= zone['y_max']):
                    overlaps_vessel = True
                    break
            
            if not overlaps_vessel:
                # Extract patch
                patch = self.sar_data[center_y - self.half_patch:center_y + self.half_patch,
                                    center_x - self.half_patch:center_x + self.half_patch]
                
                # Check if patch has valid data
                if not np.all(np.isnan(patch)) and patch.shape == (self.patch_size, self.patch_size):
                    # Normalize patch
                    patch_normalized = self._normalize_patch(patch)
                    
                    if patch_normalized is not None:
                        patch_info = {
                            'patch': patch_normalized,
                            'center_x': center_x,
                            'center_y': center_y,
                            'patch_id': len(self.sea_patches) + 1
                        }
                        
                        self.sea_patches.append(patch_info)
                        
                        if len(self.sea_patches) % 5 == 0:
                            print(f"   Generated {len(self.sea_patches)}/{target_sea_patches} sea patches")
        
        print(f"✅ Extracted {len(self.sea_patches)} sea patches (target: {target_sea_patches})")
        print(f"   Attempts: {attempts}/{max_attempts}")
        print(f"   Success rate: {len(self.sea_patches)/attempts*100:.1f}%")
        
        if len(self.sea_patches) < target_sea_patches:
            print(f"⚠️  Generated fewer sea patches than target ({len(self.sea_patches)}/{target_sea_patches})")
            print(f"   This may be due to limited valid sea areas or too many exclusion zones")
        
        self.stats['total_sea_patches'] = len(self.sea_patches)
        self.stats['target_sea_patches'] = target_sea_patches
        self.stats['sea_patch_success_rate'] = len(self.sea_patches) / target_sea_patches * 100
        
        return len(self.sea_patches) > 0
    
    def _normalize_patch(self, patch):
        """Normalize patch to [0, 1] range"""
        try:
            # Remove NaN values for normalization
            valid_pixels = patch[~np.isnan(patch)]
            
            if len(valid_pixels) == 0:
                return None
            
            # Use percentile-based normalization for robustness
            p1, p99 = np.percentile(valid_pixels, [1, 99])
            
            if p99 <= p1:
                return None
            
            # Normalize to [0, 1]
            patch_norm = np.clip((patch - p1) / (p99 - p1), 0, 1)
            
            # Fill NaN values with 0
            patch_norm[np.isnan(patch_norm)] = 0
            
            return patch_norm.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️  Normalization error: {e}")
            return None
    
    def load_ais_results(self, ais_integration_path='output/json/ship_detection_with_ais_status.json'):
        """Load AIS integration data - now integrated in main input file"""
        print("\n=== AIS STATUS ALREADY INTEGRATED ===")
        print("✅ AIS status information is already included in the input file")
        return {}
    
    def save_dataset(self):
        """Save CNN dataset to disk with incremental support"""
        print("\n=== SAVING CNN DATASET (INCREMENTAL MODE) ===")
        
        if not self.sea_patches:
            print("❌ No sea patches to save")
            return False
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        
        # Load existing labels if they exist
        labels_file = os.path.join(self.output_dir, 'labels.json')
        existing_labels = []
        existing_ship_count = 0
        existing_sea_count = 0
        
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    existing_labels = json.load(f)
                
                # Count existing patches
                existing_ship_count = len([l for l in existing_labels if l.get('label') == 1])
                existing_sea_count = len([l for l in existing_labels if l.get('label') == 0])
                
                print(f"📂 Found existing dataset:")
                print(f"   Existing ship patches: {existing_ship_count}")
                print(f"   Existing sea patches: {existing_sea_count}")
                print(f"   Total existing patches: {len(existing_labels)}")
                
            except Exception as e:
                print(f"⚠️  Error loading existing labels: {e}")
                existing_labels = []
        else:
            print("📂 No existing dataset found - creating new one")
        
        # Load AIS results (now integrated in input)
        ais_lookup = self.load_ais_results()
        
        # Start with existing labels
        self.labels = existing_labels.copy()
        new_labels_added = 0
        
        # Save ship patches (if any exist) with incremental naming
        if self.ship_patches:
            print(f"Adding {len(self.ship_patches)} new ship patches...")
            for i, ship_patch in enumerate(self.ship_patches):
                # Generate incremental filename
                ship_number = existing_ship_count + i + 1
                filename = f"ship_{ship_number}.png"
                filepath = os.path.join(self.output_dir, 'images', filename)
                
                # Check if file already exists (avoid duplicates)
                if os.path.exists(filepath):
                    print(f"   ⚠️  Skipping {filename} (already exists)")
                    continue
                
                # Convert to 8-bit image and save
                patch_8bit = (ship_patch['patch'] * 255).astype(np.uint8)
                Image.fromarray(patch_8bit, mode='L').save(filepath)
                
                # Get vessel information (now directly from ship_patch)
                vessel_id = ship_patch['vessel_id']
                cluster_id = ship_patch.get('cluster_id', -1)
                ais_status = ship_patch.get('ais_status', 'UNKNOWN')
                satellite_source = ship_patch.get('satellite_source', 'unknown')
                
                # Create label entry with all available information
                label_entry = {
                    "image": filename,
                    "label": 1,  # Ship class
                    "vessel_id": vessel_id,
                    "cluster_id": cluster_id,
                    "ais_status": ais_status,
                    "area_pixels": ship_patch.get('area_pixels', 0),
                    "mean_sbci": round(ship_patch['mean_sbci'], 6),
                    "max_sbci": round(ship_patch['max_sbci'], 6),
                    "detection_confidence": round(ship_patch['detection_confidence'], 6),
                    "center_coordinates": [ship_patch['center_x'], ship_patch['center_y']],
                    "geographic_coordinates": [ship_patch.get('latitude'), ship_patch.get('longitude')],
                    "satellite_source": satellite_source,
                    "added_timestamp": datetime.now().isoformat()
                }
                
                self.labels.append(label_entry)
                new_labels_added += 1
                print(f"   ✅ Added ship_{ship_number}.png: Vessel {vessel_id} (Cluster {cluster_id}, AIS: {ais_status}, Satellite: {satellite_source})")
        else:
            print("No ship patches to add")
        
        # Save sea patches with incremental naming
        print(f"Adding {len(self.sea_patches)} new sea patches...")
        for i, sea_patch in enumerate(self.sea_patches):
            # Generate incremental filename
            sea_number = existing_sea_count + i + 1
            filename = f"sea_{sea_number}.png"
            filepath = os.path.join(self.output_dir, 'images', filename)
            
            # Check if file already exists (avoid duplicates)
            if os.path.exists(filepath):
                print(f"   ⚠️  Skipping {filename} (already exists)")
                continue
            
            # Convert to 8-bit image and save
            patch_8bit = (sea_patch['patch'] * 255).astype(np.uint8)
            Image.fromarray(patch_8bit, mode='L').save(filepath)
            
            # Create label entry
            label_entry = {
                "image": filename,
                "label": 0,  # Sea class
                "center_coordinates": [sea_patch['center_x'], sea_patch['center_y']],
                "satellite_source": getattr(self, 'current_satellite', 'unknown'),
                "added_timestamp": datetime.now().isoformat()
            }
            
            self.labels.append(label_entry)
            new_labels_added += 1
            print(f"   ✅ Added sea_{sea_number}.png")
        
        # Save updated labels file
        with open(labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        
        print(f"✅ Updated labels file: {labels_file}")
        print(f"📊 Dataset Update Summary:")
        print(f"   New patches added: {new_labels_added}")
        print(f"   Total patches now: {len(self.labels)}")
        
        # Update dataset statistics
        total_ship_patches = len([l for l in self.labels if l.get('label') == 1])
        total_sea_patches = len([l for l in self.labels if l.get('label') == 0])
        
        self.stats['generation_time'] = datetime.now().isoformat()
        self.stats['total_ships'] = len(self.ship_patches)  # Current batch
        self.stats['total_sea_patches'] = len(self.sea_patches)  # Current batch
        self.stats['cumulative_ship_patches'] = total_ship_patches  # All time
        self.stats['cumulative_sea_patches'] = total_sea_patches  # All time
        self.stats['cumulative_total_patches'] = len(self.labels)  # All time
        
        stats_file = os.path.join(self.output_dir, 'dataset_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ Updated dataset statistics: {stats_file}")
        
        return True
    
    def create_sample_visualization(self):
        """Create visualization showing sample patches"""
        print("\n=== CREATING SAMPLE VISUALIZATION ===")
        
        if not self.sea_patches:
            print("❌ No patches to visualize")
            return False
        
        try:
            # Determine layout based on available patches
            has_ships = len(self.ship_patches) > 0
            
            if has_ships:
                # Create figure with sample patches (ships + sea)
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle('CNN Dataset Sample Patches', fontsize=16, fontweight='bold')
                
                # Show ship patches
                for i in range(min(4, len(self.ship_patches))):
                    ax = axes[0, i]
                    patch = self.ship_patches[i]['patch']
                    vessel_id = self.ship_patches[i]['vessel_id']
                    cluster_id = self.ship_patches[i]['cluster_id']
                    
                    ax.imshow(patch, cmap='gray')
                    ax.set_title(f'Ship {vessel_id}\n(Cluster {cluster_id})', fontsize=10)
                    ax.axis('off')
                
                # Fill remaining ship slots if needed
                for i in range(len(self.ship_patches), 4):
                    axes[0, i].axis('off')
                
                # Show sea patches
                for i in range(min(4, len(self.sea_patches))):
                    ax = axes[1, i]
                    patch = self.sea_patches[i]['patch']
                    patch_id = self.sea_patches[i]['patch_id']
                    
                    ax.imshow(patch, cmap='gray')
                    ax.set_title(f'Sea {patch_id}', fontsize=10)
                    ax.axis('off')
                
                # Add row labels
                axes[0, 0].text(-0.1, 0.5, 'SHIPS\n(Label: 1)', transform=axes[0, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)
                axes[1, 0].text(-0.1, 0.5, 'SEA\n(Label: 0)', transform=axes[1, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)
            else:
                # Sea-only dataset visualization
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle('CNN Dataset Sample Patches (Sea-Only Dataset)', fontsize=16, fontweight='bold')
                
                # Show sea patches in both rows
                for row in range(2):
                    for col in range(4):
                        patch_idx = row * 4 + col
                        if patch_idx < len(self.sea_patches):
                            ax = axes[row, col]
                            patch = self.sea_patches[patch_idx]['patch']
                            patch_id = self.sea_patches[patch_idx]['patch_id']
                            
                            ax.imshow(patch, cmap='gray')
                            ax.set_title(f'Sea {patch_id}', fontsize=10)
                            ax.axis('off')
                        else:
                            axes[row, col].axis('off')
                
                # Add label for sea-only dataset
                axes[0, 0].text(-0.1, 0.5, 'SEA PATCHES\n(Label: 0)', transform=axes[0, 0].transAxes,
                               fontsize=12, fontweight='bold', ha='right', va='center', rotation=90)
            
            plt.tight_layout()
            
            # Save visualization
            viz_file = os.path.join(self.output_dir, 'sample_patches.png')
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            print(f"✅ Saved sample visualization: {viz_file}")
            
            # Close the plot instead of showing it
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return False
    
    def print_dataset_summary(self):
        """Print comprehensive dataset summary"""
        print(f"\nCNN DATASET SUMMARY (INCREMENTAL):")
        
        # Current batch statistics
        current_batch_total = len(self.labels) if hasattr(self, 'labels') else 0
        current_ship_patches = self.stats.get('total_ships', 0)
        current_sea_patches = self.stats.get('total_sea_patches', 0)
        
        # Cumulative statistics
        cumulative_total = self.stats.get('cumulative_total_patches', current_batch_total)
        cumulative_ships = self.stats.get('cumulative_ship_patches', current_ship_patches)
        cumulative_sea = self.stats.get('cumulative_sea_patches', current_sea_patches)
        
        print(f"   📊 CURRENT BATCH:")
        print(f"      Ship patches: {current_ship_patches} (1 per detected vessel)")
        print(f"      Sea patches: {current_sea_patches} (target: {self.sea_patches_per_satellite} per satellite)")
        print(f"      Batch total: {current_ship_patches + current_sea_patches}")
        
        print(f"   📈 CUMULATIVE DATASET:")
        print(f"      Total images: {cumulative_total}")
        print(f"      Ship patches: {cumulative_ships} (Label: 1)")
        print(f"      Sea patches: {cumulative_sea} (Label: 0)")
        print(f"      Sea:Ship ratio: {cumulative_sea/max(cumulative_ships,1):.1f}:1")
        print(f"      Patch size: {self.patch_size}x{self.patch_size} pixels")
        print(f"      Output directory: {self.output_dir}")
        
        # Show sea patch generation success rate
        if 'sea_patch_success_rate' in self.stats:
            success_rate = self.stats['sea_patch_success_rate']
            print(f"      Sea patch success rate: {success_rate:.1f}%")
        
        # Analyze ship patch distribution if we have labels
        if hasattr(self, 'labels') and self.labels:
            # Analyze ship patch distribution
            cluster_counts = {}
            ais_counts = {}
            satellite_counts = {}
            
            for label in self.labels:
                if label['label'] == 1:  # Ship patches
                    cluster_id = label.get('cluster_id', -1)
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
                    
                    ais_status = label.get('ais_status', 'UNKNOWN')
                    ais_counts[ais_status] = ais_counts.get(ais_status, 0) + 1
                    
                    satellite = label.get('satellite_source', 'unknown')
                    satellite_counts[satellite] = satellite_counts.get(satellite, 0) + 1
            
            if cluster_counts:
                print(f"\n   🎯 SHIP PATCH ANALYSIS (CUMULATIVE):")
                print(f"      Cluster distribution:")
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"        Cluster {cluster_id}: {count} patches")
                
                print(f"      AIS status distribution:")
                for status, count in sorted(ais_counts.items()):
                    print(f"        {status}: {count} patches")
                
                print(f"      Satellite source distribution:")
                for satellite, count in sorted(satellite_counts.items()):
                    print(f"        {satellite}: {count} patches")
        
        print(f"\n   📁 OUTPUT FILES:")
        print(f"      • {self.output_dir}/images/ - {cumulative_total} PNG image patches")
        print(f"      • {self.output_dir}/labels.json - Label mapping file (incremental)")
        print(f"      • {self.output_dir}/dataset_stats.json - Dataset statistics")
        print(f"      • {self.output_dir}/sample_patches.png - Sample visualization")
        
        print(f"\n   🎯 CNN TRAINING READINESS:")
        print(f"      Fixed sea patches per satellite: {self.sea_patches_per_satellite}")
        print(f"      Balanced dataset for better CNN training")
        print(f"      Incremental dataset building supported")
    
    def generate_full_dataset(self, satellite_file_path: str, satellite_name: str):
        """Generate complete CNN dataset for specific satellite"""
        print(f"CNN DATASET GENERATOR - {satellite_name.upper()}")
        print("=" * 60)
        
        # Store current satellite for tracking
        self.current_satellite = satellite_name
        
        # Execute full pipeline
        success = True
        success &= self.load_detection_results(satellite_file_path, satellite_name)
        success &= self.extract_ship_patches()
        success &= self.extract_sea_patches()
        success &= self.save_dataset()
        success &= self.create_sample_visualization()
        
        if success:
            self.print_dataset_summary()
            print("\n" + "=" * 60)
            print(f"CNN DATASET GENERATION COMPLETE FOR {satellite_name.upper()}!")
            print("Dataset ready for CNN training with AIS status labels!")
        else:
            print(f"\nCNN dataset generation failed for {satellite_name}. Check error messages above.")
        
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

def main():
    """Main execution function with command line argument support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CNN Dataset Generator')
    parser.add_argument('--satellite-path', required=True, 
                       help='Path to satellite TIFF file')
    
    args = parser.parse_args()
    
    # Initialize CNN dataset generator
    generator = CNNDatasetGenerator(
        patch_size=64,
        output_dir='cnn_dataset',
        sea_patches_per_satellite=10  # Fixed 10 sea patches per satellite
    )
    
    # Extract satellite name from path
    satellite_name = generator._extract_satellite_name(args.satellite_path)
    
    # Generate dataset for specific satellite
    generator.generate_full_dataset(args.satellite_path, satellite_name)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # New command-line mode
        main()
    else:
        # Legacy mode - process first available satellite
        print("CNN DATASET GENERATOR (LEGACY MODE)")
        print("=" * 60)
        print("⚠️  Running in legacy mode - processing first available satellite")
        
        generator = CNNDatasetGenerator(
            patch_size=64,
            output_dir='cnn_dataset',
            sea_patches_per_satellite=10  # Fixed 10 sea patches per satellite
        )
        
        # Try to find first available satellite
        import glob
        tiff_files = glob.glob('data/raw/satellite/*/measurement/*vv*.tiff')
        if tiff_files:
            satellite_file_path = tiff_files[0]
            satellite_name = generator._extract_satellite_name(satellite_file_path)
            generator.generate_full_dataset(satellite_file_path, satellite_name)
        else:
            print("❌ No satellite files found")
            sys.exit(1)