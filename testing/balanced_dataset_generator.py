#!/usr/bin/env python3
"""
Balanced Dataset Generator for CNN Confidence Improvement

Generates additional sea patches to balance the existing CNN dataset.
Target: ~400 samples each for ship and sea classes.

Usage:
    python balanced_dataset_generator.py
"""
import numpy as np
import json
import os
import sys
import random
from datetime import datetime
from pathlib import Path

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

class BalancedDatasetGenerator:
    """Generate additional sea patches to balance the CNN dataset"""
    
    def __init__(self, patch_size=64, output_dir='cnn_dataset', target_samples_per_class=400):
        """
        Initialize balanced dataset generator
        
        Args:
            patch_size: Size of image patches (patch_size x patch_size)
            output_dir: Output directory for CNN dataset
            target_samples_per_class: Target number of samples per class
        """
        self.patch_size = patch_size
        self.output_dir = output_dir
        self.half_patch = patch_size // 2
        self.target_samples_per_class = target_samples_per_class
        
        # Data containers
        self.existing_labels = []
        self.new_sea_patches = []
        self.satellite_data_cache = {}
        
        # Statistics
        self.stats = {
            'existing_ship_count': 0,
            'existing_sea_count': 0,
            'target_sea_count': 0,
            'new_sea_patches_needed': 0,
            'new_sea_patches_generated': 0,
            'patch_size': patch_size,
            'generation_time': None
        }
    
    def analyze_current_dataset(self):
        """Analyze current dataset to determine balancing needs"""
        print("=== ANALYZING CURRENT DATASET ===")
        
        labels_file = os.path.join(self.output_dir, 'labels.json')
        if not os.path.exists(labels_file):
            print(f"❌ Labels file not found: {labels_file}")
            return False
        
        try:
            with open(labels_file, 'r') as f:
                self.existing_labels = json.load(f)
            
            # Count existing samples by class
            ship_count = len([l for l in self.existing_labels if l.get('label') == 1])
            sea_count = len([l for l in self.existing_labels if l.get('label') == 0])
            
            self.stats['existing_ship_count'] = ship_count
            self.stats['existing_sea_count'] = sea_count
            
            # Calculate target counts
            # Use the higher of ship count or target per class as the target
            target_count = max(ship_count, self.target_samples_per_class)
            self.stats['target_sea_count'] = target_count
            
            # Calculate how many new sea patches we need
            new_sea_needed = max(0, target_count - sea_count)
            self.stats['new_sea_patches_needed'] = new_sea_needed
            
            print(f"✅ Current dataset analysis:")
            print(f"   Ship patches: {ship_count}")
            print(f"   Sea patches: {sea_count}")
            print(f"   Current ratio: {sea_count/max(ship_count,1):.2f}:1 (sea:ship)")
            print(f"   Target per class: {target_count}")
            print(f"   New sea patches needed: {new_sea_needed}")
            
            if new_sea_needed == 0:
                print("✅ Dataset is already balanced!")
                return True
            
            return True
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
            return False
    
    def load_satellite_data(self, satellite_path):
        """Load and cache SAR data from satellite file"""
        if satellite_path in self.satellite_data_cache:
            return self.satellite_data_cache[satellite_path]
        
        if not os.path.exists(satellite_path):
            print(f"❌ Satellite file not found: {satellite_path}")
            return None
        
        try:
            with rasterio.open(satellite_path) as src:
                # Check if we need to downsample for memory efficiency
                height, width = src.height, src.width
                memory_gb = (height * width * 4) / (1024**3)
                
                if memory_gb > 1.0:
                    downsample_factor = int(np.ceil(np.sqrt(memory_gb)))
                    print(f"   Downsampling {os.path.basename(satellite_path)} by factor {downsample_factor}")
                    
                    sar_data = src.read(1, 
                                       out_shape=(height // downsample_factor, 
                                                width // downsample_factor),
                                       resampling=rasterio.enums.Resampling.average).astype(np.float32)
                else:
                    sar_data = src.read(1).astype(np.float32)
                
                # Handle invalid values
                sar_data[sar_data <= 0] = np.nan
                sar_data[np.isinf(sar_data)] = np.nan
                
                # Cache the data
                self.satellite_data_cache[satellite_path] = sar_data
                
                print(f"✅ Loaded SAR data: {sar_data.shape} from {os.path.basename(satellite_path)}")
                return sar_data
                
        except Exception as e:
            print(f"❌ Error loading SAR data from {satellite_path}: {e}")
            return None
    
    def find_available_satellites(self):
        """Find all available satellite data files"""
        satellite_files = []
        
        # Look for satellite data in the standard location
        satellite_dir = "data/raw/satellite"
        if os.path.exists(satellite_dir):
            # Find all VV polarization TIFF files
            import glob
            pattern = os.path.join(satellite_dir, "*/measurement/*vv*.tiff")
            satellite_files = glob.glob(pattern)
        
        if not satellite_files:
            print("❌ No satellite data files found")
            return []
        
        print(f"✅ Found {len(satellite_files)} satellite data files:")
        for i, file_path in enumerate(satellite_files):
            sat_name = self._extract_satellite_name(file_path)
            print(f"   {i+1}. {sat_name}: {os.path.basename(file_path)}")
        
        return satellite_files
    
    def _extract_satellite_name(self, satellite_file_path):
        """Extract satellite name from file path"""
        try:
            path_parts = satellite_file_path.replace('\\', '/').split('/')
            for part in path_parts:
                if part.startswith('sat') and part[3:].isdigit():
                    return part
            return Path(satellite_file_path).parent.parent.name
        except:
            return f"sat_unknown"
    
    def get_existing_vessel_locations(self):
        """Get locations of existing vessels to create exclusion zones"""
        vessel_locations = []
        
        for label in self.existing_labels:
            if label.get('label') == 1:  # Ship patches
                center_coords = label.get('center_coordinates', [])
                if len(center_coords) == 2:
                    vessel_locations.append({
                        'x': center_coords[0],
                        'y': center_coords[1],
                        'satellite': label.get('satellite_source', 'unknown')
                    })
        
        print(f"✅ Found {len(vessel_locations)} existing vessel locations for exclusion zones")
        return vessel_locations
    
    def generate_diverse_sea_patches(self):
        """Generate diverse sea patches from multiple satellite sources"""
        print("\n=== GENERATING DIVERSE SEA PATCHES ===")
        
        if self.stats['new_sea_patches_needed'] == 0:
            print("✅ No new sea patches needed - dataset already balanced")
            return True
        
        # Find available satellite data
        satellite_files = self.find_available_satellites()
        if not satellite_files:
            return False
        
        # Get existing vessel locations for exclusion zones
        vessel_locations = self.get_existing_vessel_locations()
        
        # Calculate patches per satellite (distribute evenly)
        patches_needed = self.stats['new_sea_patches_needed']
        patches_per_satellite = max(1, patches_needed // len(satellite_files))
        remaining_patches = patches_needed % len(satellite_files)
        
        print(f"Target: {patches_needed} new sea patches")
        print(f"Strategy: ~{patches_per_satellite} patches per satellite")
        
        self.new_sea_patches = []
        
        for i, satellite_file in enumerate(satellite_files):
            sat_name = self._extract_satellite_name(satellite_file)
            
            # Calculate patches for this satellite
            patches_for_this_sat = patches_per_satellite
            if i < remaining_patches:
                patches_for_this_sat += 1
            
            if patches_for_this_sat == 0:
                continue
            
            print(f"\n--- Processing {sat_name} (target: {patches_for_this_sat} patches) ---")
            
            # Load satellite data
            sar_data = self.load_satellite_data(satellite_file)
            if sar_data is None:
                print(f"⚠️  Skipping {sat_name} - failed to load data")
                continue
            
            # Create exclusion zones for this satellite
            exclusion_zones = []
            for vessel in vessel_locations:
                if vessel['satellite'] == sat_name:
                    # Create exclusion zone around vessel
                    exclusion_size = self.patch_size * 2  # 2x patch size exclusion
                    exclusion_zones.append({
                        'x_min': max(0, vessel['x'] - exclusion_size),
                        'x_max': min(sar_data.shape[1], vessel['x'] + exclusion_size),
                        'y_min': max(0, vessel['y'] - exclusion_size),
                        'y_max': min(sar_data.shape[0], vessel['y'] + exclusion_size)
                    })
            
            print(f"   Created {len(exclusion_zones)} exclusion zones")
            
            # Generate sea patches for this satellite
            patches_generated = self._generate_sea_patches_from_satellite(
                sar_data, sat_name, patches_for_this_sat, exclusion_zones
            )
            
            print(f"   Generated {patches_generated}/{patches_for_this_sat} patches from {sat_name}")
        
        total_generated = len(self.new_sea_patches)
        self.stats['new_sea_patches_generated'] = total_generated
        
        print(f"\n✅ Total new sea patches generated: {total_generated}/{patches_needed}")
        
        if total_generated < patches_needed:
            print(f"⚠️  Generated fewer patches than needed ({total_generated}/{patches_needed})")
            print("   This may be due to limited valid sea areas or too many exclusion zones")
        
        return total_generated > 0
    
    def _generate_sea_patches_from_satellite(self, sar_data, sat_name, target_patches, exclusion_zones):
        """Generate sea patches from a specific satellite"""
        height, width = sar_data.shape
        patches_generated = 0
        
        # Use stratified sampling for diversity
        # Divide image into grid regions and sample from each
        grid_size = 4  # 4x4 grid
        region_width = width // grid_size
        region_height = height // grid_size
        
        patches_per_region = max(1, target_patches // (grid_size * grid_size))
        
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                if patches_generated >= target_patches:
                    break
                
                # Define region bounds
                region_x_min = grid_x * region_width + self.half_patch
                region_x_max = min((grid_x + 1) * region_width, width) - self.half_patch
                region_y_min = grid_y * region_height + self.half_patch
                region_y_max = min((grid_y + 1) * region_height, height) - self.half_patch
                
                if region_x_max <= region_x_min or region_y_max <= region_y_min:
                    continue
                
                # Generate patches from this region
                region_patches = 0
                attempts = 0
                max_attempts = patches_per_region * 50
                
                while region_patches < patches_per_region and attempts < max_attempts and patches_generated < target_patches:
                    attempts += 1
                    
                    # Random location within region
                    center_x = random.randint(region_x_min, region_x_max - 1)
                    center_y = random.randint(region_y_min, region_y_max - 1)
                    
                    # Check exclusion zones
                    in_exclusion_zone = False
                    for zone in exclusion_zones:
                        if (zone['x_min'] <= center_x <= zone['x_max'] and
                            zone['y_min'] <= center_y <= zone['y_max']):
                            in_exclusion_zone = True
                            break
                    
                    if in_exclusion_zone:
                        continue
                    
                    # Extract and validate patch
                    patch = sar_data[center_y - self.half_patch:center_y + self.half_patch,
                                   center_x - self.half_patch:center_x + self.half_patch]
                    
                    if patch.shape == (self.patch_size, self.patch_size) and not np.all(np.isnan(patch)):
                        # Normalize patch
                        patch_normalized = self._normalize_patch(patch)
                        
                        if patch_normalized is not None:
                            patch_info = {
                                'patch': patch_normalized,
                                'center_x': center_x,
                                'center_y': center_y,
                                'satellite_source': sat_name,
                                'region': f"{grid_x}_{grid_y}"
                            }
                            
                            self.new_sea_patches.append(patch_info)
                            region_patches += 1
                            patches_generated += 1
        
        return patches_generated
    
    def _normalize_patch(self, patch):
        """Normalize patch to [0, 1] range using robust statistics"""
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
    
    def save_balanced_dataset(self):
        """Save the balanced dataset with new sea patches"""
        print("\n=== SAVING BALANCED DATASET ===")
        
        if not self.new_sea_patches:
            print("✅ No new patches to save")
            return True
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        
        # Start with existing labels
        updated_labels = self.existing_labels.copy()
        
        # Get current highest sea patch number
        existing_sea_numbers = []
        for label in self.existing_labels:
            if label.get('label') == 0 and label.get('image', '').startswith('sea_'):
                try:
                    num = int(label['image'].replace('sea_', '').replace('.png', ''))
                    existing_sea_numbers.append(num)
                except:
                    pass
        
        next_sea_number = max(existing_sea_numbers, default=0) + 1
        
        # Save new sea patches
        print(f"Adding {len(self.new_sea_patches)} new sea patches...")
        
        for i, sea_patch in enumerate(self.new_sea_patches):
            # Generate filename
            filename = f"sea_{next_sea_number + i}.png"
            filepath = os.path.join(self.output_dir, 'images', filename)
            
            # Convert to 8-bit image and save
            patch_8bit = (sea_patch['patch'] * 255).astype(np.uint8)
            Image.fromarray(patch_8bit, mode='L').save(filepath)
            
            # Create label entry
            label_entry = {
                "image": filename,
                "label": 0,  # Sea class
                "center_coordinates": [sea_patch['center_x'], sea_patch['center_y']],
                "satellite_source": sea_patch['satellite_source'],
                "region": sea_patch.get('region', 'unknown'),
                "added_timestamp": datetime.now().isoformat(),
                "generation_method": "balanced_dataset_generator"
            }
            
            updated_labels.append(label_entry)
            
            if (i + 1) % 20 == 0:
                print(f"   Saved {i + 1}/{len(self.new_sea_patches)} patches...")
        
        # Save updated labels file
        labels_file = os.path.join(self.output_dir, 'labels.json')
        with open(labels_file, 'w') as f:
            json.dump(updated_labels, f, indent=2)
        
        print(f"✅ Updated labels file: {labels_file}")
        
        # Update statistics
        final_ship_count = len([l for l in updated_labels if l.get('label') == 1])
        final_sea_count = len([l for l in updated_labels if l.get('label') == 0])
        
        self.stats['generation_time'] = datetime.now().isoformat()
        self.stats['final_ship_count'] = final_ship_count
        self.stats['final_sea_count'] = final_sea_count
        self.stats['final_total_count'] = len(updated_labels)
        self.stats['final_balance_ratio'] = final_sea_count / max(final_ship_count, 1)
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'balanced_dataset_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✅ Saved statistics: {stats_file}")
        
        return True
    
    def print_balance_summary(self):
        """Print dataset balance summary"""
        print(f"\n{'='*60}")
        print("BALANCED DATASET SUMMARY")
        print(f"{'='*60}")
        
        print(f"📊 BEFORE BALANCING:")
        print(f"   Ship patches: {self.stats['existing_ship_count']}")
        print(f"   Sea patches: {self.stats['existing_sea_count']}")
        print(f"   Ratio: {self.stats['existing_sea_count']/max(self.stats['existing_ship_count'],1):.2f}:1 (sea:ship)")
        
        print(f"\n📈 AFTER BALANCING:")
        print(f"   Ship patches: {self.stats.get('final_ship_count', self.stats['existing_ship_count'])}")
        print(f"   Sea patches: {self.stats.get('final_sea_count', self.stats['existing_sea_count'])}")
        print(f"   Total patches: {self.stats.get('final_total_count', 0)}")
        print(f"   Ratio: {self.stats.get('final_balance_ratio', 0):.2f}:1 (sea:ship)")
        
        print(f"\n🎯 GENERATION RESULTS:")
        print(f"   New sea patches needed: {self.stats['new_sea_patches_needed']}")
        print(f"   New sea patches generated: {self.stats['new_sea_patches_generated']}")
        print(f"   Success rate: {self.stats['new_sea_patches_generated']/max(self.stats['new_sea_patches_needed'],1)*100:.1f}%")
        
        print(f"\n📁 OUTPUT:")
        print(f"   Dataset directory: {self.output_dir}")
        print(f"   Images: {self.output_dir}/images/")
        print(f"   Labels: {self.output_dir}/labels.json")
        print(f"   Statistics: {self.output_dir}/balanced_dataset_stats.json")
        
        # Check if dataset is now balanced
        final_ratio = self.stats.get('final_balance_ratio', 0)
        if 0.8 <= final_ratio <= 1.2:
            print(f"\n✅ DATASET IS NOW BALANCED! (Ratio: {final_ratio:.2f}:1)")
        elif final_ratio > 1.2:
            print(f"\n⚠️  Dataset has more sea than ship patches (Ratio: {final_ratio:.2f}:1)")
        else:
            print(f"\n⚠️  Dataset still needs more sea patches (Ratio: {final_ratio:.2f}:1)")
    
    def generate_balanced_dataset(self):
        """Main function to generate balanced dataset"""
        print("BALANCED DATASET GENERATOR")
        print("=" * 60)
        
        if not DEPENDENCIES_AVAILABLE:
            print("❌ Missing required dependencies")
            return False
        
        # Execute balancing pipeline
        success = True
        success &= self.analyze_current_dataset()
        
        if success and self.stats['new_sea_patches_needed'] > 0:
            success &= self.generate_diverse_sea_patches()
            success &= self.save_balanced_dataset()
        
        if success:
            self.print_balance_summary()
            print(f"\n{'='*60}")
            print("DATASET BALANCING COMPLETE!")
            print("Dataset is ready for improved CNN training!")
        else:
            print("\nDataset balancing failed. Check error messages above.")
        
        return success

def main():
    """Main execution function"""
    generator = BalancedDatasetGenerator(
        patch_size=64,
        output_dir='cnn_dataset',
        target_samples_per_class=400
    )
    
    generator.generate_balanced_dataset()

if __name__ == "__main__":
    main()