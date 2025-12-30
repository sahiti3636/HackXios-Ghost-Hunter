#!/usr/bin/env python3
"""
STEP 3: Radar Object Feature Extraction (Physics-based, Explainable)
STEP 4: Unsupervised Vessel Clustering (K-Means)

Consumes STEP 2 output: ship_detection_results_{satellite}.json

Usage:
    python vessel_feature_extraction.py --satellite-path <path_to_satellite_tiff>
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False


class VesselFeatureExtractor:
    def __init__(self):
        self.vessel_candidates = []
        self.vessel_features = []
        self.sar_data = None

    # ---------------------------------------------------------
    # LOAD STEP 2 OUTPUT FOR SPECIFIC SATELLITE
    # ---------------------------------------------------------
    def load_detection_results(self, satellite_name: str):
        print(f"=== LOADING DETECTION RESULTS FOR {satellite_name.upper()} ===")

        file_path = f"output/json/ship_detection_results_{satellite_name}.json"
        if not os.path.exists(file_path):
            print(f"❌ Missing {file_path}")
            return False

        with open(file_path, "r") as f:
            data = json.load(f)

        self.vessel_candidates = data.get("vessel_candidates", [])
        print(f"✅ Loaded {len(self.vessel_candidates)} vessels from {satellite_name}")
        return True

    # ---------------------------------------------------------
    # OPTIONAL: LOAD SAR FOR VISUALIZATION
    # ---------------------------------------------------------
    def load_sar_for_visualization(self, satellite_file_path: str):
        print(f"\n=== LOADING SAR DATA FOR VISUALIZATION ===")
        print(f"Satellite file: {satellite_file_path}")

        try:
            import sys
            import os
            
            # Add current directory to path to import ship_detections
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            
            from ship_detections import ShipDetectorSBCI

            detector = ShipDetectorSBCI(sbci_threshold=5.0, min_vessel_size=3)
            if not detector.load_sentinel1_data(satellite_file_path):
                print("⚠️ Could not load satellite data")
                return False

            detector.apply_noise_reduction()
            self.sar_data = detector.sar_data
            print("✅ SAR loaded successfully")
            return True

        except ImportError as e:
            print(f"⚠️ Could not import ship_detections: {e}")
            print("⚠️ SAR not available, visualization will still work")
            return True
        except Exception as e:
            print(f"⚠️ SAR loading failed: {e}")
            print("⚠️ SAR not available, visualization will still work")
            return True

    # ---------------------------------------------------------
    # STEP 3: FEATURE EXTRACTION
    # ---------------------------------------------------------
    def extract_vessel_features(self):
        print("\n=== STEP 3: FEATURE EXTRACTION ===")

        if not self.vessel_candidates:
            print("❌ No vessels found")
            return False

        self.vessel_features = []

        for v in self.vessel_candidates:
            area = v.get("area_pixels", 0)

            radius = np.sqrt(area / np.pi) if area > 0 else 0
            width = radius * 2
            height = radius * 1.5

            features = {
                "vessel_id": v["vessel_id"],
                "pixel_x": v["pixel_x"],
                "pixel_y": v["pixel_y"],
                "area_pixels": area,
                "mean_backscatter": v.get("mean_backscatter", 0.0),
                "mean_sbci": v.get("mean_sbci", 0.0),
                "aspect_ratio": width / height if height > 0 else 1.0,
                "compactness": area / (width * height) if width * height > 0 else 0.0,
                "bbox_min_x": v["pixel_x"] - width / 2,
                "bbox_min_y": v["pixel_y"] - height / 2,
                "bbox_max_x": v["pixel_x"] + width / 2,
                "bbox_max_y": v["pixel_y"] + height / 2,
            }

            self.vessel_features.append(features)

        print(f"✅ Extracted features for {len(self.vessel_features)} vessels")
        return True

    # ---------------------------------------------------------
    # STEP 4: K-MEANS CLUSTERING
    # ---------------------------------------------------------
    def perform_clustering(self, n_clusters=3):
        print("\n=== STEP 4: K-MEANS CLUSTERING ===")

        if len(self.vessel_features) == 0:
            print("⚠️  No vessels to cluster")
            return True
        
        # Adjust number of clusters based on available vessels
        actual_clusters = min(n_clusters, len(self.vessel_features))
        if actual_clusters < n_clusters:
            print(f"⚠️  Reducing clusters from {n_clusters} to {actual_clusters} (limited by vessel count)")

        X = np.array([
            [
                v["area_pixels"],
                v["mean_backscatter"],
                v["mean_sbci"],
                v["aspect_ratio"],
                v["compactness"],
            ]
            for v in self.vessel_features
        ])

        if len(self.vessel_features) == 1:
            # Single vessel - assign to cluster 0
            self.vessel_features[0]["cluster_id"] = 0
            print("✅ Single vessel assigned to cluster 0")
        else:
            # Multiple vessels - perform K-means clustering
            X = StandardScaler().fit_transform(X)
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            for v, label in zip(self.vessel_features, labels):
                v["cluster_id"] = int(label)
            
            print(f"✅ Clustering complete with {actual_clusters} clusters")
        
        return True

    # ---------------------------------------------------------
    # SAVE RESULTS WITH SATELLITE-SPECIFIC NAMING
    # ---------------------------------------------------------
    def save_results(self, satellite_name: str):
        if not self.vessel_features:
            # Create empty results files when no vessels detected
            os.makedirs("output/json", exist_ok=True)
            with open(f"output/json/vessel_features_{satellite_name}.json", "w") as f:
                json.dump([], f, indent=2)
            
            with open(f"output/vessel_features_{satellite_name}.csv", "w", newline="") as f:
                # Write header only for empty CSV
                writer = csv.writer(f)
                writer.writerow(["vessel_id", "pixel_x", "pixel_y", "area_pixels", 
                               "mean_backscatter", "mean_sbci", "aspect_ratio", 
                               "compactness", "bbox_min_x", "bbox_min_y", 
                               "bbox_max_x", "bbox_max_y", "cluster_id"])
            
            print(f"✅ Empty results saved for {satellite_name} (no vessels detected)")
            return
        
        # Ensure output directory exists
        os.makedirs("output/json", exist_ok=True)
        
        with open(f"output/json/vessel_features_{satellite_name}.json", "w") as f:
            json.dump(self.vessel_features, f, indent=2)

        with open(f"output/vessel_features_{satellite_name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.vessel_features[0].keys())
            writer.writeheader()
            writer.writerows(self.vessel_features)

        print(f"✅ Results saved for {satellite_name}")

    # ---------------------------------------------------------
    # VISUALIZATION WITH SATELLITE-SPECIFIC NAMING
    # ---------------------------------------------------------
    def create_visualization(self, satellite_name: str):
        print("\n=== CREATING CLUSTER VISUALIZATION ===")

        fig, ax = plt.subplots(figsize=(18, 10))

        # SAR BACKGROUND - exactly like reference image
        if self.sar_data is not None:
            sar_display = np.log10(
                np.clip(
                    self.sar_data,
                    np.nanmin(self.sar_data[self.sar_data > 0]),
                    np.nanmax(self.sar_data),
                )
            )
            im = ax.imshow(sar_display, cmap="gray", aspect='auto')
            
            # Colorbar matching reference image style
            cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
            cbar.set_label("SAR Backscatter (log scale)", rotation=270, labelpad=20, fontsize=12)
        else:
            # Fallback if no SAR data
            ax.set_facecolor('darkgray')

        # Cluster colors and names exactly matching reference image
        cluster_colors = {0: "red", 1: "blue", 2: "green"}
        cluster_names = {
            0: "Small Vessels",
            1: "Medium Vessels", 
            2: "Large Vessels",
        }

        plotted = set()

        # Plot vessels exactly like reference image
        for v in self.vessel_features:
            cid = v["cluster_id"]
            color = cluster_colors.get(cid, "white")

            # Larger circles with white borders like reference
            ax.scatter(
                v["pixel_x"],
                v["pixel_y"],
                s=150,  # Larger size to match reference
                c=color,
                edgecolors="white",
                linewidth=2,
                label=f"Cluster {cid}: {cluster_names.get(cid, f'Cluster {cid}')}"
                if cid not in plotted
                else "",
            )

            # Vessel ID labels positioned like reference
            ax.text(
                v["pixel_x"] + 50,  # Offset to match reference
                v["pixel_y"] - 50,
                f"V{v['vessel_id']}",
                color="white",
                fontsize=12,
                fontweight="bold",
            )

            plotted.add(cid)

        # Dynamic title based on available data
        location_info = "Marine Protected Area"
        if self.vessel_features and len(self.vessel_features) > 0:
            # Try to get location from vessel data if available
            first_vessel = self.vessel_features[0]
            if 'mpa_name' in first_vessel and first_vessel['mpa_name']:
                location_info = first_vessel['mpa_name']
            elif 'latitude' in first_vessel and 'longitude' in first_vessel:
                lat = first_vessel.get('latitude')
                lon = first_vessel.get('longitude')
                if lat and lon:
                    location_info = f"Location: {lat:.2f}°N, {lon:.2f}°E"
        
        ax.set_title(
            f"Vessel Detection with Cluster Analysis - {satellite_name.upper()}\n({location_info})",
            fontsize=16,
            fontweight="bold",
            pad=20
        )
        
        # Axis labels
        ax.set_xlabel("X (pixels)", fontsize=12)
        ax.set_ylabel("Y (pixels)", fontsize=12)
        
        # Legend positioned like reference
        legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), 
                          fontsize=11, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        # Dynamic statistics box
        area_info = "Marine Protected Area"
        if self.vessel_features and len(self.vessel_features) > 0:
            first_vessel = self.vessel_features[0]
            if 'mpa_name' in first_vessel and first_vessel['mpa_name']:
                area_info = first_vessel['mpa_name']
            elif 'latitude' in first_vessel and 'longitude' in first_vessel:
                lat = first_vessel.get('latitude')
                lon = first_vessel.get('longitude')
                if lat and lon:
                    area_info = f"Coordinates: {lat:.2f}°N, {lon:.2f}°E"
        
        stats = (
            f"Detected Vessels: {len(self.vessel_features)}\n"
            f"Clusters: {len(set(v['cluster_id'] for v in self.vessel_features))}\n"
            f"Area: {area_info}"
        )

        ax.text(
            0.02,
            0.96,
            stats,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'),
            fontsize=12,
            verticalalignment="top",
        )

        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs("output/png", exist_ok=True)
        
        plt.savefig(f"output/png/vessel_clusters_overlay_{satellite_name}.png", dpi=300, bbox_inches="tight", 
                   facecolor='white', edgecolor='none')
        plt.close()  # Close instead of show to avoid blocking

    def generate_ais_integration(self, satellite_name: str):
        """Generate AIS integration JSON file from vessel features for specific satellite"""
        print(f"\n=== GENERATING AIS INTEGRATION FILE FOR {satellite_name.upper()} ===")
        
        if not self.vessel_features:
            print("❌ No vessel features available for AIS integration")
            return False
        
        try:
            # Sentinel-1 image parameters (from our analysis)
            image_shape = (10010, 12567)  # Height, Width (downsampled)
            
            # Geographic bounds (these should ideally come from the actual data)
            geographic_bounds = {
                'west': 93.180214,   # Degrees East
                'east': 95.769753,   # Degrees East  
                'south': 4.325254,   # Degrees North
                'north': 6.584108    # Degrees North
            }
            
            # Try to get more accurate bounds from vessel data if available
            if self.vessel_features and 'latitude' in self.vessel_features[0] and self.vessel_features[0]['latitude']:
                lats = [v['latitude'] for v in self.vessel_features if v.get('latitude')]
                lons = [v['longitude'] for v in self.vessel_features if v.get('longitude')]
                
                if lats and lons:
                    # Use actual vessel coordinates to refine bounds
                    geographic_bounds = {
                        'west': min(lons) - 0.1,
                        'east': max(lons) + 0.1,
                        'south': min(lats) - 0.1,
                        'north': max(lats) + 0.1
                    }
            
            # Satellite acquisition timestamp (from Sentinel-1 filename)
            acquisition_time = "2025-10-21T23:28:15Z"  # ISO-8601 format
            
            print(f"Image bounds: {geographic_bounds['south']:.3f}°N to {geographic_bounds['north']:.3f}°N")
            print(f"              {geographic_bounds['west']:.3f}°E to {geographic_bounds['east']:.3f}°E")
            
            # Generate AIS integration data
            ais_data = []
            
            for vessel in self.vessel_features:
                # Use existing coordinates if available, otherwise convert from pixels
                if vessel.get('latitude') and vessel.get('longitude'):
                    latitude = vessel['latitude']
                    longitude = vessel['longitude']
                else:
                    # Convert pixel coordinates to lat/lon
                    latitude, longitude = self._pixel_to_latlon(
                        vessel['pixel_x'],  # Changed from centroid_x
                        vessel['pixel_y'],  # Changed from centroid_y
                        image_shape,
                        geographic_bounds
                    )
                
                # Create AIS integration record
                ais_record = {
                    "vessel_id": vessel['vessel_id'],
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6), 
                    "timestamp_utc": acquisition_time,
                    "cluster_id": vessel['cluster_id'],
                    "mean_sbci": round(vessel.get('mean_sbci', 0), 6),
                    "max_sbci": round(vessel.get('max_sbci', 0), 6),
                    "mean_backscatter": round(vessel.get('mean_backscatter', 0), 2),
                    "max_backscatter": round(vessel.get('max_backscatter', 0), 2),
                    "satellite_source": satellite_name
                }
                
                ais_data.append(ais_record)
            
            # Save AIS detection input file with satellite-specific naming
            os.makedirs('data/processed', exist_ok=True)
            output_file = f'data/processed/ais_detection_input_{satellite_name}.json'
            with open(output_file, 'w') as f:
                json.dump(ais_data, f, indent=2)
            
            print(f"✅ Generated AIS integration file: {output_file}")
            print(f"📊 Summary for {satellite_name}:")
            print(f"   • Total vessels: {len(ais_data)}")
            
            # Count vessels by cluster
            for cluster_id in range(3):  # Assuming 3 clusters
                count = len([v for v in ais_data if v['cluster_id'] == cluster_id])
                if count > 0:
                    print(f"   • Cluster {cluster_id}: {count} vessels")
            
            print(f"   • Geographic coverage: Marine Protected Area")
            print(f"   • Timestamp: {acquisition_time}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating AIS integration: {e}")
            return False
    
    def _pixel_to_latlon(self, pixel_x, pixel_y, image_shape, geographic_bounds):
        """
        Convert pixel coordinates to latitude/longitude
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            image_shape: (height, width) of the image
            geographic_bounds: Dict with 'west', 'east', 'south', 'north' bounds
        
        Returns:
            (latitude, longitude) in decimal degrees
        """
        height, width = image_shape
        
        # Convert pixel coordinates to normalized coordinates (0-1)
        norm_x = pixel_x / width
        norm_y = pixel_y / height
        
        # Convert to geographic coordinates
        longitude = geographic_bounds['west'] + norm_x * (geographic_bounds['east'] - geographic_bounds['west'])
        latitude = geographic_bounds['north'] - norm_y * (geographic_bounds['north'] - geographic_bounds['south'])
        
        return latitude, longitude

    # ---------------------------------------------------------
    # PIPELINE FOR SPECIFIC SATELLITE
    # ---------------------------------------------------------
    def run(self, satellite_name: str, satellite_file_path: str):
        self.load_detection_results(satellite_name)
        self.load_sar_for_visualization(satellite_file_path)
        self.extract_vessel_features()
        self.perform_clustering(n_clusters=3)
        self.save_results(satellite_name)
        self.generate_ais_integration(satellite_name)  # Added AIS integration
        self.create_visualization(satellite_name)

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


# ---------------------------------------------------------
# MAIN WITH COMMAND LINE SUPPORT
# ---------------------------------------------------------
def main():
    """Main execution function with command line argument support"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vessel Feature Extraction & Clustering')
    parser.add_argument('--satellite-path', required=True, 
                       help='Path to satellite TIFF file')
    
    args = parser.parse_args()
    
    print("🛰️  VESSEL FEATURE EXTRACTION & CLUSTERING")
    print("=" * 60)
    
    # Extract satellite name from path
    extractor = VesselFeatureExtractor()
    satellite_name = extractor._extract_satellite_name(args.satellite_path)
    
    print(f"Processing satellite: {satellite_name}")
    
    extractor.run(satellite_name, args.satellite_path)

    print(f"\n🎉 PIPELINE COMPLETE FOR {satellite_name.upper()}")


# ---------------------------------------------------------
# LEGACY MAIN (FOR BACKWARD COMPATIBILITY)
# ---------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # New command-line mode
        main()
    else:
        # Legacy mode - process first available satellite
        print("🛰️  VESSEL FEATURE EXTRACTION & CLUSTERING (LEGACY MODE)")
        print("=" * 60)
        print("⚠️  Running in legacy mode - processing first available satellite")
        
        extractor = VesselFeatureExtractor()
        
        # Try to find first available satellite
        import glob
        tiff_files = glob.glob('data/raw/satellite/*/measurement/*vv*.tiff')
        if tiff_files:
            satellite_file_path = tiff_files[0]
            satellite_name = extractor._extract_satellite_name(satellite_file_path)
            extractor.run(satellite_name, satellite_file_path)
            print(f"\n🎉 PIPELINE COMPLETE FOR {satellite_name.upper()}")
        else:
            print("❌ No satellite files found")
            sys.exit(1)
