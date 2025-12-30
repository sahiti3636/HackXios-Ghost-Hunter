#!/usr/bin/env python3
"""
Dataset Analysis Script for CNN Confidence Improvement

This script analyzes the current CNN training dataset to identify class distribution
problems and calculate the imbalance ratio. It provides detailed statistics and
recommendations for addressing the class imbalance issue.

Requirements addressed: 1.1, 1.3
"""

import json
import os
from collections import Counter
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyzer:
    """Analyzes CNN training dataset for class distribution and quality issues."""
    
    def __init__(self, labels_path: str = "cnn_dataset/labels.json", 
                 images_dir: str = "cnn_dataset/images"):
        """
        Initialize the dataset analyzer.
        
        Args:
            labels_path: Path to the labels.json file
            images_dir: Path to the images directory
        """
        self.labels_path = labels_path
        self.images_dir = images_dir
        self.labels_data = None
        self.analysis_results = {}
        
    def load_labels(self) -> bool:
        """
        Load the labels.json file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.labels_path, 'r') as f:
                self.labels_data = json.load(f)
            print(f"✓ Successfully loaded {len(self.labels_data)} samples from {self.labels_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Error: Labels file not found at {self.labels_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON in labels file: {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading labels: {e}")
            return False
    
    def analyze_class_distribution(self) -> Dict:
        """
        Analyze the class distribution in the dataset.
        
        Returns:
            Dict: Analysis results including counts and ratios
        """
        if not self.labels_data:
            print("✗ No labels data loaded. Call load_labels() first.")
            return {}
        
        # Count classes
        class_counts = Counter()
        ship_samples = []
        sea_samples = []
        
        for sample in self.labels_data:
            label = sample.get('label')
            if label == 1:  # Ship
                class_counts['ship'] += 1
                ship_samples.append(sample)
            elif label == 0:  # Sea
                class_counts['sea'] += 1
                sea_samples.append(sample)
            else:
                print(f"⚠ Warning: Unknown label {label} found in sample {sample.get('image', 'unknown')}")
        
        total_samples = sum(class_counts.values())
        
        # Calculate ratios and imbalance
        ship_count = class_counts['ship']
        sea_count = class_counts['sea']
        
        if sea_count > 0:
            imbalance_ratio = ship_count / sea_count
        else:
            imbalance_ratio = float('inf')
        
        # Determine severity
        if imbalance_ratio > 2.0:
            severity = "SEVERE"
        elif imbalance_ratio > 1.5:
            severity = "MODERATE"
        elif imbalance_ratio < 0.5:
            severity = "SEVERE (reversed)"
        elif imbalance_ratio < 0.67:
            severity = "MODERATE (reversed)"
        else:
            severity = "ACCEPTABLE"
        
        # Calculate required samples for balance
        target_count = max(ship_count, sea_count)
        ship_needed = max(0, target_count - ship_count)
        sea_needed = max(0, target_count - sea_count)
        
        results = {
            'total_samples': total_samples,
            'ship_count': ship_count,
            'sea_count': sea_count,
            'ship_percentage': (ship_count / total_samples * 100) if total_samples > 0 else 0,
            'sea_percentage': (sea_count / total_samples * 100) if total_samples > 0 else 0,
            'imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'target_count': target_count,
            'ship_needed': ship_needed,
            'sea_needed': sea_needed,
            'ship_samples': ship_samples,
            'sea_samples': sea_samples
        }
        
        self.analysis_results.update(results)
        return results
    
    def analyze_satellite_distribution(self) -> Dict:
        """
        Analyze the distribution of samples across different satellites.
        
        Returns:
            Dict: Satellite distribution analysis
        """
        if not self.labels_data:
            return {}
        
        satellite_counts = Counter()
        satellite_class_counts = {}
        
        for sample in self.labels_data:
            satellite = sample.get('satellite_source', 'unknown')
            label = sample.get('label')
            
            satellite_counts[satellite] += 1
            
            if satellite not in satellite_class_counts:
                satellite_class_counts[satellite] = {'ship': 0, 'sea': 0}
            
            if label == 1:
                satellite_class_counts[satellite]['ship'] += 1
            elif label == 0:
                satellite_class_counts[satellite]['sea'] += 1
        
        results = {
            'satellite_counts': dict(satellite_counts),
            'satellite_class_counts': satellite_class_counts,
            'num_satellites': len(satellite_counts)
        }
        
        self.analysis_results.update(results)
        return results
    
    def check_file_existence(self) -> Dict:
        """
        Check if all image files referenced in labels actually exist.
        
        Returns:
            Dict: File existence analysis
        """
        if not self.labels_data:
            return {}
        
        missing_files = []
        existing_files = []
        
        for sample in self.labels_data:
            image_name = sample.get('image')
            if image_name:
                image_path = os.path.join(self.images_dir, image_name)
                if os.path.exists(image_path):
                    existing_files.append(image_name)
                else:
                    missing_files.append(image_name)
        
        results = {
            'total_referenced': len(self.labels_data),
            'existing_files': len(existing_files),
            'missing_files': len(missing_files),
            'missing_file_list': missing_files,
            'file_existence_rate': (len(existing_files) / len(self.labels_data) * 100) if self.labels_data else 0
        }
        
        self.analysis_results.update(results)
        return results
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on the analysis results.
        
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if 'imbalance_ratio' in self.analysis_results:
            ratio = self.analysis_results['imbalance_ratio']
            severity = self.analysis_results['severity']
            
            if severity in ['SEVERE', 'MODERATE']:
                sea_needed = self.analysis_results.get('sea_needed', 0)
                ship_needed = self.analysis_results.get('ship_needed', 0)
                
                if sea_needed > 0:
                    recommendations.append(
                        f"Generate {sea_needed} additional sea patches to balance the dataset "
                        f"(current ratio: {ratio:.2f}:1 ship:sea)"
                    )
                
                if ship_needed > 0:
                    recommendations.append(
                        f"Generate {ship_needed} additional ship patches to balance the dataset "
                        f"(current ratio: {ratio:.2f}:1 ship:sea)"
                    )
                
                recommendations.append(
                    "Ensure diverse sampling from different ocean regions to avoid overfitting"
                )
        
        if 'missing_files' in self.analysis_results:
            missing_count = self.analysis_results['missing_files']
            if missing_count > 0:
                recommendations.append(
                    f"Fix {missing_count} missing image files before training"
                )
        
        if 'num_satellites' in self.analysis_results:
            num_sats = self.analysis_results['num_satellites']
            if num_sats < 3:
                recommendations.append(
                    "Consider adding samples from more satellite sources for better generalization"
                )
        
        return recommendations
    
    def print_analysis_report(self):
        """Print a comprehensive analysis report."""
        print("\n" + "="*80)
        print("CNN DATASET ANALYSIS REPORT")
        print("="*80)
        
        if not self.analysis_results:
            print("✗ No analysis results available. Run analyze_class_distribution() first.")
            return
        
        # Class Distribution
        print(f"\n📊 CLASS DISTRIBUTION:")
        print(f"   Total Samples: {self.analysis_results.get('total_samples', 0)}")
        print(f"   Ship Samples: {self.analysis_results.get('ship_count', 0)} "
              f"({self.analysis_results.get('ship_percentage', 0):.1f}%)")
        print(f"   Sea Samples:  {self.analysis_results.get('sea_count', 0)} "
              f"({self.analysis_results.get('sea_percentage', 0):.1f}%)")
        print(f"   Imbalance Ratio: {self.analysis_results.get('imbalance_ratio', 0):.2f}:1 (ship:sea)")
        print(f"   Severity: {self.analysis_results.get('severity', 'UNKNOWN')}")
        
        # File Existence
        if 'file_existence_rate' in self.analysis_results:
            print(f"\n📁 FILE EXISTENCE:")
            print(f"   Files Found: {self.analysis_results.get('existing_files', 0)}")
            print(f"   Missing Files: {self.analysis_results.get('missing_files', 0)}")
            print(f"   Existence Rate: {self.analysis_results.get('file_existence_rate', 0):.1f}%")
        
        # Satellite Distribution
        if 'satellite_counts' in self.analysis_results:
            print(f"\n🛰️  SATELLITE DISTRIBUTION:")
            sat_counts = self.analysis_results['satellite_counts']
            for satellite, count in sorted(sat_counts.items()):
                print(f"   {satellite}: {count} samples")
        
        # Recommendations
        recommendations = self.generate_recommendations()
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
    
    def save_analysis_to_json(self, output_path: str = "dataset_analysis_results.json"):
        """
        Save analysis results to a JSON file.
        
        Args:
            output_path: Path to save the analysis results
        """
        # Create a serializable version of results
        serializable_results = {}
        for key, value in self.analysis_results.items():
            if key in ['ship_samples', 'sea_samples']:
                # Don't save the full sample data, just counts
                continue
            serializable_results[key] = value
        
        serializable_results['recommendations'] = self.generate_recommendations()
        serializable_results['analysis_timestamp'] = str(pd.Timestamp.now()) if 'pd' in globals() else "unknown"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"✓ Analysis results saved to {output_path}")
        except Exception as e:
            print(f"✗ Error saving analysis results: {e}")
    
    def create_visualization(self, output_path: str = "dataset_distribution.png"):
        """
        Create visualization of the dataset distribution.
        
        Args:
            output_path: Path to save the visualization
        """
        try:
            # Set up the plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('CNN Dataset Analysis', fontsize=16, fontweight='bold')
            
            # Class distribution pie chart
            if 'ship_count' in self.analysis_results and 'sea_count' in self.analysis_results:
                labels = ['Ship', 'Sea']
                sizes = [self.analysis_results['ship_count'], self.analysis_results['sea_count']]
                colors = ['#ff9999', '#66b3ff']
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Class Distribution')
            
            # Class distribution bar chart
            if 'ship_count' in self.analysis_results and 'sea_count' in self.analysis_results:
                classes = ['Ship', 'Sea']
                counts = [self.analysis_results['ship_count'], self.analysis_results['sea_count']]
                
                bars = ax2.bar(classes, counts, color=['#ff9999', '#66b3ff'])
                ax2.set_title('Sample Counts by Class')
                ax2.set_ylabel('Number of Samples')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            str(count), ha='center', va='bottom')
            
            # Satellite distribution
            if 'satellite_counts' in self.analysis_results:
                sat_data = self.analysis_results['satellite_counts']
                satellites = list(sat_data.keys())
                sat_counts = list(sat_data.values())
                
                ax3.bar(range(len(satellites)), sat_counts, color='#99ff99')
                ax3.set_title('Samples by Satellite Source')
                ax3.set_xlabel('Satellite')
                ax3.set_ylabel('Number of Samples')
                ax3.set_xticks(range(len(satellites)))
                ax3.set_xticklabels(satellites, rotation=45, ha='right')
            
            # Imbalance severity indicator
            if 'imbalance_ratio' in self.analysis_results:
                ratio = self.analysis_results['imbalance_ratio']
                severity = self.analysis_results['severity']
                
                # Create a simple indicator
                ax4.text(0.5, 0.7, f'Imbalance Ratio\n{ratio:.2f}:1', 
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        transform=ax4.transAxes)
                ax4.text(0.5, 0.3, f'Severity: {severity}', 
                        ha='center', va='center', fontsize=12,
                        transform=ax4.transAxes)
                
                # Color code by severity
                if 'SEVERE' in severity:
                    color = 'red'
                elif 'MODERATE' in severity:
                    color = 'orange'
                else:
                    color = 'green'
                
                ax4.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, 
                                          facecolor=color, alpha=0.3, transform=ax4.transAxes))
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.set_title('Imbalance Assessment')
                ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {output_path}")
            
        except Exception as e:
            print(f"✗ Error creating visualization: {e}")


def main():
    """Main function to run the dataset analysis."""
    print("CNN Dataset Analysis Tool")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Load labels
    if not analyzer.load_labels():
        return
    
    # Run analyses
    print("\n🔍 Analyzing class distribution...")
    analyzer.analyze_class_distribution()
    
    print("🔍 Analyzing satellite distribution...")
    analyzer.analyze_satellite_distribution()
    
    print("🔍 Checking file existence...")
    analyzer.check_file_existence()
    
    # Generate report
    analyzer.print_analysis_report()
    
    # Save results
    analyzer.save_analysis_to_json()
    
    # Create visualization
    try:
        analyzer.create_visualization()
    except ImportError:
        print("⚠ Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"⚠ Could not create visualization: {e}")


if __name__ == "__main__":
    main()