#!/usr/bin/env python3
"""
Integration Test Results for CNN Confidence Improvement
Tests the updated model in the full Ghost Hunter pipeline and documents performance.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Import our modules
from ves_verification import verify_snapshot, load_model
from sar_preprocessing import create_standard_preprocessor

def analyze_pipeline_results():
    """Analyze the results from the full pipeline run"""
    print("=" * 60)
    print("INTEGRATION TEST RESULTS - CNN CONFIDENCE IMPROVEMENT")
    print("=" * 60)
    
    # Load the final report
    report_path = "final_ghost_hunter_report_sat1.json"
    if not os.path.exists(report_path):
        print(f"❌ Final report not found: {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    vessels = report.get('vessels', [])
    print(f"\n📊 PIPELINE RESULTS SUMMARY")
    print(f"Total vessels detected: {len(vessels)}")
    print(f"Pipeline version: {report.get('pipeline_version', 'Unknown')}")
    print(f"Processing timestamp: {report.get('timestamp', 'Unknown')}")
    
    # Analyze confidence scores
    confidence_scores = [v.get('cnn_confidence', 0) for v in vessels]
    if confidence_scores:
        print(f"\n🎯 CNN CONFIDENCE ANALYSIS")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")
        print(f"Max confidence: {np.max(confidence_scores):.4f}")
        print(f"Min confidence: {np.min(confidence_scores):.4f}")
        print(f"Vessels above 0.7 threshold: {sum(1 for c in confidence_scores if c > 0.7)}/{len(confidence_scores)}")
        print(f"Vessels above 0.5 threshold: {sum(1 for c in confidence_scores if c > 0.5)}/{len(confidence_scores)}")
        
        # Individual vessel analysis
        print(f"\n📋 INDIVIDUAL VESSEL RESULTS")
        for i, vessel in enumerate(vessels, 1):
            conf = vessel.get('cnn_confidence', 0)
            verified = vessel.get('is_vessel_verified', 'UNKNOWN')
            size = vessel.get('vessel_size_pixels', 0)
            sbci = vessel.get('max_sbci', 0)
            print(f"  Vessel {i}: Confidence={conf:.4f}, Verified={verified}, Size={size}px, SBCI={sbci:.2f}")
    
    return vessels

def test_training_data_performance():
    """Test the model on original training data for comparison"""
    print(f"\n🧪 TESTING ON TRAINING DATA")
    
    # Load model
    model = load_model()
    if model is None:
        print("❌ Could not load model")
        return
    
    # Test on some training images
    training_dir = "cnn_dataset/images"
    if not os.path.exists(training_dir):
        print(f"❌ Training directory not found: {training_dir}")
        return
    
    # Get some ship and sea samples
    ship_files = [f for f in os.listdir(training_dir) if f.startswith('ship_')][:5]
    sea_files = [f for f in os.listdir(training_dir) if f.startswith('sea_')][:5]
    
    print(f"Testing on {len(ship_files)} ship samples and {len(sea_files)} sea samples")
    
    ship_scores = []
    sea_scores = []
    
    for ship_file in ship_files:
        test_data = {'image': os.path.join(training_dir, ship_file)}
        result = verify_snapshot(test_data)
        ship_scores.append(result['cnn_confidence'])
        print(f"  Ship {ship_file}: {result['cnn_confidence']:.4f}")
    
    for sea_file in sea_files:
        test_data = {'image': os.path.join(training_dir, sea_file)}
        result = verify_snapshot(test_data)
        sea_scores.append(result['cnn_confidence'])
        print(f"  Sea {sea_file}: {result['cnn_confidence']:.4f}")
    
    if ship_scores and sea_scores:
        print(f"\n📈 TRAINING DATA PERFORMANCE")
        print(f"Ship samples - Avg: {np.mean(ship_scores):.4f}, Range: {np.min(ship_scores):.4f}-{np.max(ship_scores):.4f}")
        print(f"Sea samples - Avg: {np.mean(sea_scores):.4f}, Range: {np.min(sea_scores):.4f}-{np.max(sea_scores):.4f}")
    
    return ship_scores, sea_scores

def compare_preprocessing_methods():
    """Compare different preprocessing approaches on pipeline chips"""
    print(f"\n🔬 PREPROCESSING COMPARISON")
    
    chip_dir = "output/chips/sat1"
    if not os.path.exists(chip_dir):
        print(f"❌ Chip directory not found: {chip_dir}")
        return
    
    # Get first chip for testing
    chip_files = [f for f in os.listdir(chip_dir) if f.endswith('.png')]
    if not chip_files:
        print("❌ No chip files found")
        return
    
    test_chip = os.path.join(chip_dir, chip_files[0])
    print(f"Testing preprocessing on: {test_chip}")
    
    # Load model
    model = load_model()
    if model is None:
        print("❌ Could not load model")
        return
    
    # Test different preprocessing approaches
    preprocessor = create_standard_preprocessor(img_size=64)
    
    try:
        # Method 1: Standard preprocessor
        tensor1 = preprocessor.preprocess_image(test_chip, training=False)
        tensor1 = tensor1.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        with torch.no_grad():
            logits1 = model(tensor1)
            score1 = torch.sigmoid(logits1).item()
        
        print(f"  Standard preprocessing: {score1:.6f}")
        
        # Method 2: Simple PIL + normalize
        img = Image.open(test_chip).convert('L')
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor2 = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        tensor2 = tensor2.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        with torch.no_grad():
            logits2 = model(tensor2)
            score2 = torch.sigmoid(logits2).item()
        
        print(f"  Simple PIL preprocessing: {score2:.6f}")
        
        # Show image statistics
        print(f"  Image stats - Shape: {img_array.shape}, Range: {img_array.min():.3f}-{img_array.max():.3f}")
        
    except Exception as e:
        print(f"❌ Error in preprocessing comparison: {e}")

def generate_performance_report():
    """Generate a comprehensive performance report"""
    print(f"\n📄 GENERATING PERFORMANCE REPORT")
    
    # Collect all results
    pipeline_vessels = analyze_pipeline_results()
    training_results = test_training_data_performance()
    compare_preprocessing_methods()
    
    # Load model validation report if available
    validation_report = {}
    if os.path.exists("model_validation_report.json"):
        with open("model_validation_report.json", 'r') as f:
            validation_report = json.load(f)
    
    # Create comprehensive report
    report = {
        "test_timestamp": datetime.now().isoformat(),
        "test_type": "Integration Test - Full Pipeline",
        "model_path": "sar_cnn_model.pth",
        "pipeline_performance": {
            "total_vessels_detected": len(pipeline_vessels) if pipeline_vessels else 0,
            "average_confidence": np.mean([v.get('cnn_confidence', 0) for v in pipeline_vessels]) if pipeline_vessels else 0,
            "vessels_above_threshold_70": sum(1 for v in pipeline_vessels if v.get('cnn_confidence', 0) > 0.7) if pipeline_vessels else 0,
            "vessels_above_threshold_50": sum(1 for v in pipeline_vessels if v.get('cnn_confidence', 0) > 0.5) if pipeline_vessels else 0,
            "verification_success_rate": sum(1 for v in pipeline_vessels if v.get('is_vessel_verified') == 'YES') / len(pipeline_vessels) if pipeline_vessels else 0
        },
        "training_data_performance": {
            "ship_avg_confidence": np.mean(training_results[0]) if training_results and training_results[0] else 0,
            "sea_avg_confidence": np.mean(training_results[1]) if training_results and training_results[1] else 0
        },
        "model_validation_summary": validation_report.get("requirements_validation", {}),
        "issues_identified": [],
        "recommendations": []
    }
    
    # Add issues and recommendations based on results
    if report["pipeline_performance"]["average_confidence"] < 0.1:
        report["issues_identified"].append("Very low confidence scores on pipeline-generated chips")
        report["recommendations"].append("Investigate preprocessing consistency between training and inference")
    
    if report["pipeline_performance"]["vessels_above_threshold_70"] == 0:
        report["issues_identified"].append("No vessels meet the 0.7 confidence threshold")
        report["recommendations"].append("Review chip extraction and preprocessing pipeline")
    
    if report["training_data_performance"]["ship_avg_confidence"] > 0.9:
        report["issues_identified"].append("Large performance gap between training data and pipeline chips")
        report["recommendations"].append("Ensure training data represents real pipeline conditions")
    
    # Save report
    with open("integration_test_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Integration test report saved: integration_test_report.json")
    
    return report

def main():
    """Main integration test function"""
    print("Starting CNN Confidence Improvement Integration Test...")
    
    # Run comprehensive analysis
    report = generate_performance_report()
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    pipeline_perf = report["pipeline_performance"]
    print(f"Pipeline vessels detected: {pipeline_perf['total_vessels_detected']}")
    print(f"Average confidence: {pipeline_perf['average_confidence']:.4f}")
    print(f"Vessels above 0.7 threshold: {pipeline_perf['vessels_above_threshold_70']}")
    print(f"Verification success rate: {pipeline_perf['verification_success_rate']:.2%}")
    
    if report["issues_identified"]:
        print(f"\n⚠️  ISSUES IDENTIFIED:")
        for issue in report["issues_identified"]:
            print(f"  • {issue}")
    
    if report["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    
    print(f"\n✅ Integration test completed successfully!")

if __name__ == "__main__":
    main()