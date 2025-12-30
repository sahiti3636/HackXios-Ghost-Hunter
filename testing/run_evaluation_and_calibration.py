#!/usr/bin/env python3
"""
Complete Model Evaluation and Calibration Workflow

This script demonstrates the complete workflow for evaluating and calibrating
the CNN model for vessel detection confidence scores.

Usage: python run_evaluation_and_calibration.py
"""

import os
import sys
from model_evaluation_suite import ModelEvaluator
from confidence_calibration import ConfidenceCalibrator


def main():
    """Run complete evaluation and calibration workflow"""
    print("🚀 Starting Complete Model Evaluation and Calibration Workflow")
    print("=" * 70)
    
    # Configuration
    MODEL_PATH = "sar_cnn_model.pth"
    JSON_FILE = "cnn_dataset/labels.json"
    IMAGE_FOLDER = "cnn_dataset/images"
    
    # Check if required files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Please train a model first using train_cnn.py")
        return False
    
    if not os.path.exists(JSON_FILE):
        print(f"❌ Dataset labels not found: {JSON_FILE}")
        print("   Please ensure the dataset is properly set up")
        return False
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Image folder not found: {IMAGE_FOLDER}")
        print("   Please ensure the dataset images are available")
        return False
    
    # Step 1: Initialize Model Evaluator
    print("\n📊 Step 1: Initializing Model Evaluator")
    evaluator = ModelEvaluator(MODEL_PATH)
    
    if not evaluator.load_model():
        print("❌ Failed to load model. Exiting.")
        return False
    
    # Step 2: Create Test Data Loader
    print("\n📊 Step 2: Loading Test Dataset")
    test_loader = evaluator.create_test_dataloader(JSON_FILE, IMAGE_FOLDER)
    print(f"✅ Loaded dataset with {len(test_loader.dataset)} samples")
    
    # Step 3: Comprehensive Model Evaluation
    print("\n📊 Step 3: Running Comprehensive Model Evaluation")
    metrics = evaluator.evaluate_model(test_loader)
    
    # Print evaluation summary
    evaluator.print_summary(metrics)
    
    # Generate detailed evaluation report
    report_path = evaluator.generate_evaluation_report(metrics, "evaluation_results")
    print(f"\n📄 Detailed evaluation report saved to: {report_path}")
    
    # Step 4: Confidence Score Calibration
    print("\n🎯 Step 4: Confidence Score Calibration")
    calibrator = ConfidenceCalibrator(evaluator)
    
    # Calibrate using validation data (same as test for this demo)
    cal_metrics = calibrator.calibrate_model(test_loader, method='auto')
    
    # Print calibration summary
    calibrator.print_calibration_summary(cal_metrics)
    
    # Generate reliability diagram
    calibrator.plot_reliability_diagram(cal_metrics, "calibration_reliability_diagram.png")
    
    # Save calibrator for future use
    calibrator.save_calibrator("confidence_calibrator.pkl")
    
    # Step 5: Demonstrate Calibrated Inference
    print("\n🧪 Step 5: Testing Calibrated Confidence Scores")
    test_scores = [0.2, 0.4, 0.6, 0.8, 0.95]
    
    print("Raw Score → Calibrated Score")
    print("-" * 30)
    for raw_score in test_scores:
        calibrated_score = calibrator.calibrate_confidence(raw_score)
        print(f"   {raw_score:.2f}   →      {calibrated_score:.3f}")
    
    # Step 6: Summary and Recommendations
    print("\n📋 Step 6: Summary and Recommendations")
    print("=" * 50)
    
    # Model performance assessment
    if metrics.accuracy > 0.95:
        print("✅ Model Performance: EXCELLENT (>95% accuracy)")
    elif metrics.accuracy > 0.90:
        print("✅ Model Performance: GOOD (>90% accuracy)")
    elif metrics.accuracy > 0.85:
        print("⚠️  Model Performance: ACCEPTABLE (>85% accuracy)")
    else:
        print("❌ Model Performance: NEEDS IMPROVEMENT (<85% accuracy)")
    
    # Confidence score assessment
    if metrics.vessel_high_confidence_ratio > 0.8:
        print("✅ Vessel Confidence: EXCELLENT (>80% vessels have >0.7 confidence)")
    elif metrics.vessel_high_confidence_ratio > 0.6:
        print("✅ Vessel Confidence: GOOD (>60% vessels have >0.7 confidence)")
    else:
        print("⚠️  Vessel Confidence: NEEDS IMPROVEMENT (<60% vessels have >0.7 confidence)")
    
    # Calibration improvement assessment
    brier_improvement = cal_metrics.brier_score_before - cal_metrics.brier_score_after
    if brier_improvement > 0.01:
        print("✅ Calibration: SIGNIFICANT IMPROVEMENT (Brier score reduced by >0.01)")
    elif brier_improvement > 0.001:
        print("✅ Calibration: MODERATE IMPROVEMENT (Brier score reduced)")
    else:
        print("⚠️  Calibration: MINIMAL IMPROVEMENT (Consider more calibration data)")
    
    print("\n🎉 Evaluation and Calibration Workflow Completed Successfully!")
    print("\nGenerated Files:")
    print("  - evaluation_results/evaluation_report.txt")
    print("  - evaluation_results/confusion_matrix.png")
    print("  - evaluation_results/confidence_distributions.png")
    print("  - evaluation_results/metrics.json")
    print("  - calibration_reliability_diagram.png")
    print("  - confidence_calibrator.pkl")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)