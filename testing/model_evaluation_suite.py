"""
Model Evaluation Suite for CNN Confidence Improvement

This module provides comprehensive evaluation metrics for the SAR vessel detection CNN,
including precision, recall, F1-score, AUC, confusion matrix, and confidence score analysis.

Requirements addressed: 4.2, 4.3, 4.4
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sar_preprocessing import create_standard_preprocessor
from train_cnn import EnhancedSARCNN, SimpleSARCNN, SARShipDataset
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confusion_matrix: np.ndarray
    confidence_distribution: Dict[str, List[float]]
    classification_report: str
    
    # Additional metrics for vessel detection
    vessel_avg_confidence: float
    sea_avg_confidence: float
    vessel_high_confidence_ratio: float  # Ratio of vessels with confidence > 0.7
    sea_low_confidence_ratio: float      # Ratio of sea with confidence < 0.3


class ModelEvaluator:
    """Comprehensive model evaluation suite for SAR vessel detection CNN"""
    
    def __init__(self, model_path: str, img_size: int = 64, device: str = None):
        """
        Initialize the model evaluator
        
        Args:
            model_path: Path to the trained model file
            img_size: Input image size for the model
            device: Device to run evaluation on (cuda/cpu)
        """
        self.model_path = model_path
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocessor = create_standard_preprocessor(img_size=img_size)
        
    def load_model(self) -> bool:
        """
        Load the trained model with automatic architecture detection
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False
            
        try:
            # Try enhanced model first
            self.model = EnhancedSARCNN().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("✅ Loaded EnhancedSARCNN model")
        except Exception as e:
            print(f"⚠️ Failed to load EnhancedSARCNN, trying SimpleSARCNN: {e}")
            try:
                self.model = SimpleSARCNN().to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("✅ Loaded SimpleSARCNN model")
            except Exception as e2:
                print(f"❌ Error loading model: {e2}")
                return False
        
        self.model.eval()
        return True
    
    def create_test_dataloader(self, json_file: str, image_folder: str, 
                              batch_size: int = 32) -> DataLoader:
        """
        Create a test data loader from dataset
        
        Args:
            json_file: Path to labels JSON file
            image_folder: Path to images folder
            batch_size: Batch size for evaluation
            
        Returns:
            DataLoader: Test data loader
        """
        # Use inference transform (no augmentation)
        test_transform = self.preprocessor.get_inference_transform()
        
        dataset = SARShipDataset(json_file, image_folder, test_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return dataloader
    
    def evaluate_model(self, test_loader: DataLoader) -> ModelMetrics:
        """
        Perform comprehensive model evaluation
        
        Args:
            test_loader: DataLoader containing test data
            
        Returns:
            ModelMetrics: Comprehensive evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("🔍 Starting comprehensive model evaluation...")
        
        # Collect predictions and ground truth
        all_probs = []
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(images)
                probs = torch.sigmoid(logits).squeeze()
                preds = (probs > 0.5).float()
                
                # Collect results
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate core metrics
        accuracy = np.mean(all_preds == all_labels)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # AUC-ROC (only if we have both classes)
        try:
            auc_roc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_roc = 0.0  # Only one class present
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        class_report = classification_report(
            all_labels, all_preds, 
            target_names=['Sea', 'Vessel'], 
            zero_division=0
        )
        
        # Confidence score analysis
        vessel_confidences = all_probs[all_labels == 1]
        sea_confidences = all_probs[all_labels == 0]
        
        confidence_dist = {
            'vessel_confidences': vessel_confidences.tolist(),
            'sea_confidences': sea_confidences.tolist()
        }
        
        # Additional vessel-specific metrics
        vessel_avg_conf = np.mean(vessel_confidences) if len(vessel_confidences) > 0 else 0.0
        sea_avg_conf = np.mean(sea_confidences) if len(sea_confidences) > 0 else 0.0
        vessel_high_conf_ratio = np.mean(vessel_confidences > 0.7) if len(vessel_confidences) > 0 else 0.0
        sea_low_conf_ratio = np.mean(sea_confidences < 0.3) if len(sea_confidences) > 0 else 0.0
        
        # Create metrics object
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            confusion_matrix=cm,
            confidence_distribution=confidence_dist,
            classification_report=class_report,
            vessel_avg_confidence=vessel_avg_conf,
            sea_avg_confidence=sea_avg_conf,
            vessel_high_confidence_ratio=vessel_high_conf_ratio,
            sea_low_confidence_ratio=sea_low_conf_ratio
        )
        
        print("✅ Model evaluation completed")
        return metrics
    
    def generate_evaluation_report(self, metrics: ModelMetrics, 
                                 output_dir: str = "evaluation_results") -> str:
        """
        Generate comprehensive evaluation report with visualizations
        
        Args:
            metrics: ModelMetrics object from evaluate_model()
            output_dir: Directory to save report and visualizations
            
        Returns:
            str: Path to the generated report file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Generating evaluation report in {output_dir}...")
        
        # 1. Create confusion matrix plot
        self._plot_confusion_matrix(metrics.confusion_matrix, 
                                   os.path.join(output_dir, "confusion_matrix.png"))
        
        # 2. Create confidence distribution plots
        self._plot_confidence_distributions(metrics.confidence_distribution,
                                          os.path.join(output_dir, "confidence_distributions.png"))
        
        # 3. Generate text report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("CNN Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("CORE METRICS:\n")
            f.write(f"Accuracy:  {metrics.accuracy:.4f}\n")
            f.write(f"Precision: {metrics.precision:.4f}\n")
            f.write(f"Recall:    {metrics.recall:.4f}\n")
            f.write(f"F1-Score:  {metrics.f1_score:.4f}\n")
            f.write(f"AUC-ROC:   {metrics.auc_roc:.4f}\n\n")
            
            f.write("VESSEL-SPECIFIC METRICS:\n")
            f.write(f"Vessel Avg Confidence:     {metrics.vessel_avg_confidence:.4f}\n")
            f.write(f"Sea Avg Confidence:        {metrics.sea_avg_confidence:.4f}\n")
            f.write(f"Vessels >0.7 Confidence:   {metrics.vessel_high_confidence_ratio:.4f}\n")
            f.write(f"Sea <0.3 Confidence:       {metrics.sea_low_confidence_ratio:.4f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write(f"True Negatives (Sea):      {metrics.confusion_matrix[0,0]}\n")
            f.write(f"False Positives:           {metrics.confusion_matrix[0,1]}\n")
            f.write(f"False Negatives:           {metrics.confusion_matrix[1,0]}\n")
            f.write(f"True Positives (Vessels):  {metrics.confusion_matrix[1,1]}\n\n")
            
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write(metrics.classification_report)
        
        # 4. Save metrics as JSON
        metrics_dict = {
            'accuracy': float(metrics.accuracy),
            'precision': float(metrics.precision),
            'recall': float(metrics.recall),
            'f1_score': float(metrics.f1_score),
            'auc_roc': float(metrics.auc_roc),
            'vessel_avg_confidence': float(metrics.vessel_avg_confidence),
            'sea_avg_confidence': float(metrics.sea_avg_confidence),
            'vessel_high_confidence_ratio': float(metrics.vessel_high_confidence_ratio),
            'sea_low_confidence_ratio': float(metrics.sea_low_confidence_ratio),
            'confusion_matrix': metrics.confusion_matrix.tolist(),
            'confidence_distribution': {
                'vessel_confidences': [float(x) for x in metrics.confidence_distribution['vessel_confidences']],
                'sea_confidences': [float(x) for x in metrics.confidence_distribution['sea_confidences']]
            }
        }
        
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"✅ Evaluation report saved to {report_path}")
        return report_path
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sea', 'Vessel'],
                   yticklabels=['Sea', 'Vessel'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distributions(self, confidence_dist: Dict[str, List[float]], 
                                     save_path: str):
        """Plot and save confidence score distributions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Vessel confidence distribution
        vessel_confs = confidence_dist['vessel_confidences']
        if vessel_confs:
            ax1.hist(vessel_confs, bins=20, alpha=0.7, color='red', edgecolor='black')
            ax1.axvline(0.7, color='darkred', linestyle='--', 
                       label=f'Threshold (0.7)')
            ax1.set_title(f'Vessel Confidence Distribution (n={len(vessel_confs)})')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Sea confidence distribution
        sea_confs = confidence_dist['sea_confidences']
        if sea_confs:
            ax2.hist(sea_confs, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(0.3, color='darkblue', linestyle='--', 
                       label=f'Threshold (0.3)')
            ax2.set_title(f'Sea Confidence Distribution (n={len(sea_confs)})')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, metrics: ModelMetrics):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy:                  {metrics.accuracy:.3f}")
        print(f"Precision:                 {metrics.precision:.3f}")
        print(f"Recall:                    {metrics.recall:.3f}")
        print(f"F1-Score:                  {metrics.f1_score:.3f}")
        print(f"AUC-ROC:                   {metrics.auc_roc:.3f}")
        print(f"Vessel Avg Confidence:     {metrics.vessel_avg_confidence:.3f}")
        print(f"Sea Avg Confidence:        {metrics.sea_avg_confidence:.3f}")
        print(f"Vessels >0.7 Confidence:   {metrics.vessel_high_confidence_ratio:.3f}")
        print(f"Sea <0.3 Confidence:       {metrics.sea_low_confidence_ratio:.3f}")
        print("="*60)


def main():
    """Main evaluation function for testing"""
    # Configuration
    MODEL_PATH = "sar_cnn_model.pth"
    JSON_FILE = "cnn_dataset/labels.json"
    IMAGE_FOLDER = "cnn_dataset/images"
    
    # Initialize evaluator
    evaluator = ModelEvaluator(MODEL_PATH)
    
    # Load model
    if not evaluator.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Create test data loader
    test_loader = evaluator.create_test_dataloader(JSON_FILE, IMAGE_FOLDER)
    
    # Evaluate model
    metrics = evaluator.evaluate_model(test_loader)
    
    # Print summary
    evaluator.print_summary(metrics)
    
    # Generate report
    report_path = evaluator.generate_evaluation_report(metrics)
    print(f"\n📄 Full report available at: {report_path}")


if __name__ == "__main__":
    main()