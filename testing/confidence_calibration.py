"""
Confidence Score Calibration for CNN Vessel Detection

This module implements confidence score calibration using Platt scaling and isotonic regression
to ensure that predicted confidence scores correlate with actual performance.

Requirements addressed: 5.5, 4.4
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass

from model_evaluation_suite import ModelEvaluator
from train_cnn import SARShipDataset


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality"""
    brier_score_before: float
    brier_score_after: float
    log_loss_before: float
    log_loss_after: float
    calibration_error_before: float
    calibration_error_after: float
    reliability_diagram_data: Dict


class ConfidenceCalibrator:
    """
    Confidence score calibration for SAR vessel detection CNN
    
    Implements both Platt scaling (sigmoid) and isotonic regression methods
    to calibrate model confidence scores.
    """
    
    def __init__(self, model_evaluator: ModelEvaluator):
        """
        Initialize calibrator with a model evaluator
        
        Args:
            model_evaluator: ModelEvaluator instance with loaded model
        """
        self.model_evaluator = model_evaluator
        self.platt_calibrator = None
        self.isotonic_calibrator = None
        self.calibration_method = None
        self.calibration_data = None
        
    def collect_calibration_data(self, val_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect raw confidence scores and true labels from validation set
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (confidence_scores, true_labels)
        """
        print("📊 Collecting calibration data from validation set...")
        
        if self.model_evaluator.model is None:
            raise ValueError("Model not loaded in evaluator")
        
        all_probs = []
        all_labels = []
        
        self.model_evaluator.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.model_evaluator.device)
                labels = labels.to(self.model_evaluator.device)
                
                # Get raw logits and convert to probabilities
                logits = self.model_evaluator.model(images)
                probs = torch.sigmoid(logits).squeeze()
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        confidence_scores = np.array(all_probs)
        true_labels = np.array(all_labels)
        
        print(f"✅ Collected {len(confidence_scores)} samples for calibration")
        return confidence_scores, true_labels
    
    def fit_platt_scaling(self, confidence_scores: np.ndarray, 
                         true_labels: np.ndarray) -> LogisticRegression:
        """
        Fit Platt scaling (sigmoid) calibration
        
        Args:
            confidence_scores: Raw model confidence scores
            true_labels: True binary labels
            
        Returns:
            Fitted LogisticRegression calibrator
        """
        print("🔧 Fitting Platt scaling calibration...")
        
        # Convert probabilities to logits for Platt scaling
        # Avoid log(0) and log(1) by clipping
        eps = 1e-15
        clipped_scores = np.clip(confidence_scores, eps, 1 - eps)
        logits = np.log(clipped_scores / (1 - clipped_scores))
        
        # Fit logistic regression on logits
        platt_calibrator = LogisticRegression()
        platt_calibrator.fit(logits.reshape(-1, 1), true_labels)
        
        print("✅ Platt scaling calibration fitted")
        return platt_calibrator
    
    def fit_isotonic_regression(self, confidence_scores: np.ndarray,
                               true_labels: np.ndarray) -> IsotonicRegression:
        """
        Fit isotonic regression calibration
        
        Args:
            confidence_scores: Raw model confidence scores
            true_labels: True binary labels
            
        Returns:
            Fitted IsotonicRegression calibrator
        """
        print("🔧 Fitting isotonic regression calibration...")
        
        isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        isotonic_calibrator.fit(confidence_scores, true_labels)
        
        print("✅ Isotonic regression calibration fitted")
        return isotonic_calibrator
    
    def calibrate_model(self, val_loader: DataLoader, 
                       method: str = 'auto') -> CalibrationMetrics:
        """
        Calibrate the model using validation data
        
        Args:
            val_loader: DataLoader for validation data
            method: Calibration method ('platt', 'isotonic', or 'auto')
            
        Returns:
            CalibrationMetrics with before/after comparison
        """
        print(f"🎯 Starting confidence calibration with method: {method}")
        
        # Collect calibration data
        confidence_scores, true_labels = self.collect_calibration_data(val_loader)
        
        # Split data for calibration and evaluation
        cal_scores, eval_scores, cal_labels, eval_labels = train_test_split(
            confidence_scores, true_labels, test_size=0.3, random_state=42, 
            stratify=true_labels
        )
        
        # Calculate metrics before calibration
        brier_before = brier_score_loss(eval_labels, eval_scores)
        log_loss_before = log_loss(eval_labels, eval_scores)
        cal_error_before = self._calculate_calibration_error(eval_scores, eval_labels)
        
        # Fit calibrators
        platt_cal = self.fit_platt_scaling(cal_scores, cal_labels)
        isotonic_cal = self.fit_isotonic_regression(cal_scores, cal_labels)
        
        # Apply calibrations to evaluation set
        platt_calibrated = self._apply_platt_calibration(eval_scores, platt_cal)
        isotonic_calibrated = self._apply_isotonic_calibration(eval_scores, isotonic_cal)
        
        # Calculate metrics after calibration for both methods
        platt_brier = brier_score_loss(eval_labels, platt_calibrated)
        isotonic_brier = brier_score_loss(eval_labels, isotonic_calibrated)
        
        platt_log_loss = log_loss(eval_labels, platt_calibrated)
        isotonic_log_loss = log_loss(eval_labels, isotonic_calibrated)
        
        platt_cal_error = self._calculate_calibration_error(platt_calibrated, eval_labels)
        isotonic_cal_error = self._calculate_calibration_error(isotonic_calibrated, eval_labels)
        
        # Choose best method based on Brier score (lower is better)
        if method == 'auto':
            if platt_brier < isotonic_brier:
                self.calibration_method = 'platt'
                self.platt_calibrator = platt_cal
                final_calibrated = platt_calibrated
                brier_after = platt_brier
                log_loss_after = platt_log_loss
                cal_error_after = platt_cal_error
            else:
                self.calibration_method = 'isotonic'
                self.isotonic_calibrator = isotonic_cal
                final_calibrated = isotonic_calibrated
                brier_after = isotonic_brier
                log_loss_after = isotonic_log_loss
                cal_error_after = isotonic_cal_error
        elif method == 'platt':
            self.calibration_method = 'platt'
            self.platt_calibrator = platt_cal
            final_calibrated = platt_calibrated
            brier_after = platt_brier
            log_loss_after = platt_log_loss
            cal_error_after = platt_cal_error
        elif method == 'isotonic':
            self.calibration_method = 'isotonic'
            self.isotonic_calibrator = isotonic_cal
            final_calibrated = isotonic_calibrated
            brier_after = isotonic_brier
            log_loss_after = isotonic_log_loss
            cal_error_after = isotonic_cal_error
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        print(f"✅ Selected calibration method: {self.calibration_method}")
        
        # Generate reliability diagram data
        reliability_data = self._generate_reliability_diagram_data(
            eval_scores, final_calibrated, eval_labels
        )
        
        # Store calibration data for future use
        self.calibration_data = {
            'method': self.calibration_method,
            'validation_scores': eval_scores,
            'validation_labels': eval_labels,
            'calibrated_scores': final_calibrated
        }
        
        return CalibrationMetrics(
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            log_loss_before=log_loss_before,
            log_loss_after=log_loss_after,
            calibration_error_before=cal_error_before,
            calibration_error_after=cal_error_after,
            reliability_diagram_data=reliability_data
        )
    
    def _apply_platt_calibration(self, scores: np.ndarray, 
                                calibrator: LogisticRegression) -> np.ndarray:
        """Apply Platt scaling to confidence scores"""
        eps = 1e-15
        clipped_scores = np.clip(scores, eps, 1 - eps)
        logits = np.log(clipped_scores / (1 - clipped_scores))
        return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    
    def _apply_isotonic_calibration(self, scores: np.ndarray,
                                   calibrator: IsotonicRegression) -> np.ndarray:
        """Apply isotonic regression to confidence scores"""
        return calibrator.predict(scores)
    
    def _calculate_calibration_error(self, probs: np.ndarray, 
                                   labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        
        Args:
            probs: Predicted probabilities
            labels: True binary labels
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _generate_reliability_diagram_data(self, uncalibrated: np.ndarray,
                                         calibrated: np.ndarray,
                                         labels: np.ndarray) -> Dict:
        """Generate data for reliability diagram plotting"""
        n_bins = 10
        
        # Calculate calibration curves
        fraction_pos_uncal, mean_pred_uncal = calibration_curve(
            labels, uncalibrated, n_bins=n_bins
        )
        fraction_pos_cal, mean_pred_cal = calibration_curve(
            labels, calibrated, n_bins=n_bins
        )
        
        return {
            'uncalibrated': {
                'fraction_positives': fraction_pos_uncal,
                'mean_predicted': mean_pred_uncal
            },
            'calibrated': {
                'fraction_positives': fraction_pos_cal,
                'mean_predicted': mean_pred_cal
            }
        }
    
    def calibrate_confidence(self, raw_score: float) -> float:
        """
        Calibrate a single confidence score
        
        Args:
            raw_score: Raw model confidence score
            
        Returns:
            Calibrated confidence score
        """
        if self.calibration_method is None:
            raise ValueError("Model not calibrated. Call calibrate_model() first.")
        
        # Ensure score is in valid range
        raw_score = np.clip(raw_score, 1e-15, 1 - 1e-15)
        
        if self.calibration_method == 'platt':
            logit = np.log(raw_score / (1 - raw_score))
            calibrated = self.platt_calibrator.predict_proba([[logit]])[0, 1]
        elif self.calibration_method == 'isotonic':
            calibrated = self.isotonic_calibrator.predict([raw_score])[0]
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def save_calibrator(self, filepath: str):
        """Save the fitted calibrator to disk"""
        if self.calibration_method is None:
            raise ValueError("No calibrator fitted")
        
        calibrator_data = {
            'method': self.calibration_method,
            'platt_calibrator': self.platt_calibrator,
            'isotonic_calibrator': self.isotonic_calibrator
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(calibrator_data, f)
        
        print(f"✅ Calibrator saved to {filepath}")
    
    def load_calibrator(self, filepath: str):
        """Load a fitted calibrator from disk"""
        with open(filepath, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        self.calibration_method = calibrator_data['method']
        self.platt_calibrator = calibrator_data['platt_calibrator']
        self.isotonic_calibrator = calibrator_data['isotonic_calibrator']
        
        print(f"✅ Calibrator loaded from {filepath}")
    
    def plot_reliability_diagram(self, metrics: CalibrationMetrics, 
                                save_path: str = None):
        """
        Plot reliability diagram showing calibration quality
        
        Args:
            metrics: CalibrationMetrics from calibrate_model()
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        
        # Uncalibrated curve
        uncal_data = metrics.reliability_diagram_data['uncalibrated']
        ax.plot(uncal_data['mean_predicted'], uncal_data['fraction_positives'],
                'ro-', label='Uncalibrated', markersize=8)
        
        # Calibrated curve
        cal_data = metrics.reliability_diagram_data['calibrated']
        ax.plot(cal_data['mean_predicted'], cal_data['fraction_positives'],
                'bo-', label=f'Calibrated ({self.calibration_method})', markersize=8)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram (Calibration Plot)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add metrics text
        textstr = f'''Brier Score: {metrics.brier_score_before:.4f} → {metrics.brier_score_after:.4f}
Log Loss: {metrics.log_loss_before:.4f} → {metrics.log_loss_after:.4f}
Calibration Error: {metrics.calibration_error_before:.4f} → {metrics.calibration_error_after:.4f}'''
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Reliability diagram saved to {save_path}")
        
        plt.tight_layout()
        if save_path:
            plt.close()
        else:
            plt.show()
    
    def print_calibration_summary(self, metrics: CalibrationMetrics):
        """Print summary of calibration results"""
        print("\n" + "="*60)
        print("🎯 CONFIDENCE CALIBRATION SUMMARY")
        print("="*60)
        print(f"Selected Method:           {self.calibration_method}")
        print(f"Brier Score:               {metrics.brier_score_before:.4f} → {metrics.brier_score_after:.4f}")
        print(f"Log Loss:                  {metrics.log_loss_before:.4f} → {metrics.log_loss_after:.4f}")
        print(f"Calibration Error (ECE):   {metrics.calibration_error_before:.4f} → {metrics.calibration_error_after:.4f}")
        
        # Improvement indicators
        brier_improvement = metrics.brier_score_before - metrics.brier_score_after
        log_loss_improvement = metrics.log_loss_before - metrics.log_loss_after
        ece_improvement = metrics.calibration_error_before - metrics.calibration_error_after
        
        print(f"\nIMPROVEMENTS:")
        print(f"Brier Score:               {brier_improvement:+.4f} {'✅' if brier_improvement > 0 else '❌'}")
        print(f"Log Loss:                  {log_loss_improvement:+.4f} {'✅' if log_loss_improvement > 0 else '❌'}")
        print(f"Calibration Error:         {ece_improvement:+.4f} {'✅' if ece_improvement > 0 else '❌'}")
        print("="*60)


def main():
    """Main function for testing calibration"""
    from model_evaluation_suite import ModelEvaluator
    
    # Configuration
    MODEL_PATH = "sar_cnn_model.pth"
    JSON_FILE = "cnn_dataset/labels.json"
    IMAGE_FOLDER = "cnn_dataset/images"
    
    # Initialize evaluator and calibrator
    evaluator = ModelEvaluator(MODEL_PATH)
    if not evaluator.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    calibrator = ConfidenceCalibrator(evaluator)
    
    # Create validation data loader
    val_loader = evaluator.create_test_dataloader(JSON_FILE, IMAGE_FOLDER)
    
    # Calibrate model
    metrics = calibrator.calibrate_model(val_loader, method='auto')
    
    # Print summary
    calibrator.print_calibration_summary(metrics)
    
    # Plot reliability diagram
    calibrator.plot_reliability_diagram(metrics, "calibration_reliability_diagram.png")
    
    # Save calibrator
    calibrator.save_calibrator("confidence_calibrator.pkl")
    
    # Test calibration on a few examples
    print("\n🧪 Testing calibration on sample scores:")
    test_scores = [0.3, 0.5, 0.7, 0.9]
    for score in test_scores:
        calibrated = calibrator.calibrate_confidence(score)
        print(f"Raw: {score:.2f} → Calibrated: {calibrated:.3f}")


if __name__ == "__main__":
    main()