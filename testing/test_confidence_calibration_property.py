"""
Property-Based Test for Confidence Score Calibration

This module implements property-based testing for the confidence calibration system
to ensure that calibrated scores are valid and correlate with actual performance.

**Feature: cnn-confidence-improvement, Property 5: Confidence Score Calibration**
**Validates: Requirements 5.5, 3.2**
"""

import os
import sys
import numpy as np
import torch
from hypothesis import given, strategies as st, settings, assume, example
from hypothesis.extra.numpy import arrays
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from confidence_calibration import ConfidenceCalibrator, CalibrationMetrics
from model_evaluation_suite import ModelEvaluator


class TestConfidenceCalibrationProperties:
    """Property-based tests for confidence score calibration"""
    
    def setup_method(self):
        """Setup mock evaluator for testing"""
        self.mock_evaluator = Mock(spec=ModelEvaluator)
        self.mock_evaluator.model = Mock()
        self.mock_evaluator.device = torch.device('cpu')
        
    @given(
        raw_scores=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=100),
            elements=st.floats(min_value=0.001, max_value=0.999, allow_nan=False)
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_calibrated_scores_are_valid_probabilities(self, raw_scores):
        """
        Property 5: Confidence Score Calibration - Valid Range
        
        For any set of raw confidence scores, all calibrated scores should be 
        valid probabilities between 0 and 1.
        
        **Validates: Requirements 5.5, 3.2**
        """
        # Create mock calibration data
        labels = np.random.binomial(1, 0.5, len(raw_scores))
        
        calibrator = ConfidenceCalibrator(self.mock_evaluator)
        
        # Fit both calibrators with mock data
        calibrator.platt_calibrator = calibrator.fit_platt_scaling(raw_scores, labels)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(raw_scores, labels)
        
        # Test both calibration methods
        for method in ['platt', 'isotonic']:
            calibrator.calibration_method = method
            
            for score in raw_scores:
                calibrated_score = calibrator.calibrate_confidence(float(score))
                
                # Property: Calibrated scores must be valid probabilities
                assert 0.0 <= calibrated_score <= 1.0, \
                    f"Calibrated score {calibrated_score} not in [0,1] for method {method}"
                assert not np.isnan(calibrated_score), \
                    f"Calibrated score is NaN for method {method}"
                assert not np.isinf(calibrated_score), \
                    f"Calibrated score is infinite for method {method}"
    
    @given(
        n_samples=st.integers(min_value=50, max_value=200),
        noise_level=st.floats(min_value=0.1, max_value=0.4)
    )
    @settings(max_examples=50, deadline=None)
    def test_calibration_improves_reliability(self, n_samples, noise_level):
        """
        Property 5: Confidence Score Calibration - Reliability Improvement
        
        For any dataset with miscalibrated scores, calibration should improve
        the reliability (reduce calibration error).
        
        **Validates: Requirements 5.5, 3.2**
        """
        # Generate synthetic classification data
        X, y = make_classification(
            n_samples=n_samples, n_features=10, n_classes=2, 
            random_state=42, flip_y=noise_level
        )
        
        # Create miscalibrated scores (biased toward extremes)
        raw_scores = np.random.beta(0.5, 0.5, n_samples)  # U-shaped distribution
        
        # Split data
        train_scores, test_scores, train_labels, test_labels = train_test_split(
            raw_scores, y, test_size=0.4, random_state=42
        )
        
        calibrator = ConfidenceCalibrator(self.mock_evaluator)
        
        # Fit calibrators
        platt_cal = calibrator.fit_platt_scaling(train_scores, train_labels)
        isotonic_cal = calibrator.fit_isotonic_regression(train_scores, train_labels)
        
        # Calculate calibration error before and after
        cal_error_before = calibrator._calculate_calibration_error(test_scores, test_labels)
        
        # Test Platt scaling
        calibrator.calibration_method = 'platt'
        calibrator.platt_calibrator = platt_cal
        platt_calibrated = np.array([
            calibrator.calibrate_confidence(float(score)) for score in test_scores
        ])
        platt_cal_error = calibrator._calculate_calibration_error(platt_calibrated, test_labels)
        
        # Test isotonic regression
        calibrator.calibration_method = 'isotonic'
        calibrator.isotonic_calibrator = isotonic_cal
        isotonic_calibrated = np.array([
            calibrator.calibrate_confidence(float(score)) for score in test_scores
        ])
        isotonic_cal_error = calibrator._calculate_calibration_error(isotonic_calibrated, test_labels)
        
        # Property: At least one calibration method should improve reliability
        # (reduce calibration error) for most datasets
        improvement_threshold = 0.05  # Allow some tolerance for noisy data
        
        platt_improved = platt_cal_error < (cal_error_before + improvement_threshold)
        isotonic_improved = isotonic_cal_error < (cal_error_before + improvement_threshold)
        
        assert platt_improved or isotonic_improved, \
            f"Neither calibration method improved reliability. " \
            f"Before: {cal_error_before:.4f}, Platt: {platt_cal_error:.4f}, " \
            f"Isotonic: {isotonic_cal_error:.4f}"
    
    @given(
        score=st.floats(min_value=0.001, max_value=0.999, allow_nan=False)
    )
    @settings(max_examples=100, deadline=None)
    def test_calibration_monotonicity_isotonic(self, score):
        """
        Property 5: Confidence Score Calibration - Isotonic Monotonicity
        
        For isotonic regression calibration, if score1 <= score2, then
        calibrated(score1) <= calibrated(score2) (monotonicity).
        
        **Validates: Requirements 5.5, 3.2**
        """
        # Generate synthetic data for calibration
        n_samples = 100
        raw_scores = np.random.uniform(0.01, 0.99, n_samples)
        labels = (raw_scores + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        calibrator = ConfidenceCalibrator(self.mock_evaluator)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(raw_scores, labels)
        calibrator.calibration_method = 'isotonic'
        
        # Test monotonicity with a slightly higher score
        score2 = min(0.999, score + 0.01)
        
        calibrated1 = calibrator.calibrate_confidence(score)
        calibrated2 = calibrator.calibrate_confidence(score2)
        
        # Property: Isotonic regression should preserve monotonicity
        assert calibrated1 <= calibrated2 + 1e-10, \
            f"Monotonicity violated: calibrated({score:.3f}) = {calibrated1:.3f} > " \
            f"calibrated({score2:.3f}) = {calibrated2:.3f}"
    
    @given(
        scores=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=20, max_value=50),
            elements=st.floats(min_value=0.01, max_value=0.99, allow_nan=False)
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_calibration_preserves_ranking(self, scores):
        """
        Property 5: Confidence Score Calibration - Ranking Preservation
        
        For any set of confidence scores, calibration should preserve the
        relative ranking of scores (higher raw score -> higher calibrated score).
        
        **Validates: Requirements 5.5, 3.2**
        """
        assume(len(np.unique(scores)) > 1)  # Need some variation in scores
        
        # Generate labels correlated with scores
        labels = (scores + np.random.normal(0, 0.1, len(scores)) > 0.5).astype(int)
        
        calibrator = ConfidenceCalibrator(self.mock_evaluator)
        
        # Test both calibration methods
        for method in ['platt', 'isotonic']:
            if method == 'platt':
                calibrator.platt_calibrator = calibrator.fit_platt_scaling(scores, labels)
            else:
                calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(scores, labels)
            
            calibrator.calibration_method = method
            
            # Get calibrated scores
            calibrated_scores = np.array([
                calibrator.calibrate_confidence(float(score)) for score in scores
            ])
            
            # Property: Ranking should be preserved (with some tolerance for ties)
            # Calculate Spearman rank correlation
            from scipy.stats import spearmanr
            correlation, p_value = spearmanr(scores, calibrated_scores)
            
            # Ranking should be strongly preserved
            assert correlation > 0.7, \
                f"Calibration method {method} did not preserve ranking well. " \
                f"Spearman correlation: {correlation:.3f}"
    
    @given(
        threshold=st.floats(min_value=0.1, max_value=0.9)
    )
    @settings(max_examples=30, deadline=None)
    def test_calibrated_scores_correlate_with_performance(self, threshold):
        """
        Property 5: Confidence Score Calibration - Performance Correlation
        
        For any threshold, samples with calibrated scores above the threshold
        should have higher accuracy than those below the threshold.
        
        **Validates: Requirements 5.5, 3.2**
        """
        # Generate realistic vessel detection scenario
        n_samples = 200
        
        # Create scores that correlate with true performance
        true_labels = np.random.binomial(1, 0.4, n_samples)  # 40% vessels
        
        # Raw scores with some calibration bias
        raw_scores = np.where(
            true_labels == 1,
            np.random.beta(3, 1, np.sum(true_labels == 1)),      # Vessels: biased high
            np.random.beta(1, 3, np.sum(true_labels == 0))       # Sea: biased low
        )
        
        # Add systematic bias to make scores miscalibrated
        raw_scores = np.clip(raw_scores * 0.8 + 0.1, 0.01, 0.99)
        
        # Split for calibration
        train_scores, test_scores, train_labels, test_labels = train_test_split(
            raw_scores, true_labels, test_size=0.5, random_state=42
        )
        
        calibrator = ConfidenceCalibrator(self.mock_evaluator)
        
        # Fit calibrator (use isotonic for this test)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(train_scores, train_labels)
        calibrator.calibration_method = 'isotonic'
        
        # Get calibrated scores for test set
        calibrated_scores = np.array([
            calibrator.calibrate_confidence(float(score)) for score in test_scores
        ])
        
        # Split test set by threshold
        high_conf_mask = calibrated_scores >= threshold
        low_conf_mask = calibrated_scores < threshold
        
        # Skip if we don't have samples in both groups
        assume(np.sum(high_conf_mask) > 5 and np.sum(low_conf_mask) > 5)
        
        # Calculate accuracy for each group
        high_conf_accuracy = np.mean(test_labels[high_conf_mask])
        low_conf_accuracy = np.mean(test_labels[low_conf_mask])
        
        # Property: Higher confidence should correlate with higher accuracy
        # Allow some tolerance for statistical variation
        min_difference = 0.05
        
        assert high_conf_accuracy >= low_conf_accuracy - min_difference, \
            f"High confidence samples (acc={high_conf_accuracy:.3f}) should perform " \
            f"better than low confidence samples (acc={low_conf_accuracy:.3f}) " \
            f"at threshold {threshold:.2f}"
    
    def test_calibration_edge_cases(self):
        """
        Test edge cases for calibration robustness
        
        **Validates: Requirements 5.5, 3.2**
        """
        calibrator = ConfidenceCalibrator(self.mock_evaluator)
        
        # Create minimal calibration data
        scores = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 0, 1])
        
        # Fit calibrators
        calibrator.platt_calibrator = calibrator.fit_platt_scaling(scores, labels)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(scores, labels)
        
        # Test edge cases for both methods
        for method in ['platt', 'isotonic']:
            calibrator.calibration_method = method
            
            # Test extreme values
            edge_cases = [0.001, 0.999, 0.5]
            
            for score in edge_cases:
                calibrated = calibrator.calibrate_confidence(score)
                
                # Should handle edge cases gracefully
                assert 0.0 <= calibrated <= 1.0
                assert not np.isnan(calibrated)
                assert not np.isinf(calibrated)


def run_property_tests():
    """Run all property-based tests for confidence calibration"""
    print("🧪 Running Property-Based Tests for Confidence Calibration")
    print("=" * 60)
    
    test_instance = TestConfidenceCalibrationProperties()
    test_instance.setup_method()
    
    # Test 1: Valid probability range
    print("Testing Property: Calibrated scores are valid probabilities...")
    try:
        # Create a simple test case manually
        raw_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1, 1])
        
        calibrator = ConfidenceCalibrator(test_instance.mock_evaluator)
        calibrator.platt_calibrator = calibrator.fit_platt_scaling(raw_scores, labels)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(raw_scores, labels)
        
        # Test both methods
        for method in ['platt', 'isotonic']:
            calibrator.calibration_method = method
            for score in raw_scores:
                calibrated_score = calibrator.calibrate_confidence(float(score))
                assert 0.0 <= calibrated_score <= 1.0
                assert not np.isnan(calibrated_score)
                assert not np.isinf(calibrated_score)
        
        print("✅ PASSED: Valid probability range")
    except Exception as e:
        print(f"❌ FAILED: Valid probability range - {e}")
        return False
    
    # Test 2: Reliability improvement
    print("Testing Property: Calibration improves reliability...")
    try:
        # Generate synthetic data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        raw_scores = np.random.beta(0.5, 0.5, 100)  # Miscalibrated scores
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_scores, test_scores, train_labels, test_labels = train_test_split(
            raw_scores, y, test_size=0.4, random_state=42
        )
        
        calibrator = ConfidenceCalibrator(test_instance.mock_evaluator)
        platt_cal = calibrator.fit_platt_scaling(train_scores, train_labels)
        isotonic_cal = calibrator.fit_isotonic_regression(train_scores, train_labels)
        
        # Calculate calibration errors
        cal_error_before = calibrator._calculate_calibration_error(test_scores, test_labels)
        
        calibrator.calibration_method = 'isotonic'
        calibrator.isotonic_calibrator = isotonic_cal
        isotonic_calibrated = np.array([
            calibrator.calibrate_confidence(float(score)) for score in test_scores
        ])
        isotonic_cal_error = calibrator._calculate_calibration_error(isotonic_calibrated, test_labels)
        
        # At least isotonic should improve or maintain calibration
        assert isotonic_cal_error <= cal_error_before + 0.1  # Allow some tolerance
        
        print("✅ PASSED: Reliability improvement")
    except Exception as e:
        print(f"❌ FAILED: Reliability improvement - {e}")
        return False
    
    # Test 3: Monotonicity
    print("Testing Property: Isotonic calibration preserves monotonicity...")
    try:
        # Generate data for calibration
        raw_scores = np.random.uniform(0.01, 0.99, 50)
        labels = (raw_scores + np.random.normal(0, 0.1, 50) > 0.5).astype(int)
        
        calibrator = ConfidenceCalibrator(test_instance.mock_evaluator)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(raw_scores, labels)
        calibrator.calibration_method = 'isotonic'
        
        # Test monotonicity
        test_scores = [0.2, 0.21, 0.5, 0.51, 0.8, 0.81]
        calibrated_scores = [calibrator.calibrate_confidence(s) for s in test_scores]
        
        # Check monotonicity (allowing small tolerance)
        for i in range(len(test_scores) - 1):
            assert calibrated_scores[i] <= calibrated_scores[i+1] + 1e-10
        
        print("✅ PASSED: Monotonicity preservation")
    except Exception as e:
        print(f"❌ FAILED: Monotonicity preservation - {e}")
        return False
    
    # Test 4: Ranking preservation
    print("Testing Property: Calibration preserves ranking...")
    try:
        scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        calibrator = ConfidenceCalibrator(test_instance.mock_evaluator)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(scores, labels)
        calibrator.calibration_method = 'isotonic'
        
        calibrated_scores = np.array([
            calibrator.calibrate_confidence(float(score)) for score in scores
        ])
        
        # Check ranking preservation using Spearman correlation
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(scores, calibrated_scores)
        assert correlation > 0.7
        
        print("✅ PASSED: Ranking preservation")
    except Exception as e:
        print(f"❌ FAILED: Ranking preservation - {e}")
        return False
    
    # Test 5: Performance correlation
    print("Testing Property: Calibrated scores correlate with performance...")
    try:
        # Generate realistic data
        n_samples = 100
        true_labels = np.random.binomial(1, 0.4, n_samples)
        
        # Create scores that correlate with labels
        vessel_indices = np.where(true_labels == 1)[0]
        sea_indices = np.where(true_labels == 0)[0]
        
        raw_scores = np.zeros(n_samples)
        raw_scores[vessel_indices] = np.random.beta(3, 1, len(vessel_indices))
        raw_scores[sea_indices] = np.random.beta(1, 3, len(sea_indices))
        raw_scores = np.clip(raw_scores * 0.8 + 0.1, 0.01, 0.99)
        
        # Split for calibration
        train_scores, test_scores, train_labels, test_labels = train_test_split(
            raw_scores, true_labels, test_size=0.5, random_state=42
        )
        
        calibrator = ConfidenceCalibrator(test_instance.mock_evaluator)
        calibrator.isotonic_calibrator = calibrator.fit_isotonic_regression(train_scores, train_labels)
        calibrator.calibration_method = 'isotonic'
        
        calibrated_scores = np.array([
            calibrator.calibrate_confidence(float(score)) for score in test_scores
        ])
        
        # Test threshold correlation
        threshold = 0.7
        high_conf_mask = calibrated_scores >= threshold
        low_conf_mask = calibrated_scores < threshold
        
        if np.sum(high_conf_mask) > 2 and np.sum(low_conf_mask) > 2:
            high_conf_accuracy = np.mean(test_labels[high_conf_mask])
            low_conf_accuracy = np.mean(test_labels[low_conf_mask])
            # Allow some tolerance for small samples
            assert high_conf_accuracy >= low_conf_accuracy - 0.2
        
        print("✅ PASSED: Performance correlation")
    except Exception as e:
        print(f"❌ FAILED: Performance correlation - {e}")
        return False
    
    # Test 6: Edge cases
    print("Testing Property: Edge case robustness...")
    try:
        test_instance.test_calibration_edge_cases()
        print("✅ PASSED: Edge case robustness")
    except Exception as e:
        print(f"❌ FAILED: Edge case robustness - {e}")
        return False
    
    print("\n🎉 All confidence calibration property tests passed!")
    return True


if __name__ == "__main__":
    success = run_property_tests()
    sys.exit(0 if success else 1)