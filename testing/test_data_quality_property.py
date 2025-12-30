#!/usr/bin/env python3
"""
Property-Based Test for Training Data Quality

This module implements Property 2: Training Data Quality Assurance
**Validates: Requirements 2.3, 6.1**

Property 2: For any training image in the dataset, the image file should exist, 
be readable, and have valid dimensions matching the expected input size.

Feature: cnn-confidence-improvement, Property 2: Training Data Quality Assurance
"""

import json
import os
import tempfile
import shutil
import random
from typing import Dict, List, Tuple, Set
import numpy as np
from PIL import Image
import cv2

# Import the classes we're testing
from training_data_quality_validator import TrainingDataQualityValidator, ValidationResult


class DataQualityPropertyTest:
    """Property-based test for training data quality validation."""
    
    def __init__(self, num_iterations: int = 100):
        """
        Initialize the property test.
        
        Args:
            num_iterations: Number of test iterations to run (minimum 100)
        """
        self.num_iterations = max(num_iterations, 100)
        self.test_results = []
        self.failures = []
        self.expected_image_size = (64, 64)
    
    def generate_valid_image(self, filename: str, temp_dir: str, 
                           image_type: str = "normal") -> str:
        """
        Generate a valid test image.
        
        Args:
            filename: Name for the image file
            temp_dir: Temporary directory for the image
            image_type: Type of image to generate ("normal", "ship", "sea")
            
        Returns:
            Path to the generated image
        """
        filepath = os.path.join(temp_dir, 'images', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if image_type == "ship":
            # Generate ship-like pattern (higher intensity center)
            img_array = np.random.randint(50, 150, self.expected_image_size, dtype=np.uint8)
            center_x, center_y = self.expected_image_size[0] // 2, self.expected_image_size[1] // 2
            img_array[center_y-5:center_y+5, center_x-8:center_x+8] = np.random.randint(180, 255, (10, 16))
        elif image_type == "sea":
            # Generate sea-like pattern (lower, more uniform intensity)
            img_array = np.random.randint(10, 80, self.expected_image_size, dtype=np.uint8)
        else:
            # Generate normal random image
            img_array = np.random.randint(0, 255, self.expected_image_size, dtype=np.uint8)
        
        Image.fromarray(img_array, mode='L').save(filepath)
        return filepath
    
    def generate_corrupted_image(self, filename: str, temp_dir: str, 
                               corruption_type: str) -> str:
        """
        Generate a corrupted test image.
        
        Args:
            filename: Name for the image file
            temp_dir: Temporary directory for the image
            corruption_type: Type of corruption to apply
            
        Returns:
            Path to the corrupted image
        """
        filepath = os.path.join(temp_dir, 'images', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if corruption_type == "wrong_dimensions":
            # Create image with wrong dimensions
            wrong_size = (32, 32) if random.choice([True, False]) else (128, 128)
            img_array = np.random.randint(0, 255, wrong_size, dtype=np.uint8)
            Image.fromarray(img_array, mode='L').save(filepath)
        
        elif corruption_type == "empty_file":
            # Create empty file
            with open(filepath, 'w') as f:
                pass
        
        elif corruption_type == "invalid_format":
            # Create text file with image extension
            with open(filepath, 'w') as f:
                f.write("This is not an image file")
        
        elif corruption_type == "all_black":
            # Create all-black image (insufficient intensity range - should be < 10.0)
            img_array = np.zeros(self.expected_image_size, dtype=np.uint8)
            Image.fromarray(img_array, mode='L').save(filepath)
        
        elif corruption_type == "all_white":
            # Create all-white image (insufficient intensity range - should be < 10.0)
            img_array = np.full(self.expected_image_size, 255, dtype=np.uint8)
            Image.fromarray(img_array, mode='L').save(filepath)
        
        elif corruption_type == "low_contrast":
            # Create low contrast image (intensity range < 10.0)
            base_value = 128
            img_array = np.full(self.expected_image_size, base_value, dtype=np.uint8)
            # Add very small variations (< 10 pixel difference)
            noise = np.random.randint(-3, 4, self.expected_image_size)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            Image.fromarray(img_array, mode='L').save(filepath)
        
        elif corruption_type == "nan_values":
            # Create image with NaN values (will be saved as corrupted)
            img_array = np.random.randint(0, 255, self.expected_image_size, dtype=np.float32)
            img_array[10:20, 10:20] = np.nan
            # PIL will handle NaN by converting to 0, but our validator should catch this
            Image.fromarray(img_array.astype(np.uint8), mode='L').save(filepath)
        
        elif corruption_type == "truncated":
            # Create a valid image first, then truncate the file
            img_array = np.random.randint(0, 255, self.expected_image_size, dtype=np.uint8)
            Image.fromarray(img_array, mode='L').save(filepath)
            
            # Truncate the file
            with open(filepath, 'r+b') as f:
                f.seek(0, 2)  # Go to end
                size = f.tell()
                f.truncate(size // 2)  # Cut file in half
        
        return filepath
    
    def generate_test_dataset(self, 
                            valid_count: int, 
                            corrupted_specs: List[Tuple[str, str]], 
                            missing_files: List[str],
                            temp_dir: str) -> Tuple[str, str]:
        """
        Generate a test dataset with specified valid and corrupted samples.
        
        Args:
            valid_count: Number of valid samples to create
            corrupted_specs: List of (filename, corruption_type) tuples
            missing_files: List of filenames that should be missing
            temp_dir: Temporary directory for test files
            
        Returns:
            Tuple of (labels_file_path, images_dir_path)
        """
        images_dir = os.path.join(temp_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        labels = []
        
        # Generate valid samples
        for i in range(valid_count):
            label_class = random.choice([0, 1])
            filename = f"{'ship' if label_class == 1 else 'sea'}_{i+1}.png"
            image_type = "ship" if label_class == 1 else "sea"
            
            self.generate_valid_image(filename, temp_dir, image_type)
            
            labels.append({
                "image": filename,
                "label": label_class,
                "center_coordinates": [random.randint(32, 1000), random.randint(32, 1000)],
                "satellite_source": f"test_sat_{random.randint(1, 3)}"
            })
        
        # Generate corrupted samples
        for filename, corruption_type in corrupted_specs:
            label_class = random.choice([0, 1])
            
            self.generate_corrupted_image(filename, temp_dir, corruption_type)
            
            labels.append({
                "image": filename,
                "label": label_class,
                "center_coordinates": [random.randint(32, 1000), random.randint(32, 1000)],
                "satellite_source": f"test_sat_{random.randint(1, 3)}"
            })
        
        # Add missing file entries (files that don't exist)
        for filename in missing_files:
            label_class = random.choice([0, 1])
            
            labels.append({
                "image": filename,
                "label": label_class,
                "center_coordinates": [random.randint(32, 1000), random.randint(32, 1000)],
                "satellite_source": f"test_sat_{random.randint(1, 3)}"
            })
        
        # Save labels file
        labels_file = os.path.join(temp_dir, 'labels.json')
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        return labels_file, images_dir
    
    def check_data_quality_property(self, validation_result: ValidationResult, 
                                  expected_valid: int, expected_invalid: int) -> Dict:
        """
        Check if the validation result satisfies the data quality property.
        
        Args:
            validation_result: Result from TrainingDataQualityValidator
            expected_valid: Expected number of valid samples
            expected_invalid: Expected number of invalid samples
            
        Returns:
            Dict with test results
        """
        try:
            # Property: All valid samples should have readable images with correct dimensions
            total_expected = expected_valid + expected_invalid
            
            # Check if validator correctly identified valid vs invalid samples
            correct_valid_count = validation_result.valid_samples == expected_valid
            correct_total_count = validation_result.total_samples == total_expected
            correct_invalid_count = validation_result.invalid_samples == expected_invalid
            
            # Property satisfaction: validator should correctly identify all issues
            property_satisfied = (correct_valid_count and 
                                correct_total_count and 
                                correct_invalid_count)
            
            # Additional checks: all valid samples should actually be valid
            all_valid_samples_correct = True
            if validation_result.cleaned_labels:
                # Check that cleaned labels only contain valid entries
                for label_entry in validation_result.cleaned_labels:
                    if 'image' not in label_entry or 'label' not in label_entry:
                        all_valid_samples_correct = False
                        break
            
            return {
                'passed': property_satisfied and all_valid_samples_correct,
                'reason': self._generate_failure_reason(
                    validation_result, expected_valid, expected_invalid, 
                    correct_valid_count, correct_total_count, correct_invalid_count,
                    all_valid_samples_correct
                ),
                'expected_valid': expected_valid,
                'expected_invalid': expected_invalid,
                'actual_valid': validation_result.valid_samples,
                'actual_invalid': validation_result.invalid_samples,
                'actual_total': validation_result.total_samples,
                'success_rate': validation_result.success_rate,
                'validator_detected_issues': {
                    'missing_files': len(validation_result.missing_files),
                    'corrupted_files': len(validation_result.corrupted_files),
                    'dimension_mismatches': len(validation_result.dimension_mismatches),
                    'format_issues': len(validation_result.format_issues)
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'reason': f'Error checking property: {e}',
                'expected_valid': expected_valid,
                'expected_invalid': expected_invalid,
                'actual_valid': 0,
                'actual_invalid': 0,
                'actual_total': 0,
                'success_rate': 0.0
            }
    
    def _generate_failure_reason(self, result: ValidationResult, 
                               expected_valid: int, expected_invalid: int,
                               correct_valid: bool, correct_total: bool, 
                               correct_invalid: bool, all_valid_correct: bool) -> str:
        """Generate detailed failure reason."""
        if not correct_total:
            return f"Total count mismatch: expected {expected_valid + expected_invalid}, got {result.total_samples}"
        
        if not correct_valid:
            return f"Valid count mismatch: expected {expected_valid}, got {result.valid_samples}"
        
        if not correct_invalid:
            return f"Invalid count mismatch: expected {expected_invalid}, got {result.invalid_samples}"
        
        if not all_valid_correct:
            return "Some samples marked as valid are actually invalid"
        
        return "Property satisfied"
    
    def test_training_data_quality_property(self) -> bool:
        """
        Test Property 2: Training Data Quality Assurance
        
        For any training image in the dataset, the image file should exist, 
        be readable, and have valid dimensions matching the expected input size.
        
        Returns:
            bool: True if all iterations pass, False otherwise
        """
        print(f"Running Property Test: Training Data Quality Assurance")
        print(f"Iterations: {self.num_iterations}")
        print(f"Property: All training images should exist, be readable, and have correct dimensions")
        print("-" * 80)
        
        passed_count = 0
        failed_count = 0
        
        corruption_types = [
            "wrong_dimensions", "empty_file", "invalid_format", 
            "all_black", "all_white", "low_contrast", "truncated"
        ]
        
        for iteration in range(self.num_iterations):
            # Generate random test scenario
            valid_count = random.randint(10, 50)
            corrupted_count = random.randint(1, 10)
            missing_count = random.randint(0, 5)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Create corrupted file specifications
                    corrupted_specs = []
                    for i in range(corrupted_count):
                        corruption_type = random.choice(corruption_types)
                        filename = f"corrupted_{i+1}_{corruption_type}.png"
                        corrupted_specs.append((filename, corruption_type))
                    
                    # Create missing file specifications
                    missing_files = [f"missing_{i+1}.png" for i in range(missing_count)]
                    
                    # Generate test dataset
                    labels_file, images_dir = self.generate_test_dataset(
                        valid_count=valid_count,
                        corrupted_specs=corrupted_specs,
                        missing_files=missing_files,
                        temp_dir=temp_dir
                    )
                    
                    # Test the validator
                    validator = TrainingDataQualityValidator(
                        expected_image_size=self.expected_image_size
                    )
                    
                    validation_result = validator.validate_dataset(labels_file, images_dir)
                    
                    # Check if the property is satisfied
                    expected_valid = valid_count
                    expected_invalid = corrupted_count + missing_count
                    
                    result = self.check_data_quality_property(
                        validation_result, expected_valid, expected_invalid
                    )
                    
                    self.test_results.append({
                        'iteration': iteration + 1,
                        'expected_valid': expected_valid,
                        'expected_invalid': expected_invalid,
                        'actual_valid': result['actual_valid'],
                        'actual_invalid': result['actual_invalid'],
                        'passed': result['passed'],
                        'success_rate': result['success_rate'],
                        'reason': result['reason'],
                        'validator_issues': result.get('validator_detected_issues', {})
                    })
                    
                    if result['passed']:
                        passed_count += 1
                    else:
                        failed_count += 1
                        self.failures.append({
                            'iteration': iteration + 1,
                            'reason': result['reason'],
                            'expected_valid': expected_valid,
                            'expected_invalid': expected_invalid,
                            'actual_valid': result['actual_valid'],
                            'actual_invalid': result['actual_invalid'],
                            'success_rate': result['success_rate']
                        })
                    
                    # Progress indicator
                    if (iteration + 1) % 20 == 0:
                        print(f"Progress: {iteration + 1}/{self.num_iterations} iterations completed")
                
                except Exception as e:
                    failed_count += 1
                    self.failures.append({
                        'iteration': iteration + 1,
                        'reason': f'Exception: {e}',
                        'expected_valid': valid_count,
                        'expected_invalid': corrupted_count + missing_count
                    })
        
        # Print results
        print(f"\nProperty Test Results:")
        print(f"Total iterations: {self.num_iterations}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {(passed_count / self.num_iterations) * 100:.1f}%")
        
        if failed_count > 0:
            print(f"\nFirst 5 failures:")
            for i, failure in enumerate(self.failures[:5]):
                print(f"  {i+1}. Iteration {failure['iteration']}: {failure['reason']}")
                print(f"     Expected: {failure['expected_valid']} valid, {failure['expected_invalid']} invalid")
                print(f"     Actual: {failure.get('actual_valid', 'N/A')} valid, {failure.get('actual_invalid', 'N/A')} invalid")
        
        return failed_count == 0
    
    def test_validator_edge_cases(self) -> bool:
        """
        Test edge cases for the data quality validator.
        
        Returns:
            bool: True if all edge case tests pass
        """
        print(f"\nTesting data quality validator edge cases...")
        
        edge_cases = [
            ("empty_dataset", 0, 0, []),
            ("only_valid", 10, 0, []),
            ("only_invalid", 0, 5, [("bad1.png", "wrong_dimensions"), ("bad2.png", "empty_file")]),
            ("mixed_issues", 5, 3, [("bad1.png", "all_black"), ("bad2.png", "truncated"), ("bad3.png", "low_contrast")]),
        ]
        
        passed = 0
        failed = 0
        
        for case_name, valid_count, missing_count, corrupted_specs in edge_cases:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Generate test dataset
                    labels_file, images_dir = self.generate_test_dataset(
                        valid_count=valid_count,
                        corrupted_specs=corrupted_specs,
                        missing_files=[f"missing_{i}.png" for i in range(missing_count)],
                        temp_dir=temp_dir
                    )
                    
                    # Test validator
                    validator = TrainingDataQualityValidator(
                        expected_image_size=self.expected_image_size
                    )
                    
                    validation_result = validator.validate_dataset(labels_file, images_dir)
                    
                    # Check results
                    expected_invalid = len(corrupted_specs) + missing_count
                    
                    if (validation_result.valid_samples == valid_count and 
                        validation_result.invalid_samples == expected_invalid):
                        passed += 1
                        print(f"  ✓ {case_name}: {valid_count} valid, {expected_invalid} invalid")
                    else:
                        failed += 1
                        print(f"  ✗ {case_name}: Expected {valid_count} valid, {expected_invalid} invalid")
                        print(f"    Got {validation_result.valid_samples} valid, {validation_result.invalid_samples} invalid")
                
                except Exception as e:
                    failed += 1
                    print(f"  ✗ {case_name}: Error - {e}")
        
        print(f"Edge case test results: {passed} passed, {failed} failed")
        return failed == 0


def run_property_tests():
    """
    Run all property-based tests for training data quality.
    
    **Feature: cnn-confidence-improvement, Property 2: Training Data Quality Assurance**
    **Validates: Requirements 2.3, 6.1**
    """
    print("=" * 80)
    print("PROPERTY-BASED TEST: TRAINING DATA QUALITY")
    print("Feature: cnn-confidence-improvement, Property 2: Training Data Quality Assurance")
    print("Validates: Requirements 2.3, 6.1")
    print("=" * 80)
    
    # Initialize test with minimum 100 iterations as required
    tester = DataQualityPropertyTest(num_iterations=100)
    
    # Run main property test
    main_test_passed = tester.test_training_data_quality_property()
    
    # Run edge case tests
    edge_case_test_passed = tester.test_validator_edge_cases()
    
    # Overall result
    all_tests_passed = main_test_passed and edge_case_test_passed
    
    print("\n" + "=" * 80)
    print("PROPERTY TEST SUMMARY")
    print("=" * 80)
    print(f"Main Property Test (100 iterations): {'PASSED' if main_test_passed else 'FAILED'}")
    print(f"Edge Case Tests: {'PASSED' if edge_case_test_passed else 'FAILED'}")
    print(f"Overall Result: {'PASSED' if all_tests_passed else 'FAILED'}")
    
    if not all_tests_passed:
        print("\n⚠️  Property test failed! The data quality validator")
        print("   does not correctly identify valid vs invalid training data.")
        print("   Review the failures above for details.")
        
        # Return the first failure as counterexample
        if tester.failures:
            first_failure = tester.failures[0]
            return False, first_failure
    else:
        print("\n✅ All property tests passed! The data quality validator")
        print("   correctly identifies and handles invalid training data.")
    
    return all_tests_passed, None


if __name__ == "__main__":
    success, counterexample = run_property_tests()
    
    if not success:
        print(f"\nCounterexample: {counterexample}")
        exit(1)
    else:
        exit(0)