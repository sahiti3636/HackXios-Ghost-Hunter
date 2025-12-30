#!/usr/bin/env python3
"""
Training Data Quality Validator

This module implements comprehensive validation for CNN training data quality.
It checks image file existence, readability, dimensions, format consistency,
and removes or reports corrupted/invalid samples.

Requirements: 2.3, 6.1
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
from dataclasses import dataclass

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class ValidationResult:
    """Results from data quality validation."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    corrupted_files: List[str]
    missing_files: List[str]
    dimension_mismatches: List[str]
    format_issues: List[str]
    duplicate_files: List[str]
    validation_errors: List[Dict]
    cleaned_labels: List[Dict]
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_samples == 0:
            return 0.0
        return (self.valid_samples / self.total_samples) * 100.0
    
    @property
    def is_valid_dataset(self) -> bool:
        """Check if dataset meets minimum quality standards."""
        # Require at least 80% of samples to be valid
        return self.success_rate >= 80.0 and self.valid_samples > 0


class TrainingDataQualityValidator:
    """
    Comprehensive validator for CNN training data quality.
    
    Validates:
    - Image file existence and readability
    - Image dimensions and format consistency
    - Label file integrity
    - Removes corrupted/invalid samples
    """
    
    def __init__(self, 
                 expected_image_size: Tuple[int, int] = (64, 64),
                 supported_formats: Set[str] = None,
                 min_intensity_range: float = 10.0,
                 max_file_size_mb: float = 50.0):
        """
        Initialize the validator.
        
        Args:
            expected_image_size: Expected (width, height) for training images
            supported_formats: Set of supported image formats (e.g., {'PNG', 'JPEG'})
            min_intensity_range: Minimum pixel intensity range for valid images
            max_file_size_mb: Maximum file size in MB for images
        """
        self.expected_image_size = expected_image_size
        self.supported_formats = supported_formats or {'PNG', 'JPEG', 'JPG', 'TIFF', 'TIF'}
        self.min_intensity_range = min_intensity_range
        self.max_file_size_mb = max_file_size_mb
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def validate_image_file(self, image_path: str) -> Tuple[bool, str]:
        """
        Validate a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False, "File does not exist"
            
            # Check file size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                return False, f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"
            
            # Try to open and validate image
            with Image.open(image_path) as img:
                # Check format
                if img.format not in self.supported_formats:
                    return False, f"Unsupported format: {img.format}"
                
                # Check dimensions
                if img.size != self.expected_image_size:
                    return False, f"Wrong dimensions: {img.size} != {self.expected_image_size}"
                
                # Convert to array to check pixel values
                img_array = np.array(img)
                
                # Check if image has valid pixel data
                if img_array.size == 0:
                    return False, "Empty image data"
                
                # Check intensity range (avoid completely black or white images)
                intensity_range = img_array.max() - img_array.min()
                if intensity_range < self.min_intensity_range:
                    return False, f"Insufficient intensity range: {intensity_range} < {self.min_intensity_range}"
                
                # Check for NaN or infinite values
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    return False, "Contains NaN or infinite values"
                
                # Verify image can be loaded as tensor (PyTorch compatibility)
                try:
                    tensor = torch.from_numpy(img_array.astype(np.float32))
                    if tensor.numel() == 0:
                        return False, "Cannot convert to valid tensor"
                except Exception as e:
                    return False, f"Tensor conversion failed: {e}"
                
                return True, "Valid"
                
        except Exception as e:
            return False, f"Error loading image: {e}"
    
    def validate_labels_file(self, labels_path: str) -> Tuple[bool, str, List[Dict]]:
        """
        Validate the labels JSON file.
        
        Args:
            labels_path: Path to the labels.json file
            
        Returns:
            Tuple of (is_valid, error_message, labels_data)
        """
        try:
            if not os.path.exists(labels_path):
                return False, "Labels file does not exist", []
            
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            
            if not isinstance(labels, list):
                return False, "Labels file must contain a list", []
            
            if len(labels) == 0:
                return False, "Labels file is empty", []
            
            # Validate each label entry
            required_fields = {'image', 'label'}
            for i, label_entry in enumerate(labels):
                if not isinstance(label_entry, dict):
                    return False, f"Label entry {i} is not a dictionary", []
                
                # Check required fields
                missing_fields = required_fields - set(label_entry.keys())
                if missing_fields:
                    return False, f"Label entry {i} missing fields: {missing_fields}", []
                
                # Validate label value
                if label_entry['label'] not in [0, 1]:
                    return False, f"Label entry {i} has invalid label: {label_entry['label']}", []
                
                # Validate image filename
                if not isinstance(label_entry['image'], str) or not label_entry['image']:
                    return False, f"Label entry {i} has invalid image filename", []
            
            return True, "Valid", labels
            
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {e}", []
        except Exception as e:
            return False, f"Error reading labels file: {e}", []
    
    def find_duplicate_images(self, image_dir: str, labels: List[Dict]) -> List[str]:
        """
        Find duplicate image files based on filename.
        
        Args:
            image_dir: Directory containing images
            labels: List of label entries
            
        Returns:
            List of duplicate filenames
        """
        filenames = [label['image'] for label in labels]
        seen = set()
        duplicates = []
        
        for filename in filenames:
            if filename in seen:
                duplicates.append(filename)
            else:
                seen.add(filename)
        
        return duplicates
    
    def validate_class_distribution(self, labels: List[Dict]) -> Dict[str, int]:
        """
        Analyze class distribution in the dataset.
        
        Args:
            labels: List of label entries
            
        Returns:
            Dictionary with class counts
        """
        class_counts = {}
        for label_entry in labels:
            label_value = label_entry['label']
            class_counts[label_value] = class_counts.get(label_value, 0) + 1
        
        return class_counts
    
    def validate_dataset(self, labels_path: str, images_dir: str) -> ValidationResult:
        """
        Perform comprehensive validation of the training dataset.
        
        Args:
            labels_path: Path to labels.json file
            images_dir: Directory containing training images
            
        Returns:
            ValidationResult with detailed validation information
        """
        self.logger.info("Starting comprehensive dataset validation...")
        
        # Initialize result tracking
        corrupted_files = []
        missing_files = []
        dimension_mismatches = []
        format_issues = []
        validation_errors = []
        cleaned_labels = []
        
        # Validate labels file
        labels_valid, labels_error, labels = self.validate_labels_file(labels_path)
        if not labels_valid:
            self.logger.error(f"Labels file validation failed: {labels_error}")
            return ValidationResult(
                total_samples=0,
                valid_samples=0,
                invalid_samples=0,
                corrupted_files=[],
                missing_files=[],
                dimension_mismatches=[],
                format_issues=[],
                duplicate_files=[],
                validation_errors=[{"type": "labels_file", "error": labels_error}],
                cleaned_labels=[]
            )
        
        # Find duplicate files
        duplicate_files = self.find_duplicate_images(images_dir, labels)
        if duplicate_files:
            self.logger.warning(f"Found {len(duplicate_files)} duplicate files: {duplicate_files[:5]}...")
        
        # Validate each image
        total_samples = len(labels)
        valid_samples = 0
        
        self.logger.info(f"Validating {total_samples} image files...")
        
        for i, label_entry in enumerate(labels):
            image_filename = label_entry['image']
            image_path = os.path.join(images_dir, image_filename)
            
            # Progress reporting
            if (i + 1) % 100 == 0:
                self.logger.info(f"Validated {i + 1}/{total_samples} images...")
            
            is_valid, error_message = self.validate_image_file(image_path)
            
            if is_valid:
                valid_samples += 1
                cleaned_labels.append(label_entry)
            else:
                # Categorize the error
                if "does not exist" in error_message:
                    missing_files.append(image_filename)
                elif "Wrong dimensions" in error_message:
                    dimension_mismatches.append(image_filename)
                elif "format" in error_message.lower():
                    format_issues.append(image_filename)
                else:
                    corrupted_files.append(image_filename)
                
                validation_errors.append({
                    "filename": image_filename,
                    "error": error_message,
                    "label": label_entry.get('label', 'unknown')
                })
        
        invalid_samples = total_samples - valid_samples
        
        # Log class distribution for valid samples
        if cleaned_labels:
            class_dist = self.validate_class_distribution(cleaned_labels)
            self.logger.info(f"Valid samples class distribution: {class_dist}")
        
        # Create validation result
        result = ValidationResult(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            corrupted_files=corrupted_files,
            missing_files=missing_files,
            dimension_mismatches=dimension_mismatches,
            format_issues=format_issues,
            duplicate_files=duplicate_files,
            validation_errors=validation_errors,
            cleaned_labels=cleaned_labels
        )
        
        # Log summary
        self.logger.info(f"Validation complete:")
        self.logger.info(f"  Total samples: {result.total_samples}")
        self.logger.info(f"  Valid samples: {result.valid_samples}")
        self.logger.info(f"  Invalid samples: {result.invalid_samples}")
        self.logger.info(f"  Success rate: {result.success_rate:.1f}%")
        self.logger.info(f"  Dataset valid: {result.is_valid_dataset}")
        
        if result.invalid_samples > 0:
            self.logger.warning(f"Issues found:")
            self.logger.warning(f"  Missing files: {len(result.missing_files)}")
            self.logger.warning(f"  Corrupted files: {len(result.corrupted_files)}")
            self.logger.warning(f"  Dimension mismatches: {len(result.dimension_mismatches)}")
            self.logger.warning(f"  Format issues: {len(result.format_issues)}")
            self.logger.warning(f"  Duplicate files: {len(result.duplicate_files)}")
        
        return result
    
    def clean_dataset(self, labels_path: str, images_dir: str, 
                     output_labels_path: Optional[str] = None) -> ValidationResult:
        """
        Validate dataset and create a cleaned version with only valid samples.
        
        Args:
            labels_path: Path to original labels.json file
            images_dir: Directory containing training images
            output_labels_path: Path for cleaned labels file (optional)
            
        Returns:
            ValidationResult with cleaned dataset information
        """
        self.logger.info("Cleaning dataset by removing invalid samples...")
        
        # Perform validation
        result = self.validate_dataset(labels_path, images_dir)
        
        if not result.is_valid_dataset:
            self.logger.error("Dataset does not meet minimum quality standards!")
            self.logger.error(f"Success rate: {result.success_rate:.1f}% < 80.0%")
            return result
        
        # Save cleaned labels if output path provided
        if output_labels_path and result.cleaned_labels:
            try:
                with open(output_labels_path, 'w') as f:
                    json.dump(result.cleaned_labels, f, indent=2)
                self.logger.info(f"Cleaned labels saved to: {output_labels_path}")
                self.logger.info(f"Removed {result.invalid_samples} invalid samples")
            except Exception as e:
                self.logger.error(f"Failed to save cleaned labels: {e}")
        
        return result
    
    def generate_validation_report(self, result: ValidationResult, 
                                 output_path: Optional[str] = None) -> Dict:
        """
        Generate a detailed validation report.
        
        Args:
            result: ValidationResult from dataset validation
            output_path: Optional path to save report as JSON
            
        Returns:
            Dictionary containing the validation report
        """
        report = {
            "validation_summary": {
                "total_samples": result.total_samples,
                "valid_samples": result.valid_samples,
                "invalid_samples": result.invalid_samples,
                "success_rate_percent": round(result.success_rate, 2),
                "dataset_meets_standards": result.is_valid_dataset
            },
            "issues_found": {
                "missing_files": {
                    "count": len(result.missing_files),
                    "files": result.missing_files[:10]  # First 10 for brevity
                },
                "corrupted_files": {
                    "count": len(result.corrupted_files),
                    "files": result.corrupted_files[:10]
                },
                "dimension_mismatches": {
                    "count": len(result.dimension_mismatches),
                    "files": result.dimension_mismatches[:10]
                },
                "format_issues": {
                    "count": len(result.format_issues),
                    "files": result.format_issues[:10]
                },
                "duplicate_files": {
                    "count": len(result.duplicate_files),
                    "files": result.duplicate_files[:10]
                }
            },
            "class_distribution": self.validate_class_distribution(result.cleaned_labels) if result.cleaned_labels else {},
            "validation_errors": result.validation_errors[:20],  # First 20 errors
            "recommendations": self._generate_recommendations(result)
        }
        
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Validation report saved to: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save validation report: {e}")
        
        return report
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if result.success_rate < 80.0:
            recommendations.append("Dataset quality is below acceptable threshold (80%). Consider regenerating or fixing corrupted samples.")
        
        if len(result.missing_files) > 0:
            recommendations.append(f"Remove {len(result.missing_files)} missing file references from labels.json")
        
        if len(result.dimension_mismatches) > 0:
            recommendations.append(f"Resize {len(result.dimension_mismatches)} images to expected dimensions {self.expected_image_size}")
        
        if len(result.corrupted_files) > 0:
            recommendations.append(f"Regenerate or remove {len(result.corrupted_files)} corrupted image files")
        
        if len(result.duplicate_files) > 0:
            recommendations.append(f"Remove {len(result.duplicate_files)} duplicate file entries from dataset")
        
        # Check class balance
        if result.cleaned_labels:
            class_dist = self.validate_class_distribution(result.cleaned_labels)
            if len(class_dist) == 2:
                counts = list(class_dist.values())
                if max(counts) / min(counts) > 2.0:
                    recommendations.append("Dataset is imbalanced. Consider balancing classes before training.")
        
        if not recommendations:
            recommendations.append("Dataset quality is good. Ready for training.")
        
        return recommendations


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CNN training data quality")
    parser.add_argument("--labels", required=True, help="Path to labels.json file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output-labels", help="Path for cleaned labels.json file")
    parser.add_argument("--report", help="Path for validation report JSON file")
    parser.add_argument("--image-size", nargs=2, type=int, default=[64, 64], 
                       help="Expected image dimensions (width height)")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = TrainingDataQualityValidator(
        expected_image_size=tuple(args.image_size)
    )
    
    # Perform validation and cleaning
    result = validator.clean_dataset(
        labels_path=args.labels,
        images_dir=args.images,
        output_labels_path=args.output_labels
    )
    
    # Generate report
    report = validator.generate_validation_report(
        result=result,
        output_path=args.report
    )
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Total samples: {result.total_samples}")
    print(f"Valid samples: {result.valid_samples}")
    print(f"Success rate: {result.success_rate:.1f}%")
    print(f"Dataset meets standards: {result.is_valid_dataset}")
    
    if not result.is_valid_dataset:
        print("\n⚠️  Dataset quality issues detected!")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        exit(1)
    else:
        print("\n✅ Dataset validation passed!")
        exit(0)


if __name__ == "__main__":
    main()