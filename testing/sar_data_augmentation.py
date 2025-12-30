#!/usr/bin/env python3
"""
SAR Data Augmentation Module

This module implements enhanced data augmentation specifically designed for SAR imagery.
It includes appropriate rotations, flips, and intensity variations while preserving
vessel characteristics and balancing augmentation between ship and sea classes.

Requirements: 2.5, 6.4
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms
import cv2

@dataclass
class AugmentationConfig:
    """Configuration for SAR data augmentation."""
    # Rotation parameters
    rotation_angles: List[float] = None  # Degrees, None for random [-15, 15]
    rotation_probability: float = 0.7
    
    # Flip parameters
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.3
    
    # Intensity augmentation
    intensity_variation_prob: float = 0.6
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    
    # Noise augmentation (SAR-specific)
    speckle_noise_prob: float = 0.4
    speckle_noise_variance: float = 0.1
    
    # Gaussian blur (atmospheric effects)
    blur_prob: float = 0.2
    blur_sigma_range: Tuple[float, float] = (0.5, 1.5)
    
    # Elastic deformation (sea surface variations)
    elastic_deform_prob: float = 0.3
    elastic_alpha_range: Tuple[float, float] = (5.0, 15.0)
    elastic_sigma: float = 3.0
    
    # Class-specific augmentation balance
    ship_augmentation_factor: float = 2.0  # More augmentation for ships
    sea_augmentation_factor: float = 1.5
    
    def __post_init__(self):
        """Set default rotation angles if not provided."""
        if self.rotation_angles is None:
            self.rotation_angles = [-15, -10, -5, 5, 10, 15]


class SARDataAugmentor:
    """
    Enhanced data augmentation specifically designed for SAR imagery.
    
    Features:
    - SAR-appropriate transformations that preserve vessel characteristics
    - Speckle noise simulation for realistic SAR artifacts
    - Balanced augmentation between ship and sea classes
    - Intensity variations that maintain vessel detectability
    """
    
    def __init__(self, config: AugmentationConfig = None):
        """
        Initialize the SAR data augmentor.
        
        Args:
            config: AugmentationConfig with augmentation parameters
        """
        self.config = config or AugmentationConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Random seed for reproducibility during testing
        self.rng = np.random.RandomState()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducible augmentation."""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        np.random.seed(seed)
    
    def add_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add multiplicative speckle noise typical of SAR imagery.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Image with speckle noise added
        """
        if self.rng.random() > self.config.speckle_noise_prob:
            return image
        
        # Generate multiplicative speckle noise
        noise = self.rng.normal(1.0, self.config.speckle_noise_variance, image.shape)
        noise = np.clip(noise, 0.5, 1.5)  # Limit noise range
        
        # Apply speckle noise
        noisy_image = image * noise
        return np.clip(noisy_image, 0, 255).astype(image.dtype)
    
    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction for intensity variation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Gamma-corrected image
        """
        if self.rng.random() > self.config.intensity_variation_prob:
            return image
        
        gamma = self.rng.uniform(*self.config.gamma_range)
        
        # Normalize to [0, 1], apply gamma, then scale back
        normalized = image.astype(np.float32) / 255.0
        gamma_corrected = np.power(normalized, gamma)
        return (gamma_corrected * 255).astype(image.dtype)
    
    def apply_elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply elastic deformation to simulate sea surface variations.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Elastically deformed image
        """
        if self.rng.random() > self.config.elastic_deform_prob:
            return image
        
        h, w = image.shape[:2]
        alpha = self.rng.uniform(*self.config.elastic_alpha_range)
        sigma = self.config.elastic_sigma
        
        # Generate random displacement fields
        dx = self.rng.uniform(-1, 1, (h, w)) * alpha
        dy = self.rng.uniform(-1, 1, (h, w)) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, h - 1).astype(np.float32)
        
        # Apply deformation
        if len(image.shape) == 3:
            deformed = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR)
        else:
            deformed = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR)
        
        return deformed
    
    def apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply rotation augmentation with SAR-appropriate angles.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Rotated image
        """
        if self.rng.random() > self.config.rotation_probability:
            return image
        
        angle = self.rng.choice(self.config.rotation_angles)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                               borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def apply_flips(self, image: np.ndarray) -> np.ndarray:
        """
        Apply horizontal and vertical flips.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Flipped image
        """
        # Horizontal flip
        if self.rng.random() < self.config.horizontal_flip_prob:
            image = cv2.flip(image, 1)
        
        # Vertical flip
        if self.rng.random() < self.config.vertical_flip_prob:
            image = cv2.flip(image, 0)
        
        return image
    
    def apply_intensity_variations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast variations.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Image with intensity variations
        """
        if self.rng.random() > self.config.intensity_variation_prob:
            return image
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(image)
        
        # Apply brightness variation
        brightness_factor = self.rng.uniform(*self.config.brightness_range)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # Apply contrast variation
        contrast_factor = self.rng.uniform(*self.config.contrast_range)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        return np.array(pil_image)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to simulate atmospheric effects.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Blurred image
        """
        if self.rng.random() > self.config.blur_prob:
            return image
        
        sigma = self.rng.uniform(*self.config.blur_sigma_range)
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        return blurred
    
    def augment_image(self, image: np.ndarray, label: int, 
                     preserve_vessel_characteristics: bool = True) -> np.ndarray:
        """
        Apply comprehensive augmentation to a single image.
        
        Args:
            image: Input image as numpy array
            label: Class label (0 for sea, 1 for ship)
            preserve_vessel_characteristics: Whether to use vessel-preserving augmentation
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Apply transformations in order of least to most destructive
        
        # 1. Geometric transformations (preserve vessel shape)
        augmented = self.apply_flips(augmented)
        augmented = self.apply_rotation(augmented)
        
        # 2. Intensity variations (preserve vessel contrast)
        augmented = self.apply_intensity_variations(augmented)
        augmented = self.apply_gamma_correction(augmented)
        
        # 3. Noise and blur (add realism but may affect vessel detection)
        if not preserve_vessel_characteristics or label == 0:  # More aggressive for sea
            augmented = self.add_speckle_noise(augmented)
            augmented = self.apply_gaussian_blur(augmented)
        else:  # Gentler for ships to preserve detectability
            # Reduced probability for ships
            original_speckle_prob = self.config.speckle_noise_prob
            original_blur_prob = self.config.blur_prob
            
            self.config.speckle_noise_prob *= 0.5
            self.config.blur_prob *= 0.3
            
            augmented = self.add_speckle_noise(augmented)
            augmented = self.apply_gaussian_blur(augmented)
            
            # Restore original probabilities
            self.config.speckle_noise_prob = original_speckle_prob
            self.config.blur_prob = original_blur_prob
        
        # 4. Elastic deformation (mainly for sea patches)
        if label == 0:  # Only apply to sea patches
            augmented = self.apply_elastic_deformation(augmented)
        
        return augmented
    
    def generate_augmented_dataset(self, 
                                 labels_path: str, 
                                 images_dir: str,
                                 output_dir: str,
                                 augmentation_multiplier: Optional[Dict[int, int]] = None) -> Dict:
        """
        Generate an augmented dataset with balanced class augmentation.
        
        Args:
            labels_path: Path to original labels.json file
            images_dir: Directory containing original images
            output_dir: Directory for augmented dataset
            augmentation_multiplier: Dict mapping class labels to augmentation counts
            
        Returns:
            Dictionary with augmentation statistics
        """
        self.logger.info("Starting SAR dataset augmentation...")
        
        # Load original dataset
        with open(labels_path, 'r') as f:
            original_labels = json.load(f)
        
        # Create output directories
        output_images_dir = os.path.join(output_dir, 'images')
        os.makedirs(output_images_dir, exist_ok=True)
        
        # Set default augmentation multipliers if not provided
        if augmentation_multiplier is None:
            augmentation_multiplier = {
                0: int(self.config.sea_augmentation_factor),    # Sea patches
                1: int(self.config.ship_augmentation_factor)   # Ship patches
            }
        
        # Track statistics
        stats = {
            'original_samples': len(original_labels),
            'augmented_samples': 0,
            'class_distribution': {0: 0, 1: 0},
            'augmented_class_distribution': {0: 0, 1: 0},
            'augmentation_methods_used': set()
        }
        
        # Count original class distribution
        for label_entry in original_labels:
            stats['class_distribution'][label_entry['label']] += 1
        
        augmented_labels = []
        
        # Copy original samples
        for label_entry in original_labels:
            # Copy original image
            original_image_path = os.path.join(images_dir, label_entry['image'])
            output_image_path = os.path.join(output_images_dir, label_entry['image'])
            
            if os.path.exists(original_image_path):
                image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(output_image_path, image)
                
                # Add to augmented labels
                augmented_labels.append(label_entry.copy())
                stats['augmented_class_distribution'][label_entry['label']] += 1
        
        # Generate augmented samples
        for label_entry in original_labels:
            label_class = label_entry['label']
            multiplier = augmentation_multiplier.get(label_class, 1)
            
            if multiplier <= 1:
                continue
            
            original_image_path = os.path.join(images_dir, label_entry['image'])
            
            if not os.path.exists(original_image_path):
                self.logger.warning(f"Original image not found: {original_image_path}")
                continue
            
            # Load original image
            image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.logger.warning(f"Could not load image: {original_image_path}")
                continue
            
            # Generate augmented versions
            for aug_idx in range(multiplier - 1):  # -1 because original is already included
                # Create augmented image
                augmented_image = self.augment_image(image, label_class)
                
                # Generate new filename
                base_name = os.path.splitext(label_entry['image'])[0]
                extension = os.path.splitext(label_entry['image'])[1]
                aug_filename = f"{base_name}_aug_{aug_idx + 1}{extension}"
                aug_image_path = os.path.join(output_images_dir, aug_filename)
                
                # Save augmented image
                cv2.imwrite(aug_image_path, augmented_image)
                
                # Create augmented label entry
                aug_label_entry = label_entry.copy()
                aug_label_entry['image'] = aug_filename
                aug_label_entry['augmentation_applied'] = True
                aug_label_entry['augmentation_index'] = aug_idx + 1
                aug_label_entry['original_image'] = label_entry['image']
                
                augmented_labels.append(aug_label_entry)
                stats['augmented_samples'] += 1
                stats['augmented_class_distribution'][label_class] += 1
        
        # Save augmented labels
        output_labels_path = os.path.join(output_dir, 'labels.json')
        with open(output_labels_path, 'w') as f:
            json.dump(augmented_labels, f, indent=2)
        
        # Update statistics
        stats['total_samples'] = len(augmented_labels)
        stats['augmentation_factor'] = {
            class_id: stats['augmented_class_distribution'][class_id] / max(1, stats['class_distribution'][class_id])
            for class_id in [0, 1]
        }
        
        # Log results
        self.logger.info(f"Augmentation complete:")
        self.logger.info(f"  Original samples: {stats['original_samples']}")
        self.logger.info(f"  Total samples: {stats['total_samples']}")
        self.logger.info(f"  Augmented samples: {stats['augmented_samples']}")
        self.logger.info(f"  Original distribution: {stats['class_distribution']}")
        self.logger.info(f"  Final distribution: {stats['augmented_class_distribution']}")
        self.logger.info(f"  Augmentation factors: {stats['augmentation_factor']}")
        
        return stats
    
    def create_pytorch_transforms(self, training: bool = True) -> transforms.Compose:
        """
        Create PyTorch transforms for SAR data augmentation.
        
        Args:
            training: Whether to include augmentation (training) or just normalization (validation)
            
        Returns:
            Composed transforms for PyTorch DataLoader
        """
        if training:
            transform_list = [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=self.config.horizontal_flip_prob),
                transforms.RandomVerticalFlip(p=self.config.vertical_flip_prob),
                transforms.RandomRotation(
                    degrees=max(abs(min(self.config.rotation_angles)), 
                              abs(max(self.config.rotation_angles))),
                    fill=0
                ),
                transforms.ColorJitter(
                    brightness=self.config.brightness_range,
                    contrast=self.config.contrast_range
                ),
                transforms.ToTensor(),
                # Custom SAR preprocessing
                SARNormalization(),
            ]
        else:
            transform_list = [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                SARNormalization(),
            ]
        
        return transforms.Compose(transform_list)


class SARNormalization:
    """Custom normalization for SAR imagery."""
    
    def __call__(self, tensor):
        """
        Apply SAR-specific normalization.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Log scaling for SAR data
        tensor = torch.log1p(tensor)
        
        # Percentile-based normalization
        p2 = torch.quantile(tensor, 0.02)
        p98 = torch.quantile(tensor, 0.98)
        
        # Clip and normalize
        tensor = torch.clamp(tensor, p2, p98)
        tensor = (tensor - p2) / (p98 - p2 + 1e-6)
        
        return tensor


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment SAR training dataset")
    parser.add_argument("--labels", required=True, help="Path to labels.json file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output directory for augmented dataset")
    parser.add_argument("--ship-factor", type=float, default=2.0, 
                       help="Augmentation factor for ship samples")
    parser.add_argument("--sea-factor", type=float, default=1.5,
                       help="Augmentation factor for sea samples")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create augmentation config
    config = AugmentationConfig()
    config.ship_augmentation_factor = args.ship_factor
    config.sea_augmentation_factor = args.sea_factor
    
    # Initialize augmentor
    augmentor = SARDataAugmentor(config)
    
    if args.seed is not None:
        augmentor.set_seed(args.seed)
    
    # Generate augmented dataset
    stats = augmentor.generate_augmented_dataset(
        labels_path=args.labels,
        images_dir=args.images,
        output_dir=args.output,
        augmentation_multiplier={
            0: int(args.sea_factor),
            1: int(args.ship_factor)
        }
    )
    
    print(f"\nAugmentation Summary:")
    print(f"Original samples: {stats['original_samples']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Augmentation factors: {stats['augmentation_factor']}")
    print(f"Final class distribution: {stats['augmented_class_distribution']}")
    
    print(f"\n✅ Augmented dataset saved to: {args.output}")


if __name__ == "__main__":
    main()