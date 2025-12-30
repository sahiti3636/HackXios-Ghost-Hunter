"""
Standardized SAR Image Preprocessing Pipeline

This module provides consistent preprocessing functions for both training and inference
to ensure identical normalization parameters and image handling across the pipeline.

Requirements: 5.1, 6.2, 6.4
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from typing import Union, Tuple, Optional


class SARPreprocessor:
    """
    Standardized SAR image preprocessor that ensures consistent preprocessing
    between training and inference phases.
    """
    
    def __init__(self, 
                 img_size: int = 64,
                 log_transform: bool = True,
                 percentile_clip: Tuple[float, float] = (2, 98),
                 normalize_range: Tuple[float, float] = (0, 1)):
        """
        Initialize SAR preprocessor with consistent parameters.
        
        Args:
            img_size: Target image size (square)
            log_transform: Whether to apply log1p transformation
            percentile_clip: Percentile values for clipping (p_low, p_high)
            normalize_range: Target normalization range (min, max)
        """
        self.img_size = img_size
        self.log_transform = log_transform
        self.percentile_clip = percentile_clip
        self.normalize_range = normalize_range
        
        # Create the transform pipeline
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            SARNormalize(
                log_transform=log_transform,
                percentile_clip=percentile_clip,
                normalize_range=normalize_range
            )
        ])
        
        # Training transform with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            SARNormalize(
                log_transform=log_transform,
                percentile_clip=percentile_clip,
                normalize_range=normalize_range
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.2)
        ])
    
    def get_inference_transform(self):
        """Get transform for inference (no augmentation)"""
        return self.base_transform
    
    def get_training_transform(self):
        """Get transform for training (with augmentation)"""
        return self.train_transform
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray], 
                        training: bool = False) -> torch.Tensor:
        """
        Preprocess a single image for model input.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            training: Whether to apply training augmentations
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image if path provided
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image)
        
        # Convert numpy array to PIL Image
        elif isinstance(image, np.ndarray):
            # Ensure proper format for PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply appropriate transform
        transform = self.train_transform if training else self.base_transform
        tensor = transform(image)
        
        return tensor
    
    def preprocess_batch(self, image_paths: list, training: bool = False) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths or PIL Images
            training: Whether to apply training augmentations
            
        Returns:
            Batch tensor of shape (batch_size, 1, height, width)
        """
        tensors = []
        for img_path in image_paths:
            try:
                tensor = self.preprocess_image(img_path, training=training)
                tensors.append(tensor)
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue
        
        if not tensors:
            raise ValueError("No valid images could be processed")
        
        return torch.stack(tensors)


class SARNormalize:
    """
    SAR-specific normalization transform that applies log scaling,
    percentile clipping, and range normalization consistently.
    """
    
    def __init__(self, 
                 log_transform: bool = True,
                 percentile_clip: Tuple[float, float] = (2, 98),
                 normalize_range: Tuple[float, float] = (0, 1)):
        self.log_transform = log_transform
        self.percentile_clip = percentile_clip
        self.normalize_range = normalize_range
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Apply SAR-specific normalization to PIL Image.
        
        Args:
            img: PIL Image in grayscale
            
        Returns:
            Normalized tensor of shape (1, H, W)
        """
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Apply log transformation for SAR data
        if self.log_transform:
            img_array = np.log1p(img_array)
        
        # Percentile clipping to remove speckle extremes
        p_low, p_high = self.percentile_clip
        p2, p98 = np.percentile(img_array, (p_low, p_high))
        img_array = np.clip(img_array, p2, p98)
        
        # Normalize to target range
        min_val, max_val = self.normalize_range
        img_array = (img_array - p2) / (p98 - p2 + 1e-6)
        img_array = img_array * (max_val - min_val) + min_val
        
        # Convert to tensor with channel dimension
        return torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)


def create_standard_preprocessor(img_size: int = 64) -> SARPreprocessor:
    """
    Create a standard SAR preprocessor with default parameters.
    
    Args:
        img_size: Target image size
        
    Returns:
        Configured SARPreprocessor instance
    """
    return SARPreprocessor(
        img_size=img_size,
        log_transform=True,
        percentile_clip=(2, 98),
        normalize_range=(0, 1)
    )


def validate_preprocessing_consistency(preprocessor: SARPreprocessor, 
                                     test_image_path: str) -> dict:
    """
    Validate that preprocessing produces consistent results.
    
    Args:
        preprocessor: SARPreprocessor instance to test
        test_image_path: Path to test image
        
    Returns:
        Dictionary with validation results
    """
    try:
        # Test inference preprocessing
        tensor1 = preprocessor.preprocess_image(test_image_path, training=False)
        tensor2 = preprocessor.preprocess_image(test_image_path, training=False)
        
        # Check consistency (should be identical for inference)
        consistency_check = torch.allclose(tensor1, tensor2, atol=1e-6)
        
        # Check tensor properties
        shape_check = tensor1.shape[0] == 1  # Single channel
        size_check = tensor1.shape[1] == tensor1.shape[2] == preprocessor.img_size
        range_check = (tensor1.min() >= 0) and (tensor1.max() <= 1)
        
        return {
            'consistency': consistency_check,
            'correct_shape': shape_check,
            'correct_size': size_check,
            'correct_range': range_check,
            'tensor_shape': tuple(tensor1.shape),
            'value_range': (float(tensor1.min()), float(tensor1.max()))
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'consistency': False,
            'correct_shape': False,
            'correct_size': False,
            'correct_range': False
        }


# Backward compatibility - maintain the old interface
class SARPreprocess:
    """Legacy SARPreprocess class for backward compatibility"""
    
    def __init__(self):
        self.preprocessor = create_standard_preprocessor()
    
    def __call__(self, img):
        """Legacy interface - converts PIL Image to tensor"""
        return SARNormalize()(img)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = create_standard_preprocessor()
    print("✅ SAR Preprocessor created successfully")
    print(f"   Image size: {preprocessor.img_size}")
    print(f"   Log transform: {preprocessor.log_transform}")
    print(f"   Percentile clip: {preprocessor.percentile_clip}")
    print(f"   Normalize range: {preprocessor.normalize_range}")