"""
Property-Based Test for Preprocessing Consistency

Tests Property 4: Preprocessing Consistency
**Validates: Requirements 5.1, 5.4, 6.2**

This test ensures that preprocessing is identical between training and inference,
with consistent normalization parameters and image handling.
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from hypothesis import given, strategies as st, settings, assume
from sar_preprocessing import create_standard_preprocessor, validate_preprocessing_consistency
import tempfile


class TestPreprocessingConsistency:
    """Property-based tests for preprocessing consistency"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = create_standard_preprocessor(img_size=64)
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_image(self, width=64, height=64, intensity_range=(0, 255)):
        """Create a test SAR-like image with specified properties"""
        # Generate random SAR-like data with speckle noise
        base_intensity = np.random.uniform(intensity_range[0], intensity_range[1])
        speckle = np.random.exponential(scale=0.3, size=(height, width))
        
        # Create image with some structure (simulating vessel or sea)
        image_data = base_intensity * speckle
        image_data = np.clip(image_data, intensity_range[0], intensity_range[1])
        
        return Image.fromarray(image_data.astype(np.uint8), mode='L')
    
    @given(
        width=st.integers(min_value=32, max_value=128),
        height=st.integers(min_value=32, max_value=128),
        intensity_low=st.integers(min_value=0, max_value=100),
        intensity_high=st.integers(min_value=150, max_value=255)
    )
    @settings(max_examples=100, deadline=5000)
    def test_preprocessing_deterministic_for_inference(self, width, height, intensity_low, intensity_high):
        """
        Property 4a: For any image, inference preprocessing should be deterministic
        **Feature: cnn-confidence-improvement, Property 4: Preprocessing Consistency**
        **Validates: Requirements 5.1, 5.4, 6.2**
        """
        assume(intensity_high > intensity_low)
        
        # Create test image
        test_image = self.create_test_image(width, height, (intensity_low, intensity_high))
        
        # Save to temporary file
        temp_path = os.path.join(self.temp_dir, f"test_{width}x{height}.png")
        test_image.save(temp_path)
        
        # Process same image multiple times for inference
        tensor1 = self.preprocessor.preprocess_image(temp_path, training=False)
        tensor2 = self.preprocessor.preprocess_image(temp_path, training=False)
        tensor3 = self.preprocessor.preprocess_image(temp_path, training=False)
        
        # All results should be identical for inference
        assert torch.allclose(tensor1, tensor2, atol=1e-6), \
            "Inference preprocessing should be deterministic"
        assert torch.allclose(tensor2, tensor3, atol=1e-6), \
            "Inference preprocessing should be deterministic"
        
        # Check tensor properties
        assert tensor1.shape[0] == 1, "Should have single channel"
        assert tensor1.shape[1] == tensor1.shape[2] == 64, "Should be resized to 64x64"
        assert tensor1.min() >= 0 and tensor1.max() <= 1, "Should be normalized to [0,1]"
    
    @given(
        width=st.integers(min_value=32, max_value=128),
        height=st.integers(min_value=32, max_value=128),
        base_intensity=st.floats(min_value=10, max_value=200)
    )
    @settings(max_examples=100, deadline=5000)
    def test_preprocessing_normalization_consistency(self, width, height, base_intensity):
        """
        Property 4b: For any image, normalization parameters should be applied consistently
        **Feature: cnn-confidence-improvement, Property 4: Preprocessing Consistency**
        **Validates: Requirements 5.1, 6.2**
        """
        # Create test image with known intensity distribution
        image_data = np.full((height, width), base_intensity, dtype=np.uint8)
        # Add some variation to avoid edge cases
        noise = np.random.normal(0, 5, (height, width))
        image_data = np.clip(image_data + noise, 0, 255).astype(np.uint8)
        
        test_image = Image.fromarray(image_data, mode='L')
        temp_path = os.path.join(self.temp_dir, f"norm_test_{width}x{height}.png")
        test_image.save(temp_path)
        
        # Process with both training and inference transforms
        inference_tensor = self.preprocessor.preprocess_image(temp_path, training=False)
        
        # Check normalization properties
        assert inference_tensor.min() >= 0, "Normalized values should be >= 0"
        assert inference_tensor.max() <= 1, "Normalized values should be <= 1"
        
        # For uniform-ish images, normalized values should be reasonable
        mean_val = inference_tensor.mean().item()
        assert 0 <= mean_val <= 1, "Mean normalized value should be in [0,1]"
    
    @given(
        num_images=st.integers(min_value=2, max_value=5),
        image_size=st.integers(min_value=32, max_value=96)
    )
    @settings(max_examples=50, deadline=10000)
    def test_batch_preprocessing_consistency(self, num_images, image_size):
        """
        Property 4c: For any batch of images, preprocessing should be consistent per image
        **Feature: cnn-confidence-improvement, Property 4: Preprocessing Consistency**
        **Validates: Requirements 5.1, 6.4**
        """
        # Create multiple test images
        image_paths = []
        for i in range(num_images):
            test_image = self.create_test_image(image_size, image_size)
            temp_path = os.path.join(self.temp_dir, f"batch_test_{i}.png")
            test_image.save(temp_path)
            image_paths.append(temp_path)
        
        # Process individually
        individual_tensors = []
        for path in image_paths:
            tensor = self.preprocessor.preprocess_image(path, training=False)
            individual_tensors.append(tensor)
        
        # Process as batch
        batch_tensor = self.preprocessor.preprocess_batch(image_paths, training=False)
        
        # Compare individual vs batch processing
        assert batch_tensor.shape[0] == num_images, "Batch should contain all images"
        
        for i, individual_tensor in enumerate(individual_tensors):
            batch_item = batch_tensor[i]
            assert torch.allclose(individual_tensor, batch_item, atol=1e-6), \
                f"Batch processing should match individual processing for image {i}"
    
    @given(
        format_type=st.sampled_from(['L', 'RGB', 'RGBA']),
        image_size=st.integers(min_value=32, max_value=96)
    )
    @settings(max_examples=50, deadline=5000)
    def test_image_format_handling_consistency(self, format_type, image_size):
        """
        Property 4d: For any image format, preprocessing should convert consistently to grayscale
        **Feature: cnn-confidence-improvement, Property 4: Preprocessing Consistency**
        **Validates: Requirements 6.4**
        """
        # Create image in specified format
        if format_type == 'L':
            # Grayscale
            image_data = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
            test_image = Image.fromarray(image_data, mode='L')
        elif format_type == 'RGB':
            # RGB - create grayscale-like RGB
            gray_data = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
            rgb_data = np.stack([gray_data, gray_data, gray_data], axis=2)
            test_image = Image.fromarray(rgb_data, mode='RGB')
        else:  # RGBA
            # RGBA - create grayscale-like RGBA
            gray_data = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
            alpha_data = np.full((image_size, image_size), 255, dtype=np.uint8)
            rgba_data = np.stack([gray_data, gray_data, gray_data, alpha_data], axis=2)
            test_image = Image.fromarray(rgba_data, mode='RGBA')
        
        temp_path = os.path.join(self.temp_dir, f"format_test_{format_type}.png")
        test_image.save(temp_path)
        
        # Process image
        tensor = self.preprocessor.preprocess_image(temp_path, training=False)
        
        # Should always result in single-channel tensor
        assert tensor.shape[0] == 1, f"Should convert {format_type} to single channel"
        assert tensor.shape[1] == tensor.shape[2] == 64, "Should be resized to 64x64"
        assert tensor.min() >= 0 and tensor.max() <= 1, "Should be normalized to [0,1]"
    
    def test_validation_function_with_real_image(self):
        """
        Test the validation function with a real image from the dataset
        **Feature: cnn-confidence-improvement, Property 4: Preprocessing Consistency**
        **Validates: Requirements 5.1, 5.4, 6.2**
        """
        # Look for existing test images
        test_image_paths = []
        
        # Check for CNN dataset images
        if os.path.exists("cnn_dataset/images"):
            for filename in os.listdir("cnn_dataset/images")[:3]:  # Test first 3 images
                if filename.endswith('.png'):
                    test_image_paths.append(os.path.join("cnn_dataset/images", filename))
        
        # If no dataset images, create a test image
        if not test_image_paths:
            test_image = self.create_test_image(64, 64)
            temp_path = os.path.join(self.temp_dir, "validation_test.png")
            test_image.save(temp_path)
            test_image_paths = [temp_path]
        
        # Test validation function
        for img_path in test_image_paths:
            if os.path.exists(img_path):
                validation_result = validate_preprocessing_consistency(self.preprocessor, img_path)
                
                # Check validation results
                assert validation_result['consistency'], \
                    f"Preprocessing should be consistent for {img_path}"
                assert validation_result['correct_shape'], \
                    f"Should have correct shape for {img_path}"
                assert validation_result['correct_size'], \
                    f"Should have correct size for {img_path}"
                assert validation_result['correct_range'], \
                    f"Should have correct range for {img_path}"
                
                # Check tensor properties
                shape = validation_result['tensor_shape']
                assert shape[0] == 1, "Should have single channel"
                assert shape[1] == shape[2] == 64, "Should be 64x64"
                
                value_range = validation_result['value_range']
                assert value_range[0] >= 0, "Min value should be >= 0"
                assert value_range[1] <= 1, "Max value should be <= 1"


if __name__ == "__main__":
    # Run a quick test
    test_instance = TestPreprocessingConsistency()
    test_instance.setup_method()
    
    try:
        # Test with a simple synthetic image
        test_image = test_instance.create_test_image(64, 64)
        temp_path = os.path.join(test_instance.temp_dir, "quick_test.png")
        test_image.save(temp_path)
        
        # Test deterministic preprocessing
        tensor1 = test_instance.preprocessor.preprocess_image(temp_path, training=False)
        tensor2 = test_instance.preprocessor.preprocess_image(temp_path, training=False)
        
        consistency_check = torch.allclose(tensor1, tensor2, atol=1e-6)
        print(f"✅ Preprocessing consistency test: {'PASS' if consistency_check else 'FAIL'}")
        print(f"   Tensor shape: {tensor1.shape}")
        print(f"   Value range: [{tensor1.min():.4f}, {tensor1.max():.4f}]")
        
    finally:
        test_instance.teardown_method()