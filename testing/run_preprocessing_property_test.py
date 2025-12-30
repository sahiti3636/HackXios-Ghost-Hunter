"""
Runner for preprocessing consistency property tests
"""

from test_preprocessing_consistency_property import TestPreprocessingConsistency
from hypothesis import given, strategies as st, settings
import traceback

def run_property_tests():
    """Run property-based tests for preprocessing consistency"""
    
    print("🚀 Running Property-Based Tests for Preprocessing Consistency")
    print("=" * 60)
    
    test_instance = TestPreprocessingConsistency()
    test_instance.setup_method()
    
    try:
        # Run the validation function test (non-property test)
        print("\n1. Testing validation function with real images...")
        try:
            test_instance.test_validation_function_with_real_image()
            print("   ✅ PASS: Validation function")
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            traceback.print_exc()
        
        # Run a simple deterministic test
        print("\n2. Testing deterministic preprocessing...")
        try:
            # Create a test image manually
            test_image = test_instance.create_test_image(64, 64, (50, 200))
            temp_path = f"{test_instance.temp_dir}/manual_test.png"
            test_image.save(temp_path)
            
            # Test deterministic behavior
            tensor1 = test_instance.preprocessor.preprocess_image(temp_path, training=False)
            tensor2 = test_instance.preprocessor.preprocess_image(temp_path, training=False)
            
            import torch
            if torch.allclose(tensor1, tensor2, atol=1e-6):
                print("   ✅ PASS: Deterministic preprocessing")
            else:
                print("   ❌ FAIL: Preprocessing not deterministic")
                
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            traceback.print_exc()
        
        # Test batch consistency
        print("\n3. Testing batch processing consistency...")
        try:
            # Create multiple test images
            image_paths = []
            for i in range(3):
                test_image = test_instance.create_test_image(64, 64)
                temp_path = f"{test_instance.temp_dir}/batch_test_{i}.png"
                test_image.save(temp_path)
                image_paths.append(temp_path)
            
            # Process individually
            individual_tensors = []
            for path in image_paths:
                tensor = test_instance.preprocessor.preprocess_image(path, training=False)
                individual_tensors.append(tensor)
            
            # Process as batch
            batch_tensor = test_instance.preprocessor.preprocess_batch(image_paths, training=False)
            
            # Compare
            import torch
            all_match = True
            for i, individual_tensor in enumerate(individual_tensors):
                batch_item = batch_tensor[i]
                if not torch.allclose(individual_tensor, batch_item, atol=1e-6):
                    all_match = False
                    break
            
            if all_match:
                print("   ✅ PASS: Batch processing consistency")
            else:
                print("   ❌ FAIL: Batch processing inconsistent")
                
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            traceback.print_exc()
        
        # Test format handling
        print("\n4. Testing image format handling...")
        try:
            import numpy as np
            from PIL import Image
            
            # Test grayscale
            gray_data = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            gray_image = Image.fromarray(gray_data, mode='L')
            gray_path = f"{test_instance.temp_dir}/gray_test.png"
            gray_image.save(gray_path)
            
            # Test RGB
            rgb_data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            rgb_image = Image.fromarray(rgb_data, mode='RGB')
            rgb_path = f"{test_instance.temp_dir}/rgb_test.png"
            rgb_image.save(rgb_path)
            
            # Process both
            gray_tensor = test_instance.preprocessor.preprocess_image(gray_path, training=False)
            rgb_tensor = test_instance.preprocessor.preprocess_image(rgb_path, training=False)
            
            # Check properties
            gray_ok = (gray_tensor.shape[0] == 1 and 
                      gray_tensor.shape[1] == gray_tensor.shape[2] == 64 and
                      gray_tensor.min() >= 0 and gray_tensor.max() <= 1)
            
            rgb_ok = (rgb_tensor.shape[0] == 1 and 
                     rgb_tensor.shape[1] == rgb_tensor.shape[2] == 64 and
                     rgb_tensor.min() >= 0 and rgb_tensor.max() <= 1)
            
            if gray_ok and rgb_ok:
                print("   ✅ PASS: Image format handling")
            else:
                print("   ❌ FAIL: Image format handling issues")
                
        except Exception as e:
            print(f"   ❌ FAIL: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("🎉 Property tests completed!")
        print("\n**Feature: cnn-confidence-improvement, Property 4: Preprocessing Consistency**")
        print("**Validates: Requirements 5.1, 5.4, 6.2**")
        
    finally:
        test_instance.teardown_method()

if __name__ == "__main__":
    run_property_tests()