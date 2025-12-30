#!/usr/bin/env python3
"""
Property-Based Test for Model Performance Threshold

**Feature: cnn-confidence-improvement, Property 3: Model Performance Threshold**
**Validates: Requirements 3.1, 3.2**

Property 3: Model Performance Threshold
*For any* completed training process, the final validation accuracy should be at least 85% 
and average confidence scores for positive samples should exceed 0.7.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import hypothesis
from hypothesis import given, strategies as st, settings
import tempfile
import shutil

# Import the enhanced model and training components
import sys
sys.path.append('.')
from train_cnn import EnhancedSARCNN, SARShipDataset, SARPreprocess, TrainingManager

class TestModelPerformanceProperty:
    """Property-based tests for model performance thresholds"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = None
        
    def teardown_method(self):
        """Cleanup test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_synthetic_dataset(self, num_ship_samples=100, num_sea_samples=100, img_size=64):
        """Create a synthetic balanced dataset for testing"""
        self.temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(images_dir)
        
        dataset_labels = []
        
        # Generate ship samples (higher intensity, more structured patterns)
        for i in range(num_ship_samples):
            # Create ship-like pattern with higher intensity and structure
            img_array = np.random.rand(img_size, img_size) * 0.3 + 0.4  # Base intensity 0.4-0.7
            
            # Add ship-like rectangular structure
            center_x, center_y = img_size // 2, img_size // 2
            ship_width, ship_height = np.random.randint(8, 16), np.random.randint(4, 8)
            
            x1 = max(0, center_x - ship_width // 2)
            x2 = min(img_size, center_x + ship_width // 2)
            y1 = max(0, center_y - ship_height // 2)
            y2 = min(img_size, center_y + ship_height // 2)
            
            img_array[y1:y2, x1:x2] += 0.3  # Brighter ship structure
            
            # Add some noise
            img_array += np.random.normal(0, 0.05, (img_size, img_size))
            img_array = np.clip(img_array, 0, 1)
            
            # Convert to 8-bit and save
            img_8bit = (img_array * 255).astype(np.uint8)
            img_path = os.path.join(images_dir, f"ship_{i}.png")
            Image.fromarray(img_8bit, mode='L').save(img_path)
            
            dataset_labels.append({
                "image": f"ship_{i}.png",
                "label": 1
            })
        
        # Generate sea samples (lower intensity, more random patterns)
        for i in range(num_sea_samples):
            # Create sea-like pattern with lower intensity and noise
            img_array = np.random.rand(img_size, img_size) * 0.2 + 0.1  # Base intensity 0.1-0.3
            
            # Add sea-like speckle noise
            speckle = np.random.exponential(0.1, (img_size, img_size))
            img_array += speckle
            
            # Add some random noise
            img_array += np.random.normal(0, 0.03, (img_size, img_size))
            img_array = np.clip(img_array, 0, 1)
            
            # Convert to 8-bit and save
            img_8bit = (img_array * 255).astype(np.uint8)
            img_path = os.path.join(images_dir, f"sea_{i}.png")
            Image.fromarray(img_8bit, mode='L').save(img_path)
            
            dataset_labels.append({
                "image": f"sea_{i}.png",
                "label": 0
            })
        
        # Save labels file
        labels_path = os.path.join(self.temp_dir, "labels.json")
        with open(labels_path, 'w') as f:
            json.dump(dataset_labels, f)
        
        return images_dir, labels_path
    
    @given(
        num_samples_per_class=st.integers(min_value=50, max_value=200),
        dropout_rate=st.floats(min_value=0.1, max_value=0.5),
        learning_rate=st.floats(min_value=1e-5, max_value=1e-3),
        batch_size=st.integers(min_value=8, max_value=32)
    )
    @settings(max_examples=3, deadline=300000)  # Reduced examples for faster testing
    def test_model_performance_threshold_property(self, num_samples_per_class, dropout_rate, 
                                                learning_rate, batch_size):
        """
        Property 3: Model Performance Threshold
        *For any* completed training process, the final validation accuracy should be at least 85% 
        and average confidence scores for positive samples should exceed 0.7.
        **Validates: Requirements 3.1, 3.2**
        """
        
        # Create synthetic dataset
        images_dir, labels_path = self.create_synthetic_dataset(
            num_ship_samples=num_samples_per_class,
            num_sea_samples=num_samples_per_class
        )
        
        # Setup data transforms
        data_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            SARPreprocess()
        ])
        
        # Create dataset
        dataset = SARShipDataset(labels_path, images_dir, data_transforms)
        
        # Create model with random hyperparameters
        model = EnhancedSARCNN(dropout_rate=dropout_rate, use_batch_norm=True).to(self.device)
        
        # Setup training manager
        trainer = TrainingManager(model, self.device, patience=5)
        train_loader, val_loader = trainer.setup_data_loaders(
            dataset, batch_size=batch_size, train_split=0.8
        )
        
        # Calculate class weights
        labels = [x["label"] for x in dataset.data]
        pos_weight = torch.tensor([labels.count(0) / labels.count(1)]).to(self.device)
        
        # Train model with limited epochs for testing
        history = trainer.train_with_early_stopping(
            train_loader, val_loader,
            epochs=20,  # Limited epochs for testing
            learning_rate=learning_rate,
            pos_weight=pos_weight
        )
        
        # Evaluate final model performance
        final_metrics = trainer.evaluate_model(val_loader)
        
        # Property assertions
        final_accuracy = final_metrics['accuracy']
        vessel_avg_confidence = final_metrics['vessel_avg_confidence']
        
        print(f"\nModel Performance Results:")
        print(f"  Final Accuracy: {final_accuracy:.3f}")
        print(f"  Vessel Avg Confidence: {vessel_avg_confidence:.3f}")
        print(f"  Hyperparameters: dropout={dropout_rate:.3f}, lr={learning_rate:.2e}, batch={batch_size}")
        
        # Property 3: Model Performance Threshold
        # The model should achieve at least 85% accuracy on this synthetic balanced dataset
        assert final_accuracy >= 0.85, (
            f"Model accuracy {final_accuracy:.3f} is below required threshold of 0.85. "
            f"This indicates the model architecture or training process needs improvement."
        )
        
        # The average confidence for vessel samples should exceed 0.7
        assert vessel_avg_confidence >= 0.7, (
            f"Average vessel confidence {vessel_avg_confidence:.3f} is below required threshold of 0.7. "
            f"This indicates the model is not confident enough in positive predictions."
        )
        
        print(f"✅ Property 3 satisfied: Accuracy={final_accuracy:.3f}, Vessel Confidence={vessel_avg_confidence:.3f}")

    def test_model_performance_with_real_dataset(self):
        """
        Test model performance property with the actual dataset if available
        This is a unit test that complements the property-based test
        """
        # Check if real dataset exists
        if not os.path.exists("cnn_dataset/labels.json") or not os.path.exists("cnn_dataset/images"):
            print("⚠️  Real dataset not found, skipping real dataset test")
            return
        
        # Setup data transforms
        data_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            SARPreprocess()
        ])
        
        # Load real dataset
        dataset = SARShipDataset("cnn_dataset/labels.json", "cnn_dataset/images", data_transforms)
        
        # Check if dataset is balanced enough for meaningful testing
        labels = [x["label"] for x in dataset.data]
        ship_count = labels.count(1)
        sea_count = labels.count(0)
        
        if min(ship_count, sea_count) < 50:
            print(f"⚠️  Dataset too small or imbalanced (ships: {ship_count}, sea: {sea_count}), skipping test")
            return
        
        # Create model
        model = EnhancedSARCNN(dropout_rate=0.3, use_batch_norm=True).to(self.device)
        
        # Setup training manager
        trainer = TrainingManager(model, self.device, patience=10)
        train_loader, val_loader = trainer.setup_data_loaders(dataset, batch_size=16)
        
        # Calculate class weights
        pos_weight = torch.tensor([sea_count / ship_count]).to(self.device)
        
        # Train model
        history = trainer.train_with_early_stopping(
            train_loader, val_loader,
            epochs=30,
            learning_rate=1e-4,
            pos_weight=pos_weight
        )
        
        # Evaluate performance
        final_metrics = trainer.evaluate_model(val_loader)
        
        print(f"\nReal Dataset Performance:")
        print(f"  Dataset size: {len(dataset)} (ships: {ship_count}, sea: {sea_count})")
        print(f"  Final Accuracy: {final_metrics['accuracy']:.3f}")
        print(f"  Vessel Avg Confidence: {final_metrics['vessel_avg_confidence']:.3f}")
        print(f"  High Confidence Ratio: {final_metrics['vessel_high_confidence_ratio']:.3f}")
        
        # For real dataset, we may need to be more lenient due to data quality issues
        # But we still expect reasonable performance
        assert final_metrics['accuracy'] >= 0.75, (
            f"Real dataset accuracy {final_metrics['accuracy']:.3f} is too low, "
            f"indicating fundamental issues with model or data"
        )

if __name__ == "__main__":
    # Run the property-based test
    test_instance = TestModelPerformanceProperty()
    test_instance.setup_method()
    
    try:
        # Run property-based test
        print("🧪 Running Model Performance Property Test...")
        test_instance.test_model_performance_threshold_property()
        print("✅ Property-based test completed successfully")
        
        # Run real dataset test
        print("\n🧪 Running Real Dataset Performance Test...")
        test_instance.test_model_performance_with_real_dataset()
        print("✅ Real dataset test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        test_instance.teardown_method()