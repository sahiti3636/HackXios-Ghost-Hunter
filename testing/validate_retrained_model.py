#!/usr/bin/env python3
"""
Comprehensive validation of the retrained CNN model.
Tests on held-out validation data and actual pipeline-generated vessel chips.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from train_cnn import EnhancedSARCNN, SARShipDataset
from sar_preprocessing import create_standard_preprocessor

class ModelValidator:
    def __init__(self, model_path="sar_cnn_model.pth", img_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.preprocessor = create_standard_preprocessor(img_size=img_size)
        
        # Load the retrained model
        self.model = EnhancedSARCNN(dropout_rate=0.3, use_batch_norm=True)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        print(f"🔧 Using device: {self.device}")
    
    def create_test_split(self, json_file="cnn_dataset/labels.json", test_ratio=0.2):
        """Create a held-out test set from the dataset"""
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Stratified split to maintain class balance
        ship_samples = [x for x in data if x["label"] == 1]
        sea_samples = [x for x in data if x["label"] == 0]
        
        # Calculate test sizes
        ship_test_size = int(len(ship_samples) * test_ratio)
        sea_test_size = int(len(sea_samples) * test_ratio)
        
        # Random selection for test set
        np.random.seed(42)  # For reproducibility
        ship_test_indices = np.random.choice(len(ship_samples), ship_test_size, replace=False)
        sea_test_indices = np.random.choice(len(sea_samples), sea_test_size, replace=False)
        
        # Create test set
        test_data = []
        test_data.extend([ship_samples[i] for i in ship_test_indices])
        test_data.extend([sea_samples[i] for i in sea_test_indices])
        
        print(f"📊 Test Set Created:")
        print(f"   Ship samples: {ship_test_size}")
        print(f"   Sea samples: {sea_test_size}")
        print(f"   Total test samples: {len(test_data)}")
        
        return test_data
    
    def validate_on_test_set(self, test_data, image_folder="cnn_dataset/images"):
        """Validate model performance on held-out test set"""
        print(f"\n🧪 Testing on Held-Out Validation Set")
        
        # Create test dataset
        test_dataset = TestDataset(test_data, image_folder, self.preprocessor.get_inference_transform())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                
                # Get model predictions
                logits = self.model(images)
                probabilities = torch.sigmoid(logits).squeeze()
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = np.mean(all_predictions == all_labels)
        auc_score = roc_auc_score(all_labels, all_probabilities)
        
        # Confidence analysis
        vessel_probs = all_probabilities[all_labels == 1]
        sea_probs = all_probabilities[all_labels == 0]
        
        vessel_high_conf = np.mean(vessel_probs > 0.7) if len(vessel_probs) > 0 else 0
        vessel_avg_conf = np.mean(vessel_probs) if len(vessel_probs) > 0 else 0
        
        print(f"   ✅ Accuracy: {accuracy:.3f}")
        print(f"   📈 AUC Score: {auc_score:.3f}")
        print(f"   🚢 Vessel Avg Confidence: {vessel_avg_conf:.3f}")
        print(f"   🎯 Vessels >0.7 Confidence: {vessel_high_conf:.3f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(all_labels, all_predictions, 
                                  target_names=['Sea', 'Vessel'], digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print(f"\n🔍 Confusion Matrix:")
        print(f"   True Negatives (Sea): {cm[0,0]}")
        print(f"   False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}")
        print(f"   True Positives (Vessel): {cm[1,1]}")
        
        # Check requirements
        meets_accuracy = accuracy >= 0.85
        meets_confidence = vessel_high_conf >= 0.7
        
        print(f"\n✅ Requirements Check:")
        print(f"   Accuracy ≥ 85%: {'✅ PASS' if meets_accuracy else '❌ FAIL'} ({accuracy:.1%})")
        print(f"   Vessel Confidence >0.7: {'✅ PASS' if meets_confidence else '❌ FAIL'} ({vessel_high_conf:.1%})")
        
        return {
            'accuracy': float(accuracy),
            'auc_score': float(auc_score),
            'vessel_avg_confidence': float(vessel_avg_conf),
            'vessel_high_confidence_ratio': float(vessel_high_conf),
            'meets_requirements': bool(meets_accuracy and meets_confidence),
            'confusion_matrix': cm.tolist(),
            'all_probabilities': [float(x) for x in all_probabilities],
            'all_labels': [float(x) for x in all_labels]
        }
    
    def test_on_pipeline_chips(self, chips_folder="output/chips"):
        """Test model on actual pipeline-generated vessel chips"""
        print(f"\n🛰️  Testing on Pipeline-Generated Vessel Chips")
        
        if not os.path.exists(chips_folder):
            print(f"   ⚠️  Chips folder not found: {chips_folder}")
            print(f"   📝 Skipping pipeline chip validation")
            return None
        
        # Find all vessel chip images
        chip_files = []
        for root, dirs, files in os.walk(chips_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')) and 'vessel' in file.lower():
                    chip_files.append(os.path.join(root, file))
        
        if not chip_files:
            print(f"   ⚠️  No vessel chip files found in {chips_folder}")
            print(f"   📝 Skipping pipeline chip validation")
            return None
        
        print(f"   📁 Found {len(chip_files)} vessel chip files")
        
        # Test each chip
        chip_results = []
        transform = self.preprocessor.get_inference_transform()
        
        with torch.no_grad():
            for chip_path in chip_files:
                try:
                    # Load and preprocess image
                    image = Image.open(chip_path).convert("L")
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    logits = self.model(image_tensor)
                    probability = torch.sigmoid(logits).item()
                    prediction = probability > 0.5
                    
                    chip_results.append({
                        'file': os.path.basename(chip_path),
                        'probability': probability,
                        'prediction': prediction,
                        'high_confidence': probability > 0.7
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing {chip_path}: {e}")
        
        if chip_results:
            # Analyze results
            avg_confidence = np.mean([r['probability'] for r in chip_results])
            high_conf_ratio = np.mean([r['high_confidence'] for r in chip_results])
            vessel_predictions = np.mean([r['prediction'] for r in chip_results])
            
            print(f"   📊 Pipeline Chip Results:")
            print(f"      Average Confidence: {avg_confidence:.3f}")
            print(f"      High Confidence (>0.7): {high_conf_ratio:.3f}")
            print(f"      Predicted as Vessels: {vessel_predictions:.3f}")
            
            # Show individual results for first few chips
            print(f"   🔍 Sample Results:")
            for i, result in enumerate(chip_results[:5]):
                conf_status = "HIGH" if result['high_confidence'] else "LOW"
                print(f"      {result['file']}: {result['probability']:.3f} ({conf_status})")
            
            return {
                'total_chips': len(chip_results),
                'average_confidence': float(avg_confidence),
                'high_confidence_ratio': float(high_conf_ratio),
                'vessel_prediction_ratio': float(vessel_predictions),
                'individual_results': [{
                    'file': r['file'],
                    'probability': float(r['probability']),
                    'prediction': bool(r['prediction']),
                    'high_confidence': bool(r['high_confidence'])
                } for r in chip_results]
            }
        
        return None
    
    def generate_validation_report(self, test_results, chip_results=None):
        """Generate comprehensive validation report"""
        report = {
            'model_path': 'sar_cnn_model.pth',
            'validation_timestamp': str(np.datetime64('now')),
            'test_set_results': test_results,
            'pipeline_chip_results': chip_results,
            'requirements_validation': {
                'minimum_accuracy_85pct': test_results['accuracy'] >= 0.85,
                'vessel_confidence_above_70pct': test_results['vessel_high_confidence_ratio'] >= 0.7,
                'overall_pass': test_results['meets_requirements']
            }
        }
        
        # Save report
        with open('model_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Validation report saved to: model_validation_report.json")
        return report

class TestDataset(Dataset):
    """Dataset class for test data"""
    def __init__(self, test_data, root_dir, transform=None):
        self.test_data = test_data
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        record = self.test_data[idx]
        img_path = os.path.join(self.root_dir, record["image"])
        
        image = Image.open(img_path).convert("L")
        label = torch.tensor(record["label"], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def main():
    """Main validation function"""
    print("🔍 CNN Model Validation Suite")
    print("=" * 50)
    
    # Initialize validator
    validator = ModelValidator()
    
    # Create test split
    test_data = validator.create_test_split()
    
    # Validate on test set
    test_results = validator.validate_on_test_set(test_data)
    
    # Test on pipeline chips
    chip_results = validator.test_on_pipeline_chips()
    
    # Generate report
    report = validator.generate_validation_report(test_results, chip_results)
    
    print(f"\n🎉 Validation Complete!")
    print(f"   Overall Requirements: {'✅ PASS' if report['requirements_validation']['overall_pass'] else '❌ FAIL'}")

if __name__ == "__main__":
    main()