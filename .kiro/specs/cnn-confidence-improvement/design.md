# Design Document

## Overview

This design addresses the critical CNN confidence score issues in the Ghost Hunter marine vessel detection system. The primary problems are severe class imbalance (381 ship vs 160 sea patches), insufficient training data, and preprocessing inconsistencies. The solution involves data rebalancing, architecture improvements, enhanced training procedures, and robust evaluation metrics.

## Architecture

### Enhanced Data Pipeline
```
Raw SAR Data → Balanced Dataset Generator → Augmented Training Set → Improved CNN → Calibrated Inference
```

The architecture separates concerns into distinct components:
- **Dataset Balancer**: Ensures equal class distribution
- **Quality Validator**: Verifies data integrity and completeness  
- **Enhanced CNN**: Improved architecture with better regularization
- **Calibrated Inference**: Consistent preprocessing and confidence calibration

### Training Pipeline Flow
```
1. Dataset Analysis → 2. Class Balancing → 3. Quality Validation → 4. Augmentation → 5. Training → 6. Evaluation
```

## Components and Interfaces

### DatasetBalancer Class
```python
class DatasetBalancer:
    def analyze_distribution(self, labels_json: str) -> Dict[str, int]
    def generate_additional_samples(self, target_count: int, sample_type: str) -> List[str]
    def validate_balance(self, ship_count: int, sea_count: int) -> bool
```

### EnhancedCNN Class  
```python
class EnhancedSARCNN(nn.Module):
    def __init__(self, dropout_rate: float = 0.3, use_batch_norm: bool = True)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### TrainingManager Class
```python
class TrainingManager:
    def setup_data_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]
    def train_with_early_stopping(self, patience: int = 5) -> Dict[str, float]
    def evaluate_model(self) -> Dict[str, float]
```

### InferenceCalibrator Class
```python
class InferenceCalibrator:
    def calibrate_confidence(self, raw_score: float) -> float
    def preprocess_image(self, image_path: str) -> torch.Tensor
    def verify_vessel(self, vessel_data: Dict) -> Dict
```

## Data Models

### Enhanced Dataset Structure
```json
{
  "image": "ship_001.png",
  "label": 1,
  "confidence_weight": 1.0,
  "augmentation_source": "original|rotation|flip|intensity",
  "quality_score": 0.95,
  "vessel_metadata": {
    "area_pixels": 150,
    "mean_intensity": 0.75,
    "contrast_ratio": 2.3
  }
}
```

### Training Configuration
```python
@dataclass
class TrainingConfig:
    img_size: int = 64
    batch_size: int = 32
    learning_rate: float = 0.0001
    epochs: int = 50
    patience: int = 10
    min_samples_per_class: int = 500
    target_accuracy: float = 0.85
    confidence_threshold: float = 0.7
```

### Evaluation Metrics
```python
@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    confidence_distribution: Dict[str, List[float]]
    confusion_matrix: np.ndarray
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Balanced Dataset Generation
*For any* dataset generation process, the number of ship samples and sea samples should differ by no more than 10% of the total dataset size.
**Validates: Requirements 1.1, 1.4**

### Property 2: Training Data Quality Assurance  
*For any* training image in the dataset, the image file should exist, be readable, and have valid dimensions matching the expected input size.
**Validates: Requirements 2.3, 6.1**

### Property 3: Model Performance Threshold
*For any* completed training process, the final validation accuracy should be at least 85% and average confidence scores for positive samples should exceed 0.7.
**Validates: Requirements 3.1, 3.2**

### Property 4: Preprocessing Consistency
*For any* image processed during inference, the preprocessing steps should be identical to those used during training, including normalization parameters and resizing methods.
**Validates: Requirements 5.1, 5.4, 6.2**

### Property 5: Confidence Score Calibration
*For any* model prediction, the output confidence score should be between 0 and 1, and scores above the threshold should correlate with actual vessel presence.
**Validates: Requirements 5.5, 3.2**

### Property 6: Data Augmentation Preservation
*For any* augmented training sample, the class label should remain unchanged and the augmentation should preserve vessel characteristics while adding realistic variations.
**Validates: Requirements 2.5, 6.4**

## Error Handling

### Dataset Issues
- **Missing Images**: Log warnings and exclude from training, don't crash
- **Corrupted Files**: Skip with detailed error logging
- **Class Imbalance**: Automatically generate additional samples or warn user
- **Insufficient Data**: Require minimum thresholds before training

### Training Failures  
- **Convergence Issues**: Implement learning rate scheduling and early stopping
- **Memory Errors**: Reduce batch size automatically and retry
- **Model Divergence**: Reset to best checkpoint and adjust hyperparameters

### Inference Problems
- **Model Loading**: Graceful fallback with clear error messages
- **Preprocessing Errors**: Handle various image formats and sizes
- **Confidence Calibration**: Provide uncalibrated scores if calibration fails

## Testing Strategy

### Unit Testing
- Test dataset balancing logic with known imbalanced datasets
- Verify preprocessing consistency between training and inference
- Test model architecture initialization and forward pass
- Validate metric calculations with known ground truth

### Property-Based Testing  
- Generate random datasets and verify balance properties hold
- Test preprocessing with various image formats and sizes
- Verify confidence score ranges across different model states
- Test augmentation preserves labels across random transformations

**Dual Testing Approach**: Unit tests verify specific functionality while property tests ensure universal correctness across all inputs. Both are essential for comprehensive validation.

**Property Test Configuration**: Each property test runs minimum 100 iterations using PyTorch's property testing framework. Tests are tagged with format: **Feature: cnn-confidence-improvement, Property {number}: {property_text}**