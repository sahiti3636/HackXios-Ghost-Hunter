# Requirements Document

## Introduction

The Ghost Hunter CNN model is producing consistently low confidence scores for vessel detection, making it unreliable for marine vessel verification. This issue stems from severe class imbalance, insufficient training data, and potential model architecture limitations that need to be addressed to improve the system's effectiveness.

## Glossary

- **CNN**: Convolutional Neural Network used for vessel verification
- **Class_Imbalance**: Unequal distribution of positive (ship) and negative (sea) samples
- **Confidence_Score**: Model output probability between 0-1 indicating vessel presence
- **Training_Dataset**: Collection of labeled SAR image patches for model training
- **SAR_Imagery**: Synthetic Aperture Radar satellite imagery
- **Vessel_Verification**: Process of confirming detected objects are actual vessels

## Requirements

### Requirement 1: Address Severe Class Imbalance

**User Story:** As a marine surveillance analyst, I want balanced training data, so that the CNN model can learn to distinguish vessels from sea clutter effectively.

#### Acceptance Criteria

1. THE Training_Dataset SHALL contain approximately equal numbers of ship and sea patches
2. WHEN generating negative samples, THE System SHALL create at least 300 sea patches to match ship samples
3. THE System SHALL validate class distribution before training begins
4. WHEN class imbalance exceeds 2:1 ratio, THE System SHALL generate additional samples for the minority class
5. THE Dataset_Generator SHALL sample diverse sea regions to avoid overfitting to specific ocean patterns

### Requirement 2: Improve Training Data Quality and Quantity

**User Story:** As a machine learning engineer, I want sufficient high-quality training data, so that the model can generalize well to new vessel detections.

#### Acceptance Criteria

1. THE Training_Dataset SHALL contain minimum 1000 total samples (500 ships, 500 sea)
2. WHEN extracting vessel patches, THE System SHALL ensure vessels are centered and properly cropped
3. THE System SHALL validate that all training images exist and are readable before training
4. WHEN generating sea patches, THE System SHALL avoid areas near detected vessels
5. THE Data_Augmentation SHALL include rotation, flipping, and intensity variations appropriate for SAR imagery

### Requirement 3: Optimize Model Architecture and Training

**User Story:** As a system developer, I want an optimized CNN architecture, so that the model achieves reliable confidence scores above 0.7 for actual vessels.

#### Acceptance Criteria

1. THE Model SHALL achieve minimum 85% validation accuracy during training
2. WHEN training completes, THE Model SHALL demonstrate confidence scores above 0.7 for positive vessel samples
3. THE Training_Process SHALL include early stopping to prevent overfitting
4. THE Model SHALL use appropriate learning rate scheduling for convergence
5. THE Training_Loop SHALL track and report both loss and accuracy metrics per epoch

### Requirement 4: Implement Robust Model Evaluation

**User Story:** As a quality assurance engineer, I want comprehensive model evaluation, so that I can verify the CNN performs reliably on real vessel data.

#### Acceptance Criteria

1. THE Evaluation_Process SHALL test model on held-out validation set separate from training data
2. WHEN evaluating performance, THE System SHALL report precision, recall, and F1-score metrics
3. THE System SHALL generate confusion matrix showing true/false positives and negatives
4. THE Evaluation SHALL include confidence score distribution analysis for both classes
5. THE System SHALL test model performance on actual pipeline-generated vessel chips

### Requirement 5: Fix Inference Pipeline Issues

**User Story:** As a marine surveillance operator, I want consistent inference results, so that vessel verification produces reliable confidence scores in production.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL use identical preprocessing as training pipeline
2. WHEN loading images for inference, THE System SHALL handle missing files gracefully
3. THE Model_Loading SHALL verify model architecture matches training configuration
4. THE Preprocessing SHALL normalize images using same parameters as training
5. THE Inference SHALL output calibrated confidence scores between 0 and 1

### Requirement 6: Enhance Data Preprocessing and Normalization

**User Story:** As a data scientist, I want proper data preprocessing, so that the model receives consistent input format during training and inference.

#### Acceptance Criteria

1. THE Preprocessing SHALL convert all images to consistent grayscale format
2. WHEN normalizing pixel values, THE System SHALL use SAR-appropriate normalization ranges
3. THE Image_Resizing SHALL maintain aspect ratio or use appropriate padding
4. THE Preprocessing SHALL handle different input image formats (PNG, TIFF) consistently
5. THE Normalization SHALL be applied identically during training and inference phases