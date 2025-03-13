# Improved Traffic Sign Recognition with Hybrid CNN-Tsetlin Machine

This project implements an improved hybrid approach for traffic sign recognition that combines:

1. Adaptive Gaussian thresholding for preprocessing
2. A simplified CNN for feature extraction 
3. Tsetlin Machine for classification of extracted features

## Overview

The hybrid CNN-Tsetlin Machine approach combines the powerful feature extraction capabilities of Convolutional Neural Networks (CNNs) with the interpretability and efficiency of Tsetlin Machines for classification tasks. This implementation focuses on traffic sign recognition as a demonstration of the technique.

## Key Features

- **Adaptive Gaussian Thresholding**: Preprocessing technique that better handles varying lighting conditions across images
- **Simplified CNN Architecture**: Lightweight feature extractor with fewer parameters
- **Tsetlin Machine Classifier**: Applied to the extracted CNN features for final classification

## Dataset

This implementation uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The data should be organized in the following format:

- `train.p`: Training dataset (pickle format)
- `valid.p`: Validation dataset (pickle format)
- `test.p`: Testing dataset (pickle format)
- `signname.csv`: CSV file mapping class IDs to sign descriptions

## Requirements

The following libraries are required:
- numpy
- pandas
- opencv-python (cv2)
- matplotlib
- seaborn
- tensorflow
- scikit-learn
- tmu (Tsetlin Machine Utilities)

## Implementation Details

The Jupyter notebook `Hybrid_Tsetlin_CNN_Model.ipynb` contains the complete implementation with the following sections:

1. Import Libraries
2. Helper Functions for data loading and preprocessing
3. Load and Prepare Data
4. Preprocess Data using Adaptive Gaussian Thresholding
5. Simplified CNN Architecture for Feature Extraction
6. Pre-training the CNN Feature Extractor
7. Extract Features using the Pre-trained CNN
8. Configure and Train the Tsetlin Machine
9. Plotting Training Progress
10. Evaluate on Test Set
11. Visualize Test Predictions
12. Class-wise Accuracy Analysis
13. Conclusion

## Parameter Tuning for Tsetlin Machine

The Tsetlin Machine's performance can be significantly improved by properly tuning the T and s parameters:

### Understanding the Parameters

- **T Parameter (Threshold)**: Controls the threshold for including literals in clauses
  - Higher values = more conservative learning
  - Lower values = more aggressive feature inclusion
  
- **s Parameter (Specificity)**: Controls the balance between Type I and Type II feedback
  - Higher values = higher precision, potentially lower recall
  - Lower values = more aggressive learning, potentially more generalization

### Guidelines for Parameter Selection

1. **Scaling T with Clause Count**:
   ```python
   T = max(15, int(num_clauses * 0.15))  # Scale T with clause count, minimum of 15
   ```
   - For complex problems with many features, use a higher ratio (0.15-0.2)
   - For simpler problems, use a lower ratio (0.1-0.15)

2. **Adjusting s for Learning Behavior**:
   - Start with s = 3.0 (more aggressive learning)
   - For noisy datasets, try higher values (3.5-4.0)
   - For clean datasets with clear patterns, try lower values (2.5-3.0)

3. **Alternative Configurations**:
   - **Conservative Learning**:
     ```python
     T = max(20, int(num_clauses * 0.2))
     s = 2.5
     ```
     Better for datasets with clear discriminative features

   - **Balanced Approach**:
     ```python
     T = max(10, int(num_clauses * 0.1))
     s = 3.9
     ```
     Better for datasets with noise or overlapping classes

4. **Grid Search**:
   - For optimal results, perform a grid search over:
     - T values: [10, 15, 20, 25]
     - s values: [2.5, 3.0, 3.5, 4.0]
   - Monitor validation accuracy to select the best parameters

### Practical Tips

- Increase the number of clauses before fine-tuning T and s
- Lower T values typically require more epochs to converge
- For CNN-extracted features, slightly higher T values often work better
- Adjust clause_drop_p (0.05-0.1) for additional regularization when tuning T and s

## How to Use

1. Ensure all required libraries are installed
2. Download the GTSRB dataset and prepare the pickle files
3. Update the file paths in the notebook to match your local environment
4. Run the notebook cells sequentially

## Advantages of this Approach

1. Maintains the adaptive Gaussian thresholding preprocessing for robust image handling
2. Uses a simplified CNN architecture for efficient feature extraction
3. Leverages Tsetlin Machine's strengths for the final classification
4. Potentially achieves higher accuracy than either approach alone
5. Combines the best aspects of deep learning and interpretable machine learning

## Performance Evaluation

The notebook includes:
- Comparison between hybrid model and pure CNN performance
- Class-wise accuracy analysis to identify strengths and weaknesses
- Visualization of correctly and incorrectly classified samples
- Analysis of best and worst performing traffic sign classes
