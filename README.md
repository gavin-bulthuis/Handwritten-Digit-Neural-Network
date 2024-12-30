# Handwritten Digit Classification Using a Convolutional Neural Network (CNN)

## Overview
This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The model achieves a test accuracy of **94.97%** and highlights misclassified examples for further analysis.

## Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of 70,000 grayscale images of handwritten digits (0-9) with dimensions 28x28 pixels. The dataset is split into:
- Training set: 60,000 images
- Test set: 10,000 images

Each image is normalized using the mean and standard deviation values of `(0.1307, 0.3081)`.

## Model Architecture
The CNN consists of:
1. **Convolution Layer**: Extracts feature maps using a 3x3 kernel with 20 output channels.
2. **ReLU Activation**: Applies non-linear activation.
3. **Max Pooling Layer**: Reduces the spatial dimensions by a factor of 2.
4. **Dropout**: Prevents overfitting with a probability of 0.5.
5. **Fully Connected Layers**:
   - Layer 1: Reduces the feature space to 128 dimensions.
   - Layer 2: Outputs probabilities for 10 classes.

### Model Summary
- Convolution: 1 input channel → 20 output channels
- Fully Connected 1: 20 x 13 x 13 → 128
- Fully Connected 2: 128 → 10

## Training Details
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Batch Size**: 32
- **Epochs**: 8
- **Dropout Rate**: 0.5
- **Learning Rate**: Default value from PyTorch optimizer settings

## Training and Testing Performance
| Epoch | Train Loss | Train Error | Test Loss | Test Error |
|-------|------------|-------------|-----------|------------|
| 0     | 0.9235     | 0.2030      | 0.4081    | 0.1091     |
| 1     | 0.3644     | 0.1012      | 0.3083    | 0.0863     |
| 2     | 0.3027     | 0.0870      | 0.2695    | 0.0776     |
| 3     | 0.2690     | 0.0780      | 0.2432    | 0.0710     |
| 4     | 0.2439     | 0.0702      | 0.2223    | 0.0655     |
| 5     | 0.2232     | 0.0640      | 0.2045    | 0.0600     |
| 6     | 0.2054     | 0.0586      | 0.1892    | 0.0545     |
| 7     | 0.1899     | 0.0542      | 0.1758    | 0.0503     |

### Final Accuracy
- **Training Accuracy**: 94.58%
- **Test Accuracy**: 94.97%

## Usage
### Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy


