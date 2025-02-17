# Retinal Disease Classification Using Neural Networks

## Overview
This project involves training deep learning models for the classification of eye diseases using image data. It consists of three main scripts:
- **Image Preparation**: Prepares and balances the dataset.
- **Neural Network (NN) Model**: Implements a basic neural network for classification.
- **Convolutional Neural Network (CNN) Model**: Implements a CNN-based approach for better image feature extraction.

---

## Project Structure

## 1. Image Preparation

### Description
- This script preprocesses images to enhance quality and balance the dataset.
- Key steps:
  1. Crop black edges from images.
  2. Convert images to grayscale.
  3. Resize images to a consistent size (100x100).
  4. Normalize pixel values.
  5. Balance the dataset using augmentation techniques.
## 2. Neural Network (NN) Model

### Description
- Implements a basic neural network for classification.
- Challenges observed:
  1. Overfitting after ~10-15 epochs.
  2. High validation accuracy fluctuations.
  3. High loss values.
  4. Low classification accuracy.
 
## 3. Convolutional Neural Network (CNN) Model

### Description
- Implements a CNN with improvements such as:
  1. Early stopping to prevent overfitting.
  2. Cross-validation for model evaluation.
  3. Uses Conv2D, BatchNormalization, MaxPooling2D, and Dropout layers for better feature extraction.

## Key Libraries:
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Results & Evaluation
The CNN model generally performs better than the NN model.
Model evaluation includes accuracy, precision, recall, F1-score, and confusion matrices.
Further improvements could include hyperparameter tuning, deeper architectures, or transfer learning.
