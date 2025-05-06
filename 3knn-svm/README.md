# Water Potability Classification using KNN and SVM

A machine learning implementation that uses K-Nearest Neighbors (KNN) and Support Vector Machines (SVM) algorithms to classify water potability based on water quality metrics.

## Overview

This project demonstrates the application of two popular machine learning classification algorithms to predict whether water samples are safe for consumption:

1. **K-Nearest Neighbors (KNN)** - A non-parametric classification method that classifies data points based on the majority class of their k nearest neighbors
2. **Support Vector Machine (SVM)** - A supervised learning model that finds an optimal hyperplane to separate classes

## Features

- Data loading and preprocessing from CSV format
- Data normalization for improved model performance
- Automatic train/test split (80/20)
- Custom KNN implementation with Euclidean distance metric
- SVM classifier using Gaussian kernel via the linfa library
- Model accuracy evaluation and comparison
- Visualization of both classifiers' decision boundaries
- Detailed prediction output for sample comparison

## Requirements

- Rust (latest stable version)
- The following dependencies (specified in Cargo.toml):
  - ndarray: For numerical computation
  - linfa (with svm, logistic features): For machine learning algorithms
  - plotters: For visualization
  - csv: For data loading
  - rand: For random sampling

## Usage

1. Ensure you have a water quality dataset in CSV format in the `csv/` directory
   - Default dataset: `water_potability.csv`
   - Required format: Quality metrics columns with a final `potability` column (0 = unsafe, 1 = safe)

2. Build and run the project:
   ```bash
   cd 3knn-svm
   cargo build --release
   cargo run --release
   ```

3. The program will:
   - Load and preprocess the dataset
   - Train both KNN and SVM models
   - Output accuracy metrics and sample predictions
   - Generate visualization plots in the `output/` directory

## Output

### Console Output
- Distribution of safe/unsafe water samples in the dataset
- Classification accuracy for both KNN (k=3) and SVM models
- Comparison of predicted vs actual values for sample test data

### Generated Visualizations
- `output/knn_plot.png`: Visualization of KNN classification with sample points
- `output/svm_plot.png`: Visualization of SVM decision boundary

## Implementation Details

- The KNN algorithm is implemented from scratch with custom distance calculation
- The SVM algorithm uses the linfa-svm library with a Gaussian kernel
- Both models are evaluated on the same normalized test data for fair comparison
- Visualization includes color-coded regions for safe/unsafe predictions 