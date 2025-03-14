# RNA 3D Structure Prediction


# Overview
This project aims to predict the 3D structure of RNA molecules using deep learning techniques. RNA plays a crucial role in various biological processes, and accurately predicting its 3D structure can advance RNA-based medicine, including cancer immunotherapies and CRISPR gene editing.

RNA is vital to life’s most essential processes, but predicting its 3D structure remains challenging. This project builds on recent advances, including the deep learning foundation model RibonanzaNet, to take on the challenge of predicting RNA’s full 3D structure. Success in this field can significantly impact RNA-based medicine and deepen our understanding of natural RNA molecules.

If you sat down to complete a puzzle without knowing what it should look like, you’d have to rely on patterns and logic to piece it together. In the same way, predicting Ribonucleic acid (RNA)’s 3D structure involves using only its sequence to figure out how it folds into the structures that define its function.


# Installs & Downloads
Before you continue, there are a few installs you need to intall in either your terminal or your jupyter notebook to run the code
1. Tensorflow
    - pip install --upgrade tensorflow 
    - import tensorflow as tf
    - from tensorflow.keras.models import Sequential
    - from tensorflow.keras.layers import LSTM, Dense
        -  TensorFlow is a machine learning library that helps you build and train models for tasks like image recognition and natural language processing
        - Platform for building and training machine learning models, especially neural networks.

2. PyTorch
    - pip install torch
    - import torch
    - import torch.nn as nn
    - import torch_geometric.nn as pyg_nn
    - from torch_geometric.data import Data
        - open-source machine learning library widely used for deep learning applications. It provides tools for building and training neural networks


3. R
    - pip install -r
        - Allows you to install open requirements txt and install packages inside of it


4. Seaborn
    - pip install seaborn
    - import seaborn as sns
    - Seaborn is a powerful Python library used for  creating informative and attractive data visualizations.


5. Scikit-learn
    - from sklearn.preprocessing import LabelEncoder, StandardScaler
    - from sklearn.metrics import mean_absolute_error, r2_score
    - from sklearn.model_selection import ParameterGrid, train_test_split
    - from sklearn.ensemble import RandomForestRegressor
    - from sklearn.model_selection import KFold


4. Pandas/Python/Path
    - pandas as pd
    - numpy as np
    - import requests
    - import datetime
    - import os
    - from pathlib import Path


5. Matplotlib
    - import matplotlib.pyplot as plt


# Instructions
1. Open the notebook in Jupyter and ecexute cells in order
2. Import your dependencies as normal
3. Load and read path files for each excel file
4. Display all csv files in a dataframe
5. Visulaize the sample submission data and plot the graph
6. show the numeric columns for correlation analysis and create a heatmap
7. Merge the train labels nad train sequences dataframces
8. Clean the newely merged data by filling in all necessary values then display the new data
9. Check for non-numeric values in the coordinate columns

# Creating Targets with the same shape
1. Create a dummy example witht he same inputs as above

# GNN (Graph Neural Network) Model 
Graph Neural Networks (GNNs) have shown great promise in RNA prediction tasks, particularly for understanding RNA structures and interactions due to their ability to model complex relationships and structures inherent in RNA molecules.

1. Convert merged DataFrame to a tensor
2. Define dummy edge_index for this example
3. Create a Data object
4. Define an enhanced GNN model with more layers and dropout. Use EnhancedGNN as the class
5. Split the data into train, validation, and test sets
6. Convert above results to torch tensors
7. Display as dataframe
8. Calculate metrics
9. Save results as a CSV file

# Feature Importance
To analyze feature importance, we’ll use a Random Forest model and calculate the importance of each feature. The feature importance analysis determines how much each input variable (feature) contributes to a model's predictions, helping identify the most influential features for better model interpretation, feature selection, and performance optimization. 

1. Prepare the data using x_1', 'y_1', 'z_1, and resid from train labels
2. Train a Random Forest model
3. Get feature importances
4. Plot feature importances


# Residual Analysis
Analyzing the residuals from the GNN model by using the validation data then plaot the residuals 

# Cross Validation
Implementing k-fold cross-validation

1. Prepare the data using x_1', 'y_1', 'z_1, and resid from train labels
2. Set up k-fold cross-validation
3. Initialize lists to store results
4. Train a Random Forest model
5. Evaluate model
6. Convert results to DataFrame and display
7. Calculate average metrics across all folds


# Monitor Overfitting
1. Ensure the model is in evaluation mode
2. Run the model with no gradient calculation
3. Print results
4. Use the data to Predict on the test set by using model.eval
5. Calculate metrics
6. Plot learning curves


