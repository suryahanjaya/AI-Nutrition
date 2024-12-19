# FIT 

This repository contains a Jupyter Notebook that implements a food classification model using TensorFlow and Keras. The model is built on the MobileNetV2 architecture and is trained to classify images of various food items.

## Table of Contents
- Required Libraries
- Dataset Preparation
- Model Building and Training
- Evaluation
- Confusion Matrix
- Usage
- License

## Required Libraries
To run this notebook, you will need the following libraries:

```python
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
```

## Dataset Preparation
The dataset is organized into three directories: **Train**, **Valid**, and **Test**, each containing subdirectories for each food class. The model automatically detects the number of classes and prepares the dataset for training, validation, and testing.

### Dataset Structure

```
Dataset/
├── Train/
│   ├── Baked Potato/
│   ├── Burger/
│   ├── Crispy Chicken/
│   ├── Donut/
│   ├── Fries/
│   ├── Hot Dog/
│   ├── Pizza/
│   ├── Sandwich/
│   ├── Taco/
│   └── Taquito/
├── Valid/
│   ├── Baked Potato/
│   ├── Burger/
│   ├── Crispy Chicken/
│   ├── Donut/
│   ├── Fries/
│   ├── Hot Dog/
│   ├── Pizza/
│   ├── Sandwich/
│   ├── Taco/
│   └── Taquito/
└── Test/
    ├── Baked Potato/
    ├── Burger/
    ├── Crispy Chicken/
    ├── Donut/
    ├── Fries/
    ├── Hot Dog/
    ├── Pizza/
    ├── Sandwich/
    ├── Taco/
    └── Taquito/
```

## Model Building and Training
The model is built using the **MobileNetV2** architecture, which is suitable for mobile and edge devices due to its efficiency. The model is compiled with the **Adam** optimizer and trained using categorical cross-entropy loss.

### Training Process
- The model is trained for a specified number of epochs (15 in this case).
- Callbacks such as **ModelCheckpoint** and **EarlyStopping** are used to save the best model and prevent overfitting.

## Evaluation
After training, the model is evaluated on the test dataset, reporting accuracy and loss.

### Test Results
- **Test Accuracy:** 79.60%
- **Test Loss:** 0.6468

## Confusion Matrix
A confusion matrix is generated to visualize the performance of the model on the test dataset. This helps in understanding which classes are being confused by the model.

## Usage
To use this model, follow these steps:
1. Clone the repository.
2. Install the required libraries.
3. Prepare your dataset in the specified structure.
4. Run the Jupyter Notebook to train and evaluate the model.
