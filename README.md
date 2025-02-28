# CNN CIFAR-10 Classifier

This repository contains a **Convolutional Neural Network (CNN)** trained to classify images from the **CIFAR-10 dataset** into ten categories. The model was implemented from scratch using **TensorFlow/Keras** and follows a structured workflow including data exploration, preprocessing, training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Model Performance](#model-performance)
- [References](#references)
- [Author](#author)

## Project Overview

This project aims to build a **deep learning model** capable of classifying images from the **CIFAR-10 dataset** into the following categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Dataset

The **CIFAR-10 dataset** consists of **60,000 color images (32x32 pixels) in 10 classes**, with **50,000 images for training and 10,000 for testing**. The dataset is publicly available in TensorFlow and can be loaded using:

```python
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

## Project Structure

cnn-cifar10-classifier/
│── data/
│ ├── processed/
│ ├── test/ # Preprocessed test images
│ ├── train/ # Preprocessed training images
│ ├── val/ # Preprocessed validation images
│── models/
│ ├── best_model.keras # Trained CNN model (ignored in .gitignore)
│── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis
│ ├── 02_preprocessing.ipynb # Data Preprocessing
│ ├── 03_model_training.ipynb # Model Training
│ ├── 04_model_evaluation.ipynb # Model Evaluation
│── results/
│ ├── test_results.json # Classification report and metrics
│── src/
│ ├── data_loader.py # Functions for loading the dataset
│ ├── model.py # CNN model architecture
│ ├── train.py # Training script
│ ├── upload_model.py # Script to upload model to Hugging Face
│── .gitignore
│── README.md
│── requirements.txt

## Jupyter Notebooks

This project follows a structured deep learning workflow, divided into four key steps. Each step has a corresponding Jupyter Notebook:

| Step                             | Notebook                                                         | Description                                                                              |
| -------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **1. Exploratory Data Analysis** | [01_eda.ipynb](notebooks/01_eda.ipynb)                           | Analyzes the dataset distribution, class frequencies, and image samples.                 |
| **2. Data Preprocessing**        | [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb)       | Performs data normalization, augmentation, one-hot encoding and dataset splitting.       |
| **3. Model Training**            | [03_model_training.ipynb](notebooks/03_model_training.ipynb)     | Defines, compiles, and trains the CNN model on CIFAR-10.                                 |
| **4. Model Evaluation**          | [04_model_evaluation.ipynb](notebooks/04_model_evaluation.ipynb) | Evaluates the trained model using accuracy, confusion matrix, and misclassified samples. |

## Installation

To set up the environment, clone the repository and install the required packages:

```python
git clone https://github.com/GustavoGomes7/cnn-cifar10-classifier.git
cd cnn-cifar10-classifier
pip install -r requirements.txt
```

## Training the Model

To train the CNN model from scratch, execute the training script:

```python
python src/train.py
```

## Model Performance

| Metric                      | Value                                                                  |
| --------------------------- | ---------------------------------------------------------------------- |
| **Dataset**                 | CIFAR-10 (10 classes, 60,000 images)                                   |
| **Test Accuracy**           | 89.90%                                                                 |
| **Test Loss**               | 0.5551                                                                 |
| **Best-performing classes** | Automobile (F1-score 0.95), Ship (F1-score 0.94), Truck(F1-score 0.94) |
| **Most challenging class**  | Cat (F1-score 0.80)                                                    |

## References

- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## Author

Developed by Gustavo Gomes

- LinkedIn: https://www.linkedin.com/in/gustavo-gomes-581975333/
