import numpy as np
import os

def load_data():
    """
    Loads the preprocessed CIFAR-10 dataset from .npy files.
    Returns the training, validation, and test sets.
    """

    processed_data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    train_dir = os.path.join(processed_data_dir, "train/")
    val_dir = os.path.join(processed_data_dir, "val/")
    test_dir = os.path.join(processed_data_dir, "test/")

    X_train = np.load(os.path.join(train_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(train_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(val_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(val_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(test_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(test_dir, 'y_test.npy'))

    print("Data loaded successfully!")
    print(f"Training shape: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation shape: {X_val.shape}, Labels: {y_val.shape}")
    print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test