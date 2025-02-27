from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds a Convolutional Neural Network (CNN) model for CIFAR-10 classification.
    
    Args:
        input_shape (tuple): Shape of the input images (default is (32, 32, 3)).
        num_classes (int): Number of output classes (default is 10).

    Returns:
        model: A compiled CNN model.
    """
    model = Sequential([
        # Block 1: Convolution, normalization, and activation
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0005), input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Block 2: Increase in filters and regularization
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Block 3: Extraction of more complex features
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Global Average Pooling to reduce dimensionality before fully connected layers
        GlobalAveragePooling2D(),
        
        # Fully connected dense layer with regularization and dropout
        Dense(512, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.5),

        # Output layer with softmax activation for classification
        Dense(num_classes, activation='softmax')
    ])
    
    return model
