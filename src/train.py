import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import keras.saving

from data_loader import load_data
from model import build_cnn

model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(model_dir, exist_ok=True)

X_train, y_train, X_val, y_val, X_test, y_test = load_data()

model = build_cnn()

# Compile the model with optimizer, loss function, and evaluation metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss= 'categorical_crossentropy',
    metrics=['accuracy']
)
# Early stopping to prevent overfitting by stopping training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Save the best model based on validation loss
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "best_model.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Reduce learning rate when validation loss stops improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# Train the model with training data and validate using validation set
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint,reduce_lr],
    verbose=1
)



keras.saving.save_model(model, os.path.join(model_dir, "cnn_cifar10.keras"))

print("Model training completed and saved successfully!")



