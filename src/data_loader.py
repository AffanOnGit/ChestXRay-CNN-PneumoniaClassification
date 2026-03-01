import os
import tensorflow as tf
from src.config import *

def get_data_generators(batch_size=BATCH_SIZE, target_size=(IMG_HEIGHT, IMG_WIDTH), augment=True):
    """
    Returns train, validation, and test data generators.
    If the Kaggle dataset doesn't have a dedicated validation split, 
    we split it from the train directory here.
    """
    
    # Base normalization (0-1 scaling)
    if augment:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.15 # Use 15% of train data for validation if 'val' dir is too small
        )
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.15
        )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    print("Loading Training Data:")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    print("Loading Validation Data:")
    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("Loading Test Data:")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def get_subsampled_train_generator(train_generator, fraction=1.0):
    """
    Simulates training dataset size impact by subsampling.
    Note: For large scale, tf.data.Dataset filtering is more efficient, 
    but for standard generators we can cap the steps_per_epoch during training.
    """
    pass # Implementation will be managed via steps_per_epoch in train.py

if __name__ == '__main__':
    # Test dataloader execution if dataset is present
    if os.path.exists(TRAIN_DIR):
        train_gen, val_gen, test_gen = get_data_generators()
    else:
        print(f"Data directory {TRAIN_DIR} not found. Please download the dataset.")
