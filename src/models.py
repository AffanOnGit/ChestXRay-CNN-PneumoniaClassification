import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from src.config import *

def build_custom_cnn(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES, 
                     dropout_rate=DROPOUT_RATE, l2_lambda=L2_LAMBDA):
    """
    Builds the proposed custom CNN architecture from scratch based on assignment requirements.
    Architecture: 4 Conv Blocks -> Flatten -> Dense -> Output
    """
    model = models.Sequential(name='Custom_CNN')

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape, 
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate / 2)) # Lighter dropout early on
    
    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    # Block 4
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', 
                           kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_rate))

    # Fully Connected Head
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rate))
    
    # Output Layer
    # Categorical classification if >2 classes, or binary. We'll use softmax for generality.
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def build_pretrained_resnet50(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES, 
                              dropout_rate=DROPOUT_RATE, fine_tune_layers=20):
    """
    Builds a baseline ResNet50 model.
    """
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, 
                                                input_shape=input_shape)
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Fine-tuning: unfreeze the last 'fine_tune_layers' layers if specified
    if fine_tune_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

    model = models.Sequential(name='ResNet50_Baseline')
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def build_pretrained_vgg16(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES, 
                           dropout_rate=DROPOUT_RATE):
    """
    Builds a baseline VGG16 model.
    """
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, 
                                             input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential(name='VGG16_Baseline')
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def compile_model(model, learning_rate=LEARNING_RATE):
    """
    Compiles the Keras model with Adam optimizer and Categorical Crossentropy.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = build_custom_cnn()
    model.summary()
