import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from src.config import *
from src.data_loader import get_data_generators
from src.models import build_custom_cnn, build_pretrained_resnet50, build_pretrained_vgg16, compile_model
import datetime

def train(model_type='custom', batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
          epochs=EPOCHS, dropout_rate=DROPOUT_RATE, l2_lambda=L2_LAMBDA, patience=PATIENCE):
    
    print(f"--- Starting Training ---")
    print(f"Model: {model_type} | Batch Size: {batch_size} | LR: {learning_rate} | Dropout: {dropout_rate}")

    # Load Data
    train_gen, val_gen, _ = get_data_generators(batch_size=batch_size)

    # Build Model
    if model_type == 'custom':
        model = build_custom_cnn(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
    elif model_type == 'resnet':
        model = build_pretrained_resnet50(dropout_rate=dropout_rate)
    elif model_type == 'vgg':
        model = build_pretrained_vgg16(dropout_rate=dropout_rate)
    else:
        raise ValueError("Invalid model type")

    model = compile_model(model, learning_rate=learning_rate)
    
    # Callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_type}_bs{batch_size}_lr{learning_rate}_do{dropout_rate}.keras")
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    
    log_dir = os.path.join(BASE_DIR, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    return history, model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='custom', choices=['custom', 'resnet', 'vgg'])
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE)
    args = parser.parse_args()
    
    try:
        train(model_type=args.model, batch_size=args.batch_size, 
              learning_rate=args.lr, epochs=args.epochs, dropout_rate=args.dropout)
    except Exception as e:
        print(f"Training failed or dataset missing: {e}")
