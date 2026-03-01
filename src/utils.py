import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.config import *

def plot_training_history(history, save_path=None):
    """
    Plots training and validation accuracy and loss.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_sample_predictions(model, test_gen, num_samples=10, save_path=None):
    """
    Plots a grid of test images along with true/predicted labels and confidence.
    """
    class_labels = list(test_gen.class_indices.keys())
    
    # Get a batch
    x_batch, y_batch = next(test_gen)
    predictions = model.predict(x_batch)
    
    num_samples = min(num_samples, len(x_batch))
    
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        # Denormalize image if needed; generator sends 0-1
        plt.imshow(x_batch[i])
        plt.axis('off')
        
        true_label = class_labels[np.argmax(y_batch[i])]
        pred_label = class_labels[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100
        
        color = 'blue' if true_label == pred_label else 'red'
        
        plt.title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%", color=color)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Sample predictions saved to {save_path}")

    plt.show()
