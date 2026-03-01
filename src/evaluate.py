import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import *
from src.data_loader import get_data_generators

def evaluate_model(model_path, batch_size=BATCH_SIZE):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    _, _, test_gen = get_data_generators(batch_size=batch_size, augment=False)
    
    print("Evaluating on test set...")
    # Evaluate basic metrics
    results = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    # Generate predictions for detailed metrics
    test_gen.reset() # Important for ordered predictions
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    # Ensure y_true format aligns with predictions
    # This assumes dataset returns integer class indices
    
    print("\nClassification Report:")
    target_names = list(test_gen.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {plots_dir}/confusion_matrix.png")

    # AUC-ROC (binary or multi-class handled via macro/ovr if needed)
    try:
        # For Binary
        if NUM_CLASSES == 2:
            auc = roc_auc_score(y_true, predictions[:, 1])
            fpr, tpr, _ = roc_curve(y_true, predictions[:, 1])
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
            print(f"AUC: {auc:.4f}, ROC curve saved.")
        else:
            # Multi-class
            auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=NUM_CLASSES), predictions, multi_class='ovr')
            print(f"Multi-class AUC-ROC: {auc:.4f}")
    except Exception as e:
        print(f"Could not compute AUC: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .keras model file')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.batch_size)
