import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# For the chest x-ray dataset from Kaggle, assuming standard splits exist or combining them
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model saving path
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Hyperparameters (Default values, will be swept during tuning)
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50

# Regularization parameters
DROPOUT_RATE = 0.3
L2_LAMBDA = 1e-4

# Early stopping
PATIENCE = 10

# Dataset classes
# Dataset classes - dynamically infer based on train directory contents if possible
if os.path.exists(TRAIN_DIR):
    CLASSES = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
else:
    CLASSES = ['NORMAL', 'PNEUMONIA', 'COVID19'] # Fallback
    
NUM_CLASSES = len(CLASSES)
