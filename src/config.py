"""
Configuration and hyperparameters for image sequence classification
"""

# Dataset Configuration
DATASET_NAME = "daniel3303/StoryReasoning"
DATASET_SUBSET_SIZE = 300  # Number of stories to use (for faster training)
TRAIN_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Image Preprocessing
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 64
NUM_CHANNELS = 3

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
NUM_CLASSES = 5  # Positions 1-5

# Model Architecture Defaults
DEFAULT_FILTERS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT = 0.3
DEFAULT_BATCH_NORM = False

# Experiment Configurations
EXPERIMENTS = {
    'dropout': {
        'name': '1. Dropout=0.3',
        'params': {'dropout': 0.3, 'filters': 32, 'kernel': 3, 'batchnorm': False}
    },
    'no_dropout': {
        'name': '2. No Dropout',
        'params': {'dropout': 0.0, 'filters': 32, 'kernel': 3, 'batchnorm': False}
    },
    'kernel_5': {
        'name': '3. Kernel=5',
        'params': {'dropout': 0.3, 'filters': 32, 'kernel': 5, 'batchnorm': False}
    },
    'filters_64': {
        'name': '4. Filters=64',
        'params': {'dropout': 0.3, 'filters': 64, 'kernel': 3, 'batchnorm': False}
    },
    'batchnorm': {
        'name': '5. BatchNorm',
        'params': {'dropout': 0.3, 'filters': 32, 'kernel': 3, 'batchnorm': True}
    }
}

# Results Configuration
RESULTS_DIR = "Results"
RANDOM_BASELINE_ACCURACY = 0.20  # 20% for 5-class classification
