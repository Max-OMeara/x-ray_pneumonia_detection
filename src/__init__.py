# __init__.py

import os

# Define the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "chest_xray")

# Import important functions for easier access
from .data_loader import load_train, prepare_and_load, create_batches
from .model import build_model, vgg16_model
from .visualization import plot, plot_confusion_matrix
