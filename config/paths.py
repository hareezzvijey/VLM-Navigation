"""File paths and directories - PURE CONFIG, no filesystem ops"""
import os

# Get the directory where this file is located (config/)
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root
BASE_DIR = os.path.dirname(CONFIG_DIR)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")