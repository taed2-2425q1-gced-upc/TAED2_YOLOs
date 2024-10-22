"""
Person Image Segmentation Pipeline - Label Generation.

This script generates labels for a person image segmentation dataset. It uses predefined
directories for transformed data, labels, and split data, and allows for running in test mode
to use alternative data directories.

Usage:
    python generate_labels.py [--test]

Args:
    --test: Run the pipeline in test mode, which changes the data directories to use a
            test-specific path.

Functionality:
    - Updates the data directories to point to test directories if the '--test' argument is
      provided.
    - Calls the 'generate_labels' function to generate labels for the dataset based on the
      specified directories.
"""

import argparse
from pathlib import Path

from person_image_segmentation.config import TRANSFORM_DATA_DIR, LABELS_DATA_DIR, SPLIT_DATA_DIR
from person_image_segmentation.utils.dataset_utils import generate_labels


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
        TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))
        LABELS_DATA_DIR = Path(str(LABELS_DATA_DIR).replace('data', 'test_data'))

    # Generate labels
    generate_labels(
        transform_dir = TRANSFORM_DATA_DIR,
        labels_dir = LABELS_DATA_DIR,
        split_dir = SPLIT_DATA_DIR
    )
