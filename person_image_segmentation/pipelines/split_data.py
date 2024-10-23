"""
Person Image Segmentation Pipeline - Data Splitting.

This script splits the dataset into training, validation, and test sets. It uses predefined
directories for raw data and split data, and allows for running in test mode to use alternative
data directories. The script reads the train and validation split sizes from the configuration.

Usage:
    python split_data.py [--test]

Args:
    --test: Run the pipeline in test mode, which changes the data directories to use a
            test-specific path.

Functionality:
    - Updates the data directories to point to test directories if the '--test' argument is
      provided.
    - Calls the 'split_dataset' function to divide the dataset into training, validation, and
      test sets based on the specified sizes.
"""


import argparse

from pathlib import Path

from person_image_segmentation.config import RAW_DATA_DIR, SPLIT_DATA_DIR, TRAIN_SIZE, VAL_SIZE
from person_image_segmentation.utils.dataset_utils import split_dataset


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))
        SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))

    # Split into train, val and test
    split_dataset(
        train_size = TRAIN_SIZE,
        val_size = VAL_SIZE,
        data_dir = RAW_DATA_DIR,
        split_dir = SPLIT_DATA_DIR
    )
