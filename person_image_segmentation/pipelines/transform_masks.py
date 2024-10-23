"""
Person Image Segmentation Pipeline - Mask Transformation.

This script transforms the masks in the dataset by converting them from their original format
to a format suitable for training a segmentation model. It uses predefined directories for split
data and transformed data, and allows for running in test mode to use alternative data directories.

Usage:
    python transform_masks.py [--test]

Args:
    --test: Run the pipeline in test mode, which changes the data directories to use a
            test-specific path.

Functionality:
    - Updates the data directories to point to test directories if the '--test' argument is
      provided.
    - Calls the 'transform_masks' function to transform the masks in the dataset and save them in
      the specified directory.
"""


import argparse
from pathlib import Path

from person_image_segmentation.config import SPLIT_DATA_DIR, TRANSFORM_DATA_DIR
from person_image_segmentation.utils.dataset_utils import transform_masks


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
        TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))

    # Transform masks
    transform_masks(
        split_dir = SPLIT_DATA_DIR,
        transform_dir = TRANSFORM_DATA_DIR
    )
