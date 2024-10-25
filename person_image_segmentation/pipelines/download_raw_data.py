"""
Person Image Segmentation Pipeline.

This script sets up the environment for downloading a dataset for person image segmentation.
It loads environment variables, configures the dataset directory, and downloads the dataset
from a specified source (Kaggle). The script supports a test mode where an alternative data
directory is used.

Usage:
    python complete_data_folder.py [--test]

Args:
    --test: Run the pipeline in test mode, which uses a different data directory for 
    testing purposes.

Environment Variables:
    KAGGLE_USERNAME: The Kaggle username used for authentication.
    KAGGLE_KEY: The Kaggle API key used for authentication.

"""

import argparse
import os

from pathlib import Path
from dotenv import load_dotenv

from person_image_segmentation.config import RAW_DATA_DIR, DATASET_LINK, KAGGLE_KEY, KAGGLE_USERNAME
from person_image_segmentation.utils.dataset_utils import download_dataset

# Load environment variables from a .env file
load_dotenv()
os.environ['KAGGLE_USERNAME'] = KAGGLE_KEY
os.environ['KAGGLE_KEY'] = KAGGLE_USERNAME


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))

    # Download the dataset
    download_dataset(
        dataset_link = DATASET_LINK,
        data_dir = RAW_DATA_DIR
    )
