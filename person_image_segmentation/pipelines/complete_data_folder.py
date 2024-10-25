"""
Complete data folder script.

This script completes the data folder by copying specified
configuration files from the source folder to the destination
folder. It can be run in test mode to use a different data directory.
"""

import argparse
from pathlib import Path
from person_image_segmentation.config import REPO_PATH, DATA_DIR
from person_image_segmentation.utils.dataset_utils import complete_data_folder

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parser for Person Image Segmentation Pipeline")
    parser.add_argument('--test', action='store_true', help="Run the pipeline in test mode")
    args = parser.parse_args()

    if args.test:
        DATA_DIR = Path(str(DATA_DIR).replace('data', 'test_data'))

    config_names = ["config_hyps.yaml", "config_yolos.yaml"]

    src_folder = REPO_PATH / "models/configs"
    dst_folder = DATA_DIR

    complete_data_folder(
        config_names = config_names,
        src_folder = src_folder,
        dst_folder = dst_folder
    )
