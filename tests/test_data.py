""" Tests for the data pipeline """

# pylint: disable=W0621, W0613
import os
from pathlib import Path
import shutil
from dotenv import load_dotenv
import pytest

from person_image_segmentation.config import (
    RAW_DATA_DIR,
    SPLIT_DATA_DIR,
    TRANSFORM_DATA_DIR,
    LABELS_DATA_DIR,
    TRAIN_SIZE,
    VAL_SIZE,
    DATASET_LINK
)
from person_image_segmentation.utils.dataset_utils import (
    download_dataset,
    split_dataset,
    transform_masks,
    generate_labels,
    complete_data_folder
)

load_dotenv()

# Main folders
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
TEST_DATA_DIR = Path(os.getenv('PATH_TO_DATA_FOLDER').replace('data', 'test_data'))

# Folders for functions
RAW_DATA_DIR = Path(str(RAW_DATA_DIR).replace('data', 'test_data'))
SPLIT_DATA_DIR = Path(str(SPLIT_DATA_DIR).replace('data', 'test_data'))
TRANSFORM_DATA_DIR = Path(str(TRANSFORM_DATA_DIR).replace('data', 'test_data'))
LABELS_DATA_DIR = Path(str(LABELS_DATA_DIR).replace('data', 'test_data'))

@pytest.fixture(scope = "module")
def run_data_pipeline():
    """ Runs de data pipelin to be tested """
    # Run the pipeline script
    print("Running the data pipeline script...")

    # Download raw data
    download_dataset(
        dataset_link = DATASET_LINK,
        data_dir = RAW_DATA_DIR
    )

    # Split data
    split_dataset(
        train_size = TRAIN_SIZE, val_size = VAL_SIZE,
        data_dir = RAW_DATA_DIR, split_dir = SPLIT_DATA_DIR
    )

    # Transform masks
    transform_masks(
        split_dir = SPLIT_DATA_DIR,
        transform_dir = TRANSFORM_DATA_DIR
    )

    # Create labels
    generate_labels(
        transform_dir = TRANSFORM_DATA_DIR, labels_dir = LABELS_DATA_DIR,
        split_dir = SPLIT_DATA_DIR
    )


    # Complete data folder
    config_names = ["config_hyps.yaml", "config_yolos.yaml"]
    src_folder = REPO_PATH / "models/configs"
    dst_folder = TEST_DATA_DIR
    complete_data_folder(
        config_names = config_names,
        src_folder = src_folder,
        dst_folder = dst_folder
    )

    print("Data pipeline script completed successfully.")

    yield

def test_data_pipeline_and_structure(run_data_pipeline):
    """Tests the data pipeline and its structure"""
    # Check if the data folder exists
    assert TEST_DATA_DIR.exists()

    subfolders = [
        TEST_DATA_DIR / "raw/dataset_person-yolos/data",
        TEST_DATA_DIR / "raw/dataset_person-yolos/test",
        TEST_DATA_DIR / "interim/splitted/images/train",
        TEST_DATA_DIR / "interim/splitted/images/val",
        TEST_DATA_DIR / "interim/splitted/images/test",
        TEST_DATA_DIR / "interim/splitted/masks/train",
        TEST_DATA_DIR / "interim/splitted/masks/val",
        TEST_DATA_DIR / "interim/splitted/masks/test",
        TEST_DATA_DIR / "interim/transformed/images/train",
        TEST_DATA_DIR / "interim/transformed/images/val",
        TEST_DATA_DIR / "interim/transformed/images/test",
        TEST_DATA_DIR / "interim/transformed/masks/train",
        TEST_DATA_DIR / "interim/transformed/masks/val",
        TEST_DATA_DIR / "interim/transformed/masks/test",
        TEST_DATA_DIR / "processed/images/train",
        TEST_DATA_DIR / "processed/images/val",
        TEST_DATA_DIR / "processed/images/test",
        TEST_DATA_DIR / "processed/labels/train",
        TEST_DATA_DIR / "processed/labels/val",
        TEST_DATA_DIR / "processed/labels/test",
    ]

    for subfolder in subfolders:
        assert subfolder.exists()

    # Remove the test_data folder after the test
    shutil.rmtree(TEST_DATA_DIR)
