import pytest
import yaml
import os
import cv2
import subprocess
import shutil

from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from typing import Optional

load_dotenv()

REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
TEST_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER').replace('data', 'test_data'))

@pytest.fixture(scope = "module")
def run_data_pipeline():
    # Run the pipeline script
    print("Running the data pipeline script...")
    pipelines_path = REPO_PATH / 'person_image_segmentation/pipelines/'
    script_names = ["download_raw_data.py", "split_data.py", "transform_masks.py", "create_labels.py", "complete_data_folder.py"]
    for script_name in script_names:
        script_path = pipelines_path / script_name
        subprocess.run(['python', str(script_path), '--test'], check = True)
    print("Data pipeline script completed successfully.")

    yield

def test_data_pipeline_and_structure(run_data_pipeline):
    # Check if the data folder exists
    assert TEST_DATA_PATH.exists()

    subfolders = [
        TEST_DATA_PATH / "raw/dataset_person-yolos/data",
        TEST_DATA_PATH / "raw/dataset_person-yolos/test",
        TEST_DATA_PATH / "interim/splitted/images/train",
        TEST_DATA_PATH / "interim/splitted/images/val",
        TEST_DATA_PATH / "interim/splitted/images/test",
        TEST_DATA_PATH / "interim/splitted/masks/train",
        TEST_DATA_PATH / "interim/splitted/masks/val",
        TEST_DATA_PATH / "interim/splitted/masks/test",
        TEST_DATA_PATH / "interim/transformed/images/train",
        TEST_DATA_PATH / "interim/transformed/images/val",
        TEST_DATA_PATH / "interim/transformed/images/test",
        TEST_DATA_PATH / "interim/transformed/masks/train",
        TEST_DATA_PATH / "interim/transformed/masks/val",
        TEST_DATA_PATH / "interim/transformed/masks/test",
        TEST_DATA_PATH / "processed/images/train",
        TEST_DATA_PATH / "processed/images/val",
        TEST_DATA_PATH / "processed/images/test",
        TEST_DATA_PATH / "processed/labels/train",
        TEST_DATA_PATH / "processed/labels/val",
        TEST_DATA_PATH / "processed/labels/test",
    ]

    for subfolder in subfolders:
        assert subfolder.exists()
    
    # Remove the test_data folder after the test
    shutil.rmtree(TEST_DATA_PATH)