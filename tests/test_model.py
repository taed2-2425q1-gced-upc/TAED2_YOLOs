import pytest
import yaml
import os
import cv2
import subprocess
import shutil
import time

from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from typing import Optional
from ultralytics import YOLO

from person_image_segmentation.modeling.evaluation import compute_mIoU

load_dotenv()

REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
BEST_WEIGHTS = os.getenv('PATH_TO_BEST_WEIGHTS')
BEST_WEIGHTS_FULL_PATH = str(REPO_PATH / BEST_WEIGHTS) if BEST_WEIGHTS != "None" else "yolov8m-seg.pt"

@pytest.fixture(scope = "module")
def run_prediction_pipeline():
    # Run the pipeline script
    print("Running the prediction pipeline script...")
    start_time = time.time()
    script_path = REPO_PATH / 'person_image_segmentation/modeling/prediction.py'
    subprocess.run(['python', str(script_path), '--max_predictions', '10', '--test'], check = True)
    print("Prediction pipeline script completed successfully.")
    end_time = time.time()

    duration = end_time - start_time
    yield duration

@pytest.fixture(scope = "module")
def run_evaluation_pipeline():
    # Compute the mIoU on the predicted samples
    PREDS_PATH = REPO_PATH / "predictions"
    folder_path = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(PREDS_PATH)
    file_names = [str(folder_path / file) for file in file_names if os.path.isfile(str(folder_path / file))]
    mIoU = compute_mIoU(file_names, PREDS_PATH)

    yield mIoU

def test_model_inference_speed(run_prediction_pipeline):
    duration = run_prediction_pipeline
    print("Prediction pipeline duration is: ", duration), " seconds."
    assert duration < 10 # Because we are testing with 10 images

def test_model_performance(run_evaluation_pipeline):
    mIoU = run_evaluation_pipeline
    print("mIoU is: ", mIoU)
    assert mIoU > 0.85