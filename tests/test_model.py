""" Tests for the model """
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

import pytest
from person_image_segmentation.modeling.evaluation import compute_miou # pylint: disable=E0401

load_dotenv()

REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
PREDS_PATH = REPO_PATH / "predictions"
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
BEST_WEIGHTS = os.getenv('PATH_TO_BEST_WEIGHTS')
BEST_WEIGHTS_FULL_PATH = (
    str(REPO_PATH / BEST_WEIGHTS)
    if BEST_WEIGHTS != "None"
    else "yolov8m-seg.pt"
)


@pytest.fixture(scope = "module")
def run_prediction_pipeline():
    """ Runs the prediction pipeline to be tested """
    # Run the pipeline script
    print("Running the prediction pipeline script...")
    start_time = time.time()
    script_path = REPO_PATH / 'person_image_segmentation/modeling/prediction.py'
    subprocess.run(['python', str(script_path), '--max_predictions', '10'], check = True)
    print("Prediction pipeline script completed successfully.")
    end_time = time.time()

    duration = end_time - start_time
    yield duration

@pytest.fixture(scope = "module")
def run_evaluation_pipeline():
    """ Runs the evaluation pipeline to be tested """
    # Compute the mIoU on the predicted samples
    folder_path = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(PREDS_PATH)
    file_names = [
        str(folder_path / file)
        for file in file_names
        if os.path.isfile(str(folder_path / file))
    ]
    miou = compute_miou(file_names, PREDS_PATH)

    yield miou

def test_model_inference_speed(run_prediction_pipeline):
    """ Tests the speed of the model inference """
    duration = run_prediction_pipeline
    print("Prediction pipeline duration is: ", duration, " seconds.")
    assert duration < 10 # Because we are testing with 10 images

def test_model_performance(run_evaluation_pipeline):
    """ Tests the performance of the model """
    miou = run_evaluation_pipeline
    print("mIoU is: ", miou)
    assert miou > 0.85
