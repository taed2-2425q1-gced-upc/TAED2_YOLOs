""" Tests for the model """
# pylint: disable=redefined-outer-name

import os
import time
from unittest import mock
import pytest
import torch

from ultralytics import YOLO

from person_image_segmentation.modeling.evaluation import compute_miou # pylint: disable=E0401
from person_image_segmentation.utils.modeling_utils import generate_predictions
from person_image_segmentation.config import REPO_PATH, DATA_DIR, PATH_TO_BEST_WEIGHTS


PREDS_PATH = REPO_PATH / "predictions"
BASE_DATA_PATH = DATA_DIR
BEST_WEIGHTS_FULL_PATH = (
    str(REPO_PATH / PATH_TO_BEST_WEIGHTS) if PATH_TO_BEST_WEIGHTS != "None" else "yolov8m-seg.pt"
)
MAX_PREDICTIONS = 10

@pytest.fixture(scope = "module")
def run_prediction_pipeline():
    """ Runs the prediction pipeline to be tested """
    # Run the pipeline script
    print("Running the prediction pipeline script...")
    start_time = time.time()

    test_folder = BASE_DATA_PATH / "processed/images/test"
    test_file_names = os.listdir(test_folder)
    test_file_names = [
        str(test_folder / file) for file in test_file_names
        if os.path.isfile(str(test_folder / file))
        ]
    model = YOLO(BEST_WEIGHTS_FULL_PATH)

    generate_predictions(
        test_filenames = test_file_names,
        predictions_folder = PREDS_PATH,
        model = model,
        max_predictions = MAX_PREDICTIONS
    )

    end_time = time.time()
    print("Prediction pipeline script completed successfully.")

    duration = end_time - start_time
    yield duration

@pytest.fixture(scope = "module")
def run_evaluation_pipeline():
    """ Runs the evaluation pipeline to be tested """
    # Compute the mIoU on the predicted samples
    folder_path = BASE_DATA_PATH / "processed/images/test"
    test_file_names = os.listdir(PREDS_PATH)
    test_file_names = [
        str(folder_path / file)
        for file in test_file_names
        if os.path.isfile(str(folder_path / file))
    ]
    miou = compute_miou(test_file_names, PREDS_PATH)

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
    assert miou > 0.825

def test_generate_predictions_raises_exception_on_image_processing_error():
    """Test to check if an exception is raised for image processing errors."""
    mock_model = mock.Mock()
    mock_result = mock.Mock()
    mock_result.masks.data = torch.tensor([[0, 0], [0, 1]])  # Simulate masks data
    mock_model.return_value = [mock_result]

    # Mock the cv2.imread to simulate an error when reading an image
    with mock.patch("cv2.imread", side_effect=Exception("Error reading image")):
        test_filenames = ["path/to/image1.jpg"]

        # Check that the exception is raised with the correct message
        with pytest.raises(Exception, match="Error processing"):
            generate_predictions(
                test_filenames = test_filenames,
                predictions_folder = REPO_PATH / "predictions",
                model = mock_model,
                max_predictions = MAX_PREDICTIONS)

def test_compute_mio_for_wrong_file():
    """Test to check if an exception is raised for wrong file."""
    test_filenames = ["path/to/image1.jpg"]

    # Check that the exception is raised with the correct message
    with pytest.raises(Exception, match="Could not compute mIoU for image"):
        compute_miou(test_filenames, PREDS_PATH)
