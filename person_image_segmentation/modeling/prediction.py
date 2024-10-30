""" Prediction script for the model """
# General imports
import os
import argparse
from pathlib import Path

from ultralytics import YOLO # pylint: disable=E0401
from codecarbon import EmissionsTracker # pylint: disable=E0401

from person_image_segmentation.utils.modeling_utils import generate_predictions
from person_image_segmentation.config import REPO_PATH, DATA_DIR, PATH_TO_BEST_WEIGHTS

# Declare the base data path
BASE_DATA_PATH = DATA_DIR
BEST_WEIGHTS_FULL_PATH = (
    str(REPO_PATH / PATH_TO_BEST_WEIGHTS)
    if PATH_TO_BEST_WEIGHTS != "None"
    else "yolov8m-seg.pt"
)


if __name__ == "__main__":

    OUTPUT_FILE = "emissions_inference.csv"
    METRICS_FOLDER = str(REPO_PATH / "metrics")

    # Argument parser for max number of predictions
    parser = argparse.ArgumentParser(description="Generate predictions for test images.")
    parser.add_argument(
        '--max_predictions',
        type=int,
        default=10,
        help='Maximum number of predictions to generate'
        )
    args = parser.parse_args()
    max_predictions = args.max_predictions

    # Load the YOLO model
    model = YOLO(BEST_WEIGHTS_FULL_PATH)

    # Load the test images
    test_folder = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(test_folder)
    file_names = [
        str(test_folder / file) for file in file_names
        if os.path.isfile(str(test_folder / file))
        ]

    with EmissionsTracker(output_dir=METRICS_FOLDER, output_file=OUTPUT_FILE) as tracker:
        # Make predictions
        PREDS_PATH = REPO_PATH / "predictions"
        generate_predictions(
            test_filenames = file_names,
            predictions_folder = PREDS_PATH,
            model = model,
            max_predictions = max_predictions
        )
