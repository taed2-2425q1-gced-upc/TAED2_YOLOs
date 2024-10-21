""" Prediction script for the model """
# General imports
import os
import argparse
from pathlib import Path
from cv2 import imread  # pylint: disable=E0611
import torch # pylint: disable=E0401

import numpy as np

from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO # pylint: disable=E0401
from codecarbon import EmissionsTracker # pylint: disable=E0401

from person_image_segmentation.utils.modeling_utils import generate_predictions

# Load environment variables from a .env file
load_dotenv()

# Declare the base data path
BASE_DATA_PATH = Path(os.getenv('PATH_TO_DATA_FOLDER'))
REPO_PATH = Path(os.getenv('PATH_TO_REPO'))
BEST_WEIGHTS = os.getenv('PATH_TO_BEST_WEIGHTS')
BEST_WEIGHTS_FULL_PATH = (
    str(REPO_PATH / BEST_WEIGHTS)
    if BEST_WEIGHTS != "None"
    else "yolov8m-seg.pt"
)


if __name__ == "__main__":
    # Argument parser for max number of predictions
    parser = argparse.ArgumentParser(description="Generate predictions for test images.")
    parser.add_argument('--max_predictions', type=int, default=10, help='Maximum number of predictions to generate')
    args = parser.parse_args()
    max_predictions = args.max_predictions

    # Load the YOLO model
    model = YOLO(BEST_WEIGHTS_FULL_PATH)

    # Load the test images
    test_folder = BASE_DATA_PATH / "processed/images/test"
    file_names = os.listdir(test_folder)
    file_names = [str(test_folder / file) for file in file_names if os.path.isfile(str(test_folder / file))]
    
    with EmissionsTracker(output_dir=str(REPO_PATH / "metrics"), output_file="emissions_inference.csv") as tracker:
        # Make predictions
        PREDS_PATH = REPO_PATH / "predictions"
        generate_predictions(
            test_filenames = file_names,
            predictions_folder = PREDS_PATH,
            model = model,
            max_predictions = max_predictions
        )
